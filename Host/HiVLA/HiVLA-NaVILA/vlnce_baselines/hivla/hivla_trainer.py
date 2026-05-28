"""
HiVLA Trainer for NaVILA evaluation. Runs the standard VLN-CE episode loop
with two optional research extensions: Step 1 (H_tmp head extraction on
successful episodes) and Step 2 (per-step attention metric logging via
TemporalAnomalyLogger, including action-token probability).
"""

import copy
import functools
import gc
import json
import os
import re
import time
import types
from collections import defaultdict

import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs
from habitat_extensions.utils import generate_video, observations_to_image
from PIL import Image
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

# HiVLA pipeline modules
import importlib as _il
from vlnce_baselines.hivla._runtime import compute_token_ranges, load_ranked_heads


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    """Sample and pad image frames for VLM input."""
    frames = copy.deepcopy(images)
    while len(frames) < num_frames:
        frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    return sampled_frames


@baseline_registry.register_trainer(name="hivla_trainer")
class HiVLATrainer(BaseVLNCETrainer):
    """
    HiVLA Trainer: evaluates NaVILA checkpoints with optional attention-based
    data collection (Step 1: H_tmp head extraction; Step 2: temporal anomaly
    metric logging).
    """

    def __init__(self, config=None, num_chunks=1, chunk_idx=0):
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx

        super().__init__(config)

        # Parse HiVLA pipeline config
        hivla_cfg = config.get('HIVLA', config.get('ECCV', {}))

        # --- Step 2: Data Extraction ---
        extract_data_cfg = hivla_cfg.get('DATA_EXTRACTION', {})
        self.anomaly_logging_enabled = extract_data_cfg.get('ENABLED', False)
        self.anomaly_heads_set = set()
        self.anomaly_logger = None
        self._head_importance_path = extract_data_cfg.get('TEMPORAL_HEAD_IMPORTANCE_PATH', '')
        self._data_extraction_output_dir = extract_data_cfg.get('OUTPUT_DIR', '')

        def get_head_file_path(head_type):
            head_type_map = {'h_tmp': self._head_importance_path}
            return head_type_map.get(head_type, '')

        if self.anomaly_logging_enabled:
            anomaly_head_type = extract_data_cfg.get('HEAD_TYPE', 'h_tmp')
            anomaly_head_path = get_head_file_path(anomaly_head_type)
            anomaly_ratio = extract_data_cfg.get('RATIO', 0.1)
            self.anomaly_format = extract_data_cfg.get('OUTPUT_FORMAT', 'jsonl')

            if os.path.exists(anomaly_head_path):
                anomaly_heads, _ = load_ranked_heads(anomaly_head_path, anomaly_ratio)
                self.anomaly_heads_set = set(anomaly_heads)
                logger.info(
                    f"[HiVLA] Step 2 - Data extraction: {len(anomaly_heads)} "
                    f"{anomaly_head_type} heads (top {anomaly_ratio*100:.0f}%)"
                )
            else:
                logger.warning(
                    f"[HiVLA] Step 2 - Head file not found ({anomaly_head_type}): "
                    f"{anomaly_head_path}"
                )

        # --- Step 1: H_tmp Extraction ---
        extract_cfg = hivla_cfg.get('HTMP_EXTRACTION', {})
        self.head_extraction_enabled = extract_cfg.get('ENABLED', False)
        self.extract_h_tmp = extract_cfg.get('EXTRACT_H_TMP', False) if self.head_extraction_enabled else False
        self.spl_threshold = extract_cfg.get('SPL_THRESHOLD', 1.0)
        self.max_success_episodes = extract_cfg.get('MAX_SUCCESS_EPISODES', -1)

        if self.head_extraction_enabled:
            HeadExtractor = _il.import_module("vlnce_baselines.hivla.1_htmp_extraction").HeadExtractor
            self.extractor = HeadExtractor(num_layers=32, num_heads=32, head_dim=128)
            logger.info(
                f"[HiVLA] Step 1 - Head extraction: h_tmp={self.extract_h_tmp}, "
                f"SPL>={self.spl_threshold}"
            )
        else:
            self.extractor = None

        logger.info(
            f"[HiVLA] Extraction: {self.head_extraction_enabled}, "
            f"DataExtract: {self.anomaly_logging_enabled}"
        )

        # Hook storage
        self._orig_fwds = {}
        self._attn_capture = {}

    def _make_dirs(self) -> None:
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def train(self) -> None:
        raise NotImplementedError

    def _capture_attention(self, model, input_ids, images_tensor, tokenizer, target_heads=None):
        """
        Capture attention weights via explicit per-head forward pass.

        Args:
            target_heads: set of (layer, head) tuples to capture,
                          or None to capture ALL heads (extraction mode).

        Returns:
            (frame_instr, token_ranges) where frame_instr is
            {(layer, head): np.array [num_frames, num_instr_tokens]}.
        """
        _capture_attn_forward = _il.import_module("vlnce_baselines.hivla._runtime.capture_forward")._capture_attn_forward

        with torch.no_grad():
            self._attn_capture = {}

            attn_mask = torch.ones_like(input_ids)
            (_, pos_ids, amask, _, embeds, _) = model.prepare_inputs_labels_for_multimodal(
                input_ids.clone(), None, attn_mask, None, None, images_tensor
            )

            token_ranges = compute_token_ranges(input_ids, embeds.shape[1], tokenizer)

            if token_ranges['instr_start'] is None:
                del embeds, pos_ids, amask
                return None, token_ranges

            # Determine which layers to hook
            orig_fwds = {}
            if target_heads is None:
                # All layers — extraction mode
                layers_to_hook = range(len(model.llm.model.layers))
            else:
                # Only layers containing target heads
                heads_by_layer = defaultdict(set)
                for l, h in target_heads:
                    heads_by_layer[l].add(h)
                layers_to_hook = heads_by_layer.keys()

            for layer_idx in layers_to_hook:
                layer = model.llm.model.layers[layer_idx]
                orig_fwds[layer_idx] = layer.self_attn.forward

                layer.self_attn.forward = types.MethodType(
                    functools.partial(
                        _capture_attn_forward,
                        layer_idx=layer_idx,
                        capture_dict=self._attn_capture,
                        token_ranges=token_ranges,
                        vis_heads=target_heads,
                    ),
                    layer.self_attn
                )

            try:
                model.llm.model(
                    inputs_embeds=embeds,
                    attention_mask=amask,
                    position_ids=pos_ids,
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                for layer_idx, orig_fwd in orig_fwds.items():
                    model.llm.model.layers[layer_idx].self_attn.forward = orig_fwd

            del embeds, pos_ids, amask
            gc.collect()
            torch.cuda.empty_cache()

            frame_instr = self._get_frame_instr_attention(token_ranges)
            return frame_instr, token_ranges

    # Tokens per image after vision encoder + mm_projector downsampling.
    # SigLIP 384/14 = 27×27 → pad to 28×28 → 2×2 downsample → 14×14 = 196.
    _TOKENS_PER_IMAGE = 196

    def _prepare_token_ranges_fast(self, input_ids, tokenizer):
        """
        Compute token ranges analytically — NO vision encoder call.

        Uses the fixed tokens-per-image constant (196 for NaVILA) to
        calculate expanded_seq_len from the input_ids alone.
        This avoids an extra ~130ms vision encoder call.
        """
        ids_list = input_ids[0].cpu().tolist()
        num_img_placeholders = sum(1 for t in ids_list if t == IMAGE_TOKEN_INDEX)
        expanded_seq_len = (
            len(ids_list) - num_img_placeholders
            + num_img_placeholders * self._TOKENS_PER_IMAGE
        )
        return compute_token_ranges(input_ids, expanded_seq_len, tokenizer)

    class _ActionProbCapture:
        """LogitsProcessor that captures action token probability directly on CPU.

        Called at every generation step by HuggingFace generate().
        Avoids output_scores=True (which keeps GPU tensors alive) by saving
        only a single CPU scalar via prev-step logit tracking.

        Protocol (HuggingFace greedy_search):
            logits_processor(input_ids, scores) is called BEFORE the token is
            sampled.  input_ids already contains all previously generated tokens.
            scores are the raw logits for the NEXT token.

        So when we see keyword token K at input_ids[-1], the logits that
        produced K were the scores saved from the *previous* call.
        """
        _KEYWORDS = ("forward", "left", "right", "stop")

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.captured_prob = None   # set once; Python float on CPU
            self._prev = None           # CPU tensor (vocab_size,) from last call

        def __call__(self, input_ids, scores):
            if self.captured_prob is None and self._prev is not None and input_ids.shape[1] > 0:
                last_tok = int(input_ids[0, -1])
                tok_text = self.tokenizer.decode(
                    [last_tok], skip_special_tokens=True
                ).lower()
                if any(kw in tok_text for kw in self._KEYWORDS):
                    probs = torch.softmax(self._prev.float(), dim=-1)
                    self.captured_prob = float(probs[last_tok])
            # Move current logits to CPU immediately; GPU tensor is freed by caller
            self._prev = scores[0].cpu()
            return scores  # pass through unchanged

    @staticmethod
    def _extract_action_prob(scores, output_ids, input_len, tokenizer):
        """Return softmax probability of the action-discriminating token.

        LLaVA calls self.llm.generate(inputs_embeds=...) without input_ids,
        so HuggingFace prepends a BOS token to output.sequences, making
        sequences one token longer than scores. Align from the right:
        take the last len(scores) tokens so sequences[-N:][i] == scores[i].
        """
        if scores is None or len(scores) == 0:
            return None
        import torch as _torch
        keywords = ["forward", "left", "right", "stop"]
        n = len(scores)
        new_token_ids = output_ids[0][-n:]
        for token_id, step_logits in zip(new_token_ids, scores):
            token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True).lower()
            if any(kw in token_text for kw in keywords):
                probs = _torch.softmax(step_logits[0].float(), dim=-1)
                return float(probs[token_id.item()].item())
        return None

    def _capture_attention_inline(self, model, input_ids, images_tensor,
                                  tokenizer, target_heads, output_ids_out,
                                  scores_out=None):
        """
        Capture attention during model.generate() via inline hooks.

        Instead of a separate LLM forward pass (~700ms), this installs
        lightweight wrappers on target attention layers that compute Q·K^T
        for only the specified heads during the prefill step of generate().
        Token ranges are computed analytically (no extra vision encoder call).

        Args:
            model:          LLaVA model.
            input_ids:      Input token IDs [1, seq_len].
            images_tensor:  Processed image tensor.
            tokenizer:      Tokenizer instance.
            target_heads:   set of (layer, head) tuples to capture.
            output_ids_out: list — generate() output_ids will be stored in [0].
            scores_out:     list — generate() scores tuple stored in [0] (optional).

        Returns:
            (frame_instr, token_ranges) same format as _capture_attention.
        """
        _inline = _il.import_module("vlnce_baselines.hivla._runtime.inline_capture")
        install_inline_hooks = _inline.install_inline_hooks
        remove_inline_hooks = _inline.remove_inline_hooks

        # 1) Compute token ranges analytically (~0.1ms, no vision encoder)
        token_ranges = self._prepare_token_ranges_fast(input_ids, tokenizer)

        if token_ranges is None or token_ranges.get('instr_start') is None:
            return None, token_ranges

        # 2) Install inline hooks (~0.01ms)
        self._attn_capture = {}
        orig_forwards = install_inline_hooks(
            model, target_heads, token_ranges, self._attn_capture
        )

        # 3) Run generate — hooks capture attention during prefill
        try:
            with torch.inference_mode():
                _gen_out = model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=32,
                    use_cache=True,
                    stopping_criteria=[self._current_stopping_criteria],
                    pad_token_id=self._current_tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            output_ids_out[0] = _gen_out.sequences
            if scores_out is not None:
                scores_out[0] = _gen_out.scores
        finally:
            # 4) Always remove hooks
            remove_inline_hooks(model, orig_forwards)

        # 5) Process captured attention
        frame_instr = self._get_frame_instr_attention(token_ranges)
        return frame_instr, token_ranges

    def _get_frame_instr_attention(self, token_ranges):
        """Process captured attention into [num_frames x num_instr_tokens] format."""
        if not self._attn_capture or token_ranges is None:
            return None

        num_img = token_ranges['num_img_tokens']
        num_placeholders = token_ranges['num_img_placeholders']

        # CLS token offset
        cls_offset = 0
        if num_placeholders > 0 and num_img % num_placeholders != 0:
            if (num_img - 1) % num_placeholders == 0:
                cls_offset = 1

        tokens_per_frame = (
            (num_img - cls_offset) // num_placeholders
            if num_placeholders > 0 else 0
        )

        if tokens_per_frame <= 0:
            return None

        frame_instr = {}
        for (l, h), instr_to_img in self._attn_capture.items():
            try:
                arr = instr_to_img.numpy()  # [instr_len, img_tokens]

                if cls_offset > 0 and arr.shape[1] > 0:
                    arr = arr[:, 1:]

                usable = tokens_per_frame * num_placeholders
                if arr.shape[1] < usable:
                    continue
                arr = arr[:, :usable]

                # [instr_len, num_frames, tokens_per_frame] -> mean -> transpose
                arr = arr.reshape(arr.shape[0], num_placeholders, tokens_per_frame)
                arr = arr.mean(axis=2)  # [instr_len, num_frames]
                frame_instr[(l, h)] = arr.T  # [num_frames, instr_len]
            except Exception as e:
                logger.warning(f"[HiVLA] Error processing L{l}H{h}: {e}")
                continue

        return frame_instr if frame_instr else None

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
    ) -> None:
        """Evaluates a single checkpoint."""
        logger.info(f"checkpoint_path: {checkpoint_path}")

        # Load model
        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        tokenizer, model, image_processor, context_len = load_pretrained_model(checkpoint_path, model_name)

        config = self.config.clone()
        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = self.num_chunks
        config.TASK_CONFIG.DATASET.CHUNK_IDX = self.chunk_idx
        config.RESULTS_DIR = os.path.join(
            config.RESULTS_DIR, model_name, config.TASK_CONFIG.DATASET.TYPE, config.TASK_CONFIG.DATASET.SPLIT
        )
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.VIDEO_DIR = os.path.join(config.RESULTS_DIR, "videos")
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"{split}_{self.num_chunks}-{self.chunk_idx}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observations = envs.reset()
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        stats_episodes = {}

        past_rgbs = [[] for _ in range(envs.num_envs)]
        rgb_frames = [[] for _ in range(envs.num_envs)]

        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_path}]" " [Episodes evaluated: {evaluated}/{total}]" " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        assert envs.num_envs == 1

        queue_actions = []
        step_idx = 0

        # --- Extraction state ---
        episode_extract_data = []
        analyzed_episodes = 0

        # --- Step 2: Data Extraction ---
        if self.anomaly_logging_enabled and self.anomaly_heads_set:
            TemporalAnomalyLogger = _il.import_module("vlnce_baselines.hivla.2_data_extraction").TemporalAnomalyLogger
            if self._data_extraction_output_dir:
                step2_dir = self._data_extraction_output_dir
            else:
                step2_dir = os.path.join('vlnce_baselines', 'hivla', '2_data_extraction', 'outputs', split)
            os.makedirs(step2_dir, exist_ok=True)
            anomaly_path = os.path.join(
                step2_dir,
                f'anomaly_metrics_chunk{self.chunk_idx}.jsonl'
            )
            self.anomaly_logger = TemporalAnomalyLogger(
                anomaly_path, self.anomaly_heads_set,
            )

        # ----------------------------------------------------------
        # Main Loop
        # ----------------------------------------------------------

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:

            current_episodes = envs.current_episodes()
            anomaly_frame_instr = None   # only set during VLM steps
            anomaly_action_name = 'MOVE_FORWARD'  # default; updated when VLM fires
            action_prob = None           # only set during VLM steps

            # --- Execute queued action (no VLM call) ---
            if len(queue_actions) > 0:
                outputs = envs.step([queue_actions.pop(0)])

            # --- VLM prediction ---
            else:
                with torch.no_grad():
                    curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")

                    past_and_current_rgbs = past_rgbs[0] + [curr_rgb]
                    num_video_frames = model.config.num_video_frames

                    past_and_current_rgbs = sample_and_pad_images(past_and_current_rgbs, num_frames=num_video_frames)

                    instruction = current_episodes[0].instruction.instruction_text

                    interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)

                    question = (
                        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                        f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                        f"degree, moving forward a certain distance, or stop if the task is completed."
                    )

                    conv_mode = "llama_3"
                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).to(
                        model.device, dtype=torch.float16
                    )
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    # === ALL-HEADS CAPTURE (Step 1: H_tmp Extraction) ===
                    # Extraction mode captures ALL 1024 heads via separate
                    # forward pass (hooks not efficient for all heads).
                    extract_frame_instr = None
                    need_all_heads = self.extract_h_tmp and self.extractor is not None
                    if need_all_heads:
                        try:
                            extract_frame_instr, _ = self._capture_attention(
                                model, input_ids, images_tensor.half().cuda(),
                                tokenizer, target_heads=None,
                            )
                        except Exception as e:
                            logger.warning(f"[HiVLA] All-heads capture error: {e}")

                    # === INLINE CAPTURE + GENERATION (Step 2: Data Extraction) ===
                    # Collect target heads for inline capture during model.generate().
                    # This avoids a separate forward pass (~100% overhead) by using
                    # lightweight hooks (~0.3% overhead) during prefill.
                    inline_heads = set()
                    need_anomaly_capture = (
                        self.anomaly_logger is not None
                        and extract_frame_instr is None
                    )

                    if need_anomaly_capture:
                        inline_heads.update(self.anomaly_heads_set)

                    # Prepare stopping criteria for inline capture
                    self._current_stopping_criteria = stopping_criteria
                    self._current_tokenizer = tokenizer

                    anomaly_frame_instr = None
                    _inline_capture_used = False
                    _inline_all = None
                    _gen_scores = None

                    if inline_heads and not need_all_heads:
                        _output_ids_holder = [None]
                        _scores_holder = [None]
                        try:
                            _inline_all, _inline_tr = self._capture_attention_inline(
                                model, input_ids, images_tensor.half().cuda(),
                                tokenizer, inline_heads, _output_ids_holder,
                                scores_out=_scores_holder,
                            )
                            _inline_capture_used = True
                        except Exception as e:
                            logger.warning(f"[HiVLA] Inline capture error: {e}")
                            _inline_all = None

                        if _inline_capture_used and _output_ids_holder[0] is not None:
                            output_ids = _output_ids_holder[0]
                            _gen_scores = _scores_holder[0]
                        else:
                            _inline_capture_used = False

                        # Distribute captured attention to Step 2 consumer
                        if _inline_all is not None and need_anomaly_capture:
                            anomaly_frame_instr = {
                                k: v for k, v in _inline_all.items()
                                if k in self.anomaly_heads_set
                            } or None

                    # If Step 1 captured all heads, distribute from there
                    if extract_frame_instr is not None:
                        if self.anomaly_logger is not None:
                            anomaly_frame_instr = {
                                k: v for k, v in extract_frame_instr.items()
                                if k in self.anomaly_heads_set
                            } or None

                    # === VLM Generation (if not done inline) ===
                    if not _inline_capture_used:
                        _ap_capture = self._ActionProbCapture(tokenizer)
                        with torch.inference_mode():
                            output_ids = model.generate(
                                input_ids,
                                images=images_tensor.half().cuda(),
                                do_sample=False,
                                temperature=0.0,
                                max_new_tokens=32,
                                use_cache=True,
                                stopping_criteria=[stopping_criteria],
                                pad_token_id=tokenizer.eos_token_id,
                                logits_processor=[_ap_capture],
                            )
                        _gen_scores = None  # not used in this path

                    vlm_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    if vlm_outputs.endswith(stop_str):
                        vlm_outputs = vlm_outputs[: -len(stop_str)].strip()

                    # Action parsing
                    patterns = {
                        0: re.compile(r"\bstop\b", re.IGNORECASE),
                        1: re.compile(r"\bis move forward\b", re.IGNORECASE),
                        2: re.compile(r"\bis turn left\b", re.IGNORECASE),
                        3: re.compile(r"\bis turn right\b", re.IGNORECASE),
                    }

                    def map_string_to_action(s):
                        for action, pattern in patterns.items():
                            if pattern.search(s):
                                return action
                        return None

                    actions = [map_string_to_action(vlm_outputs)]
                    if actions[0] is None:
                        actions = [1]

                    _ACTION_NAMES = {0: 'STOP', 1: 'MOVE_FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT'}
                    anomaly_action_name = _ACTION_NAMES.get(actions[0], 'MOVE_FORWARD')

                    # Extract action-token probability
                    # Non-inline path: captured on CPU by _ActionProbCapture during generate()
                    # Inline path: extract from GPU scores stored by _capture_attention_inline
                    try:
                        if not _inline_capture_used:
                            action_prob = _ap_capture.captured_prob
                        else:
                            action_prob = self._extract_action_prob(
                                _gen_scores, output_ids, input_ids.shape[1], tokenizer
                            )
                    except Exception:
                        action_prob = None
                    _gen_scores = None

                    # Action execution
                    if actions[0] == 1:  # forward
                        match = re.search(r"move forward (\d+) cm", vlm_outputs)
                        distance = int(match.group(1)) if match else 25
                        if (distance % 25) != 0:
                            distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                        outputs = envs.step([1])
                        for _ in range(int(distance // 25) - 1):
                            queue_actions.append(1)

                    elif actions[0] in (2, 3):  # turn
                        direction = "left" if actions[0] == 2 else "right"
                        match = re.search(rf"turn {direction} (\d+) degree", vlm_outputs)
                        degree = int(match.group(1)) if match else 15
                        if (degree % 15) != 0:
                            degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                        outputs = envs.step([actions[0]])
                        for _ in range(int(degree // 15) - 1):
                            queue_actions.append(actions[0])

                    else:  # 0, stop
                        outputs = envs.step(actions)

                    step_idx += 1

                    # === Store Step 1 extraction data ===
                    if self.extract_h_tmp and extract_frame_instr is not None:
                        episode_extract_data.append({
                            'frame_instr': extract_frame_instr,
                            'num_frames': num_video_frames,
                        })

                    del images_tensor, input_ids, stopping_criteria
                    try:
                        del output_ids
                    except (NameError, UnboundLocalError):
                        pass

            # --- Process step results ---
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # Step 2: log attention metrics for VLM steps
            if self.anomaly_logger is not None and anomaly_frame_instr is not None:
                ep_id_log = current_episodes[0].episode_id

                try:
                    anomaly_agent_pos = envs.call_at(0, "save_agent_state", {})
                except Exception:
                    anomaly_agent_pos = {"x": 0.0, "y": 0.0, "z": 0.0}

                self.anomaly_logger.log_step(
                    anomaly_frame_instr, ep_id_log, step_idx - 1,
                    action_name=anomaly_action_name,
                    agent_pos=anomaly_agent_pos,
                    action_prob=action_prob,
                )
                anomaly_frame_instr = None  # consumed

            for i in range(envs.num_envs):
                past_rgbs[i].append(Image.fromarray(np.uint8(batch[i]["rgb"].cpu())).convert("RGB"))

                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(frame, current_episodes[i].instruction.instruction_text)
                    # Fix frame size to match the first frame of this episode
                    if len(rgb_frames[i]) > 0:
                        target_h, target_w = rgb_frames[i][0].shape[:2]
                        fh, fw = frame.shape[:2]
                        if fh != target_h or fw != target_w:
                            if fh < target_h:
                                pad = np.zeros((target_h - fh, fw, frame.shape[2]), dtype=frame.dtype)
                                frame = np.concatenate((frame, pad), axis=0)
                            elif fh > target_h:
                                frame = frame[:target_h, :, :]
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                # --- Episode finished ---
                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]

                # Step 2: end episode
                if self.anomaly_logger is not None:
                    self.anomaly_logger.end_episode()

                torch.cuda.empty_cache()

                # Step 1: accumulate H_tmp scores for successful episodes
                if self.extract_h_tmp and self.extractor is not None and len(episode_extract_data) > 0:
                    episode_spl = infos[i].get('spl', 0.0)
                    accepted = self.extractor.accumulate_h_tmp(
                        episode_extract_data, episode_spl, self.spl_threshold
                    )
                    if accepted:
                        analyzed_episodes += 1
                        if analyzed_episodes % 10 == 0:
                            logger.info(
                                f"[HiVLA] Step 1: analyzed {analyzed_episodes} "
                                f"success episodes (SPL >= {self.spl_threshold})"
                            )
                            # Checkpoint: overwrite chunk file every 10 episodes
                            os.makedirs(config.RESULTS_DIR, exist_ok=True)
                            self.extractor.save_h_tmp_chunk(config.RESULTS_DIR, self.chunk_idx)
                episode_extract_data = []

                observations[i] = envs.reset_at(i)[0]
                past_rgbs[i] = []
                queue_actions = []
                step_idx = 0

                gc.collect()
                torch.cuda.empty_cache()

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx="0",
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (envs, batch, rgb_frames) = self._pause_envs(
                envs_to_pause, envs, batch, rgb_frames,
            )

            # Early stop when enough success episodes collected
            if self.max_success_episodes > 0 and analyzed_episodes >= self.max_success_episodes:
                logger.info(
                    f"[HiVLA] Step 1: reached {self.max_success_episodes} "
                    f"success episodes on this chunk. Stopping early."
                )
                break

        envs.close()
        if config.use_pbar:
            pbar.close()

        # --- Close Step 2 logger ---
        if self.anomaly_logger is not None:
            self.anomaly_logger.close()

        # --- Save Step 1: H_tmp Extraction results (per-chunk) ---
        if self.extractor is not None:
            HeadExtractor = _il.import_module("vlnce_baselines.hivla.1_htmp_extraction").HeadExtractor
            os.makedirs(config.RESULTS_DIR, exist_ok=True)

            if self.extract_h_tmp:
                self.extractor.save_h_tmp_chunk(config.RESULTS_DIR, self.chunk_idx)
                HeadExtractor.merge_h_tmp_chunks(
                    config.RESULTS_DIR, self.num_chunks, self.spl_threshold
                )

            logger.info(
                f"[HiVLA] Step 1 done. h_tmp episodes={analyzed_episodes}"
            )

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(stats_episodes, f, indent=4)

    @staticmethod
    def _pause_envs(envs_to_pause, envs, batch, rgb_frames=None):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (envs, batch, rgb_frames)

    def eval(self) -> None:
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu")
        )
        if "tensorboard" in self.config.VIDEO_OPTION:
            assert len(self.config.TENSORBOARD_DIR) > 0
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert len(self.config.VIDEO_DIR) > 0

        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            if os.path.isdir(self.config.EVAL_CKPT_PATH_DIR):
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                )
