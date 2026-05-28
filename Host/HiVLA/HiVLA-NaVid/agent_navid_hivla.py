"""
NaVid-HiVLA Agent — path deviation detection generalizability experiment.

Ports the HiVLA pipeline (originally built for NaVILA) to NaVid to test
whether the same I_diag-based approach generalizes across VLN models.

Pipeline stages supported:
  Step 2  DATA EXTRACTION   Capture attention at each VLM step → JSONL metrics
  Step 1  H_TMP EXTRACTION  Compute diagonal-shifting temporal score → head ranking

Downstream analysis (Steps 3–5) runs on the JSONL output using the same scripts
as NaVILA (model-agnostic statistics):
  Step 3  GT labeling         hivla/3_gt_labeling/waypoint_gt_labeling.py
  Step 4  H_nav head ranking  hivla/4_hnav_selection/hnav_selection.py
  Step 5  Grid search         hivla/5_grid_search/grid_search.py

Usage (data extraction mode):
    agent = NaVidHiVLA_Agent(
        model_path="model_zoo/navid-7b-...",
        result_path="tmp/results",
        hivla_config={
            "DATA_EXTRACTION": {
                "ENABLED": True,
                "OUTPUT_PATH": "hivla/2_data_extraction/outputs/anomaly_metrics.jsonl",
                "HEAD_TYPE": "all",   # "all" → log all heads; or pass path to h_tmp JSON
            },
            "HTMP_EXTRACTION": {
                "ENABLED": False,
            },
        },
    )

Key differences from NaVILA:
  - model.model.layers  (not model.llm.model.layers)
  - NAV_SIZE = 4 tokens/frame, 1 separator between frames (Grid:2)
  - Prompt uses single quotes around instruction (not double)
  - n_frames grows over the episode (all frames accumulated, not fixed 8)
"""

import functools
import gc
import importlib as _il
import json
import os
import re
import types
from collections import defaultdict

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from habitat.core.agent import Agent
from habitat.utils.visualizations import maps

from navid.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)
from navid.conversation import conv_templates, SeparatorStyle
from navid.model.builder import load_pretrained_model
from navid.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from hivla._runtime.token_utils import (
    prepare_token_ranges_fast,
    get_frame_instr_attention,
)
from hivla._runtime.inline_capture import install_inline_hooks, remove_inline_hooks
from hivla._runtime.head_selection import load_ranked_heads


# ---------------------------------------------------------------------------
# Action name helpers
# ---------------------------------------------------------------------------

_ACTION_ID_TO_NAME = {
    0: 'STOP',
    1: 'MOVE_FORWARD',
    2: 'TURN_LEFT',
    3: 'TURN_RIGHT',
}


class _ActionProbCapture:
    """LogitsProcessor that captures the softmax probability of the action token.

    Intercepts HuggingFace generation at each step. When an action keyword
    (forward/left/right/stop) appears as the last generated token, records
    the probability from the *previous* step's logits (which produced that token).
    Uses CPU scalars only — no GPU tensors retained after each call.
    """
    _KEYWORDS = ("forward", "left", "right", "stop")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.captured_prob = None
        self._prev = None

    def __call__(self, input_ids, scores):
        if self.captured_prob is None and self._prev is not None:
            last_tok = int(input_ids[0, -1])
            tok_text = self.tokenizer.decode([last_tok], skip_special_tokens=True).lower()
            if any(kw in tok_text for kw in self._KEYWORDS):
                probs = torch.softmax(self._prev.float(), dim=-1)
                self.captured_prob = float(probs[last_tok])
        self._prev = scores[0].cpu()
        return scores


def _action_text_to_id(text: str):
    """Map NaVid output text to (action_id, magnitude)."""
    t = text.lower()
    if 'stop' in t:
        return 0, None
    elif 'forward' in t:
        m = re.search(r'-?\d+', t)
        return 1, float(m.group()) if m else None
    elif 'left' in t:
        m = re.search(r'-?\d+', t)
        return 2, float(m.group()) if m else None
    elif 'right' in t:
        m = re.search(r'-?\d+', t)
        return 3, float(m.group()) if m else None
    return None, None


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class NaVidHiVLA_Agent(Agent):
    """
    NaVid agent with HiVLA path deviation detection pipeline.

    hivla_config keys:
      DATA_EXTRACTION:
        ENABLED       bool    – capture attention & log to JSONL
        OUTPUT_PATH   str     – where to write anomaly_metrics_chunk0.jsonl
        HEAD_PATH     str     – path to temporal_head_importance.json (optional;
                                 None = log ALL heads for offline analysis)
        HEAD_RATIO    float   – top-ratio to load from head JSON (default 0.5)
      HTMP_EXTRACTION:
        ENABLED       bool    – compute h_tmp (I_diag) scores over episode
        SPL_THRESHOLD float   – only accumulate episodes with SPL >= threshold
        OUTPUT_DIR    str     – where to save temporal_head_raw_chunk0.json
    """

    PROMPT_TEMPLATE = (
        "Imagine you are a robot programmed for navigation tasks. "
        "You have been given a video of historical observations and an image "
        "of the current observation <image>. Your assigned task is: '{}'. "
        "Analyze this series of images to decide your next move, which could "
        "involve turning left or right by a specific degree or moving forward "
        "a certain distance."
    )

    def __init__(self, model_path: str, result_path: str, hivla_config: dict = None):
        self.result_path  = result_path
        self.conv_mode    = "vicuna_v1"
        self._hivla_cfg   = hivla_config or {}

        os.makedirs(result_path, exist_ok=True)

        # Load model
        self.model_name = get_model_name_from_path(model_path)
        (self.tokenizer, self.model,
         self.image_processor, self.context_len) = load_pretrained_model(
            model_path, None, self.model_name
        )

        print("[NaVidHiVLA] Model loaded.")

        # Internal state
        self.history_rgb_tensor = None
        self.rgb_list           = []
        self.pending_action_list = []
        self.episode_id         = None
        self._vlm_step          = 0   # VLM calls per episode (not pending-action steps)

        # ---- Step 2: Data Extraction ----
        extract_cfg = self._hivla_cfg.get('DATA_EXTRACTION', {})
        self.data_extraction_enabled = extract_cfg.get('ENABLED', False)
        self.anomaly_logger    = None
        self.anomaly_heads_set = set()  # empty = log ALL heads

        if self.data_extraction_enabled:
            head_path  = extract_cfg.get('HEAD_PATH', '')
            head_ratio = float(extract_cfg.get('HEAD_RATIO', 0.5))
            if head_path and os.path.exists(head_path):
                heads, _ = load_ranked_heads(head_path, head_ratio)
                self.anomaly_heads_set = set(heads)
                print(f"[NaVidHiVLA] Data extraction: {len(heads)} target heads")
            else:
                print("[NaVidHiVLA] Data extraction: ALL heads (no head file provided)")

            output_path = extract_cfg.get(
                'OUTPUT_PATH',
                os.path.join(result_path, 'anomaly_metrics_chunk0.jsonl')
            )
            TemporalAnomalyLogger = _il.import_module(
                'hivla.2_data_extraction'
            ).TemporalAnomalyLogger
            self.anomaly_logger = TemporalAnomalyLogger(
                output_path=output_path,
                h_tmp_heads=self.anomaly_heads_set if self.anomaly_heads_set else None,
            )
            print(f"[NaVidHiVLA] Logging to: {output_path}")

        # ---- Step 1: H_tmp Extraction ----
        htmp_cfg = self._hivla_cfg.get('HTMP_EXTRACTION', {})
        self.htmp_extraction_enabled = htmp_cfg.get('ENABLED', False)
        self.extractor       = None
        self._spl_threshold  = float(htmp_cfg.get('SPL_THRESHOLD', 1.0))
        self._htmp_output_dir = htmp_cfg.get(
            'OUTPUT_DIR',
            os.path.join(result_path, 'htmp_outputs')
        )
        self._episode_htmp_steps = []  # per-episode accumulator

        if self.htmp_extraction_enabled:
            HeadExtractor = _il.import_module('hivla.1_htmp_extraction').HeadExtractor
            self.extractor = HeadExtractor(num_layers=32, num_heads=32, head_dim=128)
            os.makedirs(self._htmp_output_dir, exist_ok=True)
            print(f"[NaVidHiVLA] H_tmp extraction enabled (SPL>={self._spl_threshold})")

        # Internal attention capture dict (reused per step)
        self._attn_capture: dict = {}

    # -----------------------------------------------------------------------
    # Image processing (same as NaVid_Agent)
    # -----------------------------------------------------------------------

    def _process_images(self):
        start_idx = 0 if self.history_rgb_tensor is None else self.history_rgb_tensor.shape[0]
        batch = np.asarray(self.rgb_list[start_idx:])
        video = self.image_processor.preprocess(
            batch, return_tensors='pt'
        )['pixel_values'].half().cuda()

        self.history_rgb_tensor = (
            video if self.history_rgb_tensor is None
            else torch.cat((self.history_rgb_tensor, video), dim=0)
        )
        return [self.history_rgb_tensor]

    # -----------------------------------------------------------------------
    # Input ID construction (matches NaVid_Agent.predict_inference)
    # -----------------------------------------------------------------------

    def _build_input_ids(self, prompt: str):
        """Build input_ids with NaVid special tokens. Returns (input_ids, question_str)."""
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START = "<video_special>"
        VIDEO_END   = "</video_special>"
        IMG_START   = "<image_special>"
        IMG_END     = "</image_special>"
        NAV_TOKEN   = "[Navigation]"
        IMG_SEP     = "<image_sep>"

        tok = self.tokenizer
        vs_tok  = tok(VIDEO_START, return_tensors="pt").input_ids[0][1:].cuda()
        ve_tok  = tok(VIDEO_END,   return_tensors="pt").input_ids[0][1:].cuda()
        is_tok  = tok(IMG_START,   return_tensors="pt").input_ids[0][1:].cuda()
        ie_tok  = tok(IMG_END,     return_tensors="pt").input_ids[0][1:].cuda()
        nav_tok = tok(NAV_TOKEN,   return_tensors="pt").input_ids[0][1:].cuda()
        sep_tok = tok(IMG_SEP,     return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_str = conv.get_prompt()

        token_prompt = tokenizer_image_token(
            prompt_str, tok, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).cuda()

        indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
        new_list = []
        while indices.numel() > 0:
            idx = indices[0]
            new_list.append(token_prompt[:idx])
            new_list.append(vs_tok)
            new_list.append(sep_tok)
            new_list.append(token_prompt[idx:idx + 1])  # keep the -200
            new_list.append(ve_tok)
            new_list.append(is_tok)
            new_list.append(ie_tok)
            new_list.append(nav_tok)
            token_prompt = token_prompt[idx + 1:]
            indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)

        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)
        return input_ids, question, prompt_str

    # -----------------------------------------------------------------------
    # Core predict with optional attention capture
    # -----------------------------------------------------------------------

    def _predict(self, prompt: str, agent_pos: dict = None):
        """
        Generate next action, optionally capturing attention for HiVLA pipeline.

        Returns:
            output_text (str)  — raw VLM output.
        """
        n_frames = len(self.rgb_list)
        imgs     = self._process_images()

        input_ids, question, prompt_str = self._build_input_ids(prompt)

        stop_str = (
            conv_templates[self.conv_mode].sep
            if conv_templates[self.conv_mode].sep_style != SeparatorStyle.TWO
            else conv_templates[self.conv_mode].sep2
        )
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        need_capture = (
            (self.data_extraction_enabled and self.anomaly_logger is not None)
            or self.htmp_extraction_enabled
        )

        orig_forwards = {}
        if need_capture:
            # nav_or_not_size=64: NaVid always runs in navigation mode (PROMPT_TEMPLATE
            # contains NAVIGATION_IDENTIFIER), so navid_arch inserts 64 extra tokens
            # between is_tok and ie_tok in the expanded sequence.
            token_ranges = prepare_token_ranges_fast(input_ids, n_frames, self.tokenizer, nav_or_not_size=64)
            self._attn_capture = {}

            if token_ranges.get('instr_start') is not None:
                # Decide which heads to hook
                if self.htmp_extraction_enabled:
                    # Full capture: all 1024 heads (expensive; only for h_tmp extraction)
                    target_heads = set(
                        (l, h) for l in range(32) for h in range(32)
                    )
                else:
                    # Inline capture: only the target heads
                    target_heads = self.anomaly_heads_set or set(
                        (l, h) for l in range(32) for h in range(32)
                    )

                orig_forwards = install_inline_hooks(
                    self.model, target_heads, token_ranges, self._attn_capture
                )

        action_prob_capture = _ActionProbCapture(self.tokenizer)
        with torch.inference_mode():
            self.model.update_prompt([[question]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                logits_processor=[action_prob_capture],
            )

        if orig_forwards:
            remove_inline_hooks(self.model, orig_forwards)

        # Process captured attention
        if need_capture and self._attn_capture and token_ranges.get('instr_start') is not None:
            frame_instr = get_frame_instr_attention(self._attn_capture, token_ranges)

            # Step 2: Log to anomaly metrics JSONL
            if self.data_extraction_enabled and self.anomaly_logger is not None and frame_instr:
                action_id, _ = _action_text_to_id(
                    self.tokenizer.batch_decode(
                        output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
                    )[0].strip()
                )
                action_name = _ACTION_ID_TO_NAME.get(
                    action_id if action_id is not None else 1, 'MOVE_FORWARD'
                )
                self.anomaly_logger.log_step(
                    frame_instr,
                    episode_id=self.episode_id,
                    step=self._vlm_step,
                    action_name=action_name,
                    agent_pos=agent_pos,
                    action_prob=action_prob_capture.captured_prob,
                )

            # Step 1: Accumulate h_tmp scores
            if self.htmp_extraction_enabled and frame_instr:
                self._episode_htmp_steps.append({
                    'frame_instr': frame_instr,
                    'num_frames': n_frames,
                })

        # Decode output
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        self._vlm_step += 1
        return outputs

    # -----------------------------------------------------------------------
    # Action parsing (same as NaVid_Agent)
    # -----------------------------------------------------------------------

    def _parse_action(self, output: str):
        """Return (action_id, magnitude) from VLM output text."""
        return _action_text_to_id(output)

    # -----------------------------------------------------------------------
    # Agent interface
    # -----------------------------------------------------------------------

    def act(self, observations: dict, info: dict, episode_id: str) -> dict:
        self.episode_id = episode_id
        rgb = observations['rgb']
        self.rgb_list.append(rgb)

        # Execute pending actions without a new VLM call
        if self.pending_action_list:
            return {'action': self.pending_action_list.pop(0)}

        # Build agent position dict (if available from info)
        agent_pos = None
        if 'agent_position' in info:
            p = info['agent_position']
            agent_pos = {'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}

        instruction = observations['instruction']['text']
        nav_prompt  = self.PROMPT_TEMPLATE.format(instruction)
        output      = self._predict(nav_prompt, agent_pos=agent_pos)

        action_id, magnitude = self._parse_action(output[:-1] if output else output)

        if action_id == 0:
            self.pending_action_list.append(0)
        elif action_id == 1 and magnitude is not None:
            for _ in range(min(3, int(magnitude / 25))):
                self.pending_action_list.append(1)
        elif action_id == 2 and magnitude is not None:
            for _ in range(min(3, int(magnitude / 30))):
                self.pending_action_list.append(2)
        elif action_id == 3 and magnitude is not None:
            for _ in range(min(3, int(magnitude / 30))):
                self.pending_action_list.append(3)

        if not self.pending_action_list:
            import random
            self.pending_action_list.append(random.randint(1, 3))

        return {'action': self.pending_action_list.pop(0)}

    def reset(self):
        """Reset per-episode state."""
        self.history_rgb_tensor  = None
        self.rgb_list            = []
        self.pending_action_list = []
        self._vlm_step           = 0
        self._attn_capture       = {}

        # Flush anomaly logger buffer (end of episode)
        if self.anomaly_logger is not None:
            self.anomaly_logger.end_episode()

        # Reset h_tmp episode accumulator (caller must flush via finish_episode)
        self._episode_htmp_steps = []

    def finish_episode(self, spl: float = 0.0):
        """
        Called after each episode to accumulate h_tmp scores if enabled.

        Args:
            spl: Episode SPL (only episodes with SPL >= _spl_threshold are used).
        """
        if self.htmp_extraction_enabled and self.extractor is not None:
            self.extractor.accumulate_h_tmp(
                self._episode_htmp_steps, spl, self._spl_threshold
            )

    def save_htmp_chunk(self, chunk_idx: int = 0):
        """Save h_tmp raw scores to disk after all episodes."""
        if self.htmp_extraction_enabled and self.extractor is not None:
            return self.extractor.save_h_tmp_chunk(self._htmp_output_dir, chunk_idx)
        return None

    def close(self):
        """Flush and close all loggers."""
        if self.anomaly_logger is not None:
            self.anomaly_logger.close()
            self.anomaly_logger = None
        gc.collect()
        torch.cuda.empty_cache()
