# models/vla/inference.py

import sys
import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Optional
from collections import deque

from .config import (
    NAVILA_MODEL_PATH,
    NAVILA_REPO_PATH,
    NAVILA_EVAL_PATH,
    NUM_VIDEO_FRAMES,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    CONV_TYPE,
    NAV_PROMPT_TEMPLATE,
    HIVLA_TOKENS_PER_IMAGE,
)

# Add NaVILA repo path for llava imports
if NAVILA_REPO_PATH not in sys.path:
    sys.path.insert(0, NAVILA_REPO_PATH)

# Load _runtime modules directly by file path to avoid triggering
# vlnce_baselines/__init__.py which imports Habitat trainers requiring lmdb.
import importlib.util as _ilu
_runtime_dir = os.path.join(NAVILA_EVAL_PATH, "vlnce_baselines", "hivla", "_runtime")

def _load_runtime(name):
    spec = _ilu.spec_from_file_location(name, os.path.join(_runtime_dir, f"{name}.py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_inline_capture_mod = _load_runtime("inline_capture")
_token_utils_mod    = _load_runtime("token_utils")
install_inline_hooks  = _inline_capture_mod.install_inline_hooks
remove_inline_hooks   = _inline_capture_mod.remove_inline_hooks
compute_token_ranges  = _token_utils_mod.compute_token_ranges


class NaVILAInference:
    """Local NaVILA VLA inference on Jetson AGX Orin.

    Loads the VILA-based navigation model and runs inference
    to produce mid-level action strings like "move forward 75 cm".
    """

    def __init__(self, model_path: str = NAVILA_MODEL_PATH):
        self.model_path = model_path
        self.num_frames = NUM_VIDEO_FRAMES
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self._loaded = False

        # History buffer: stores past RGB frames as PIL Images
        self.frame_history: deque = deque(maxlen=200)

    def load_model(self):
        """Load NaVILA model, tokenizer, and image processor.

        Call this once during initialization. Separated from __init__
        so the caller can control when the heavy loading happens.
        """
        if self._loaded:
            return

        print(f"[NaVILA] Loading model from {self.model_path}...")
        start = time.monotonic()

        from llava.model.builder import load_pretrained_model
        print("[NaVILA] Loading model in fp16...")
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=self.model_path,
                model_name="navila-llama3-8b",
                model_base=None,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        )
        self.model.eval()

        elapsed = time.monotonic() - start
        print(f"[NaVILA] Model loaded in {elapsed:.1f}s")
        self._loaded = True

    def add_frame(self, frame: Image.Image):
        """Add a new RGB frame to the history buffer.

        Args:
            frame: PIL Image (RGB) from ZED camera.
        """
        self.frame_history.append(frame.convert("RGB"))

    def _sample_frames(self) -> list:
        """Sample frames for model input.

        Returns NUM_VIDEO_FRAMES images:
          - first N-1: uniformly sampled from history (memory)
          - last 1: most recent frame (current observation)

        If not enough history, pads with black frames.
        """
        frames = list(self.frame_history)

        if len(frames) == 0:
            return [Image.new("RGB", (384, 384), (0, 0, 0))] * self.num_frames

        if len(frames) < self.num_frames:
            while len(frames) < self.num_frames:
                frames.insert(0, Image.new("RGB", (384, 384), (0, 0, 0)))

        latest = frames[-1]
        indices = np.linspace(
            0, len(frames) - 1, num=self.num_frames - 1, endpoint=False, dtype=int
        )
        sampled = [frames[i] for i in indices] + [latest]
        return [f.convert("RGB") for f in sampled]

    def infer(self, instruction: str) -> Optional[str]:
        """Run NaVILA inference and return the raw action string.

        Samples frames internally. For thread-safe usage, prefer
        infer_with_frames() with pre-sampled frames.
        """
        return self.infer_with_frames(instruction, self._sample_frames())

    def infer_with_frames(self, instruction: str, images: list) -> Optional[str]:
        """Run NaVILA inference with pre-sampled frames.

        Args:
            instruction: Natural language navigation instruction.
            images: List of PIL Images (already sampled/snapshotted by caller).

        Returns:
            Raw model output string (e.g. "The next action is move forward 75 cm"),
            or None if inference fails.
        """
        if not self._loaded:
            print("[NaVILA] Model not loaded. Call load_model() first.")
            return None

        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX

        try:

            # 2. Build prompt
            history_tokens = "<image>\n" * (len(images) - 1)
            question = NAV_PROMPT_TEMPLATE.format(
                history_images=history_tokens,
                instruction=instruction,
            )

            conv = conv_templates[CONV_TYPE].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # 3. Process images
            images_tensor = process_images(
                images, self.image_processor, self.model.config
            ).to(self.model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.model.device)
            )

            # 4. Stopping criteria
            stop_str = (
                conv.sep
                if conv.sep_style != SeparatorStyle.TWO
                else conv.sep2
            )
            stopping_criteria = KeywordsStoppingCriteria(
                [stop_str], self.tokenizer, input_ids
            )

            # 5. Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=False,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # 6. Decode
            output = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()

            if output.endswith(stop_str):
                output = output[: -len(stop_str)]
            output = output.strip()

            print(f"[NaVILA] Output: {output}")
            return output

        except Exception as e:
            print(f"[NaVILA] Inference error: {e}")
            return None

    def infer_with_frames_and_attention(
        self, instruction: str, images: list, target_heads: list
    ):
        """Run inference and capture attention for HiVLA replanning.

        Same as infer_with_frames(), but additionally installs inline hooks
        on target_heads to capture Q·K^T attention during the prefill step.

        Args:
            instruction: Natural language navigation instruction.
            images:       Pre-sampled list of PIL Images.
            target_heads: List of (layer_idx, head_idx) tuples to capture.

        Returns:
            (output_str, frame_instr) where frame_instr is
            {(layer, head): np.ndarray [num_frames, num_instr_tokens]},
            or (output_str, None) if attention capture fails.
        """
        if not self._loaded:
            print("[NaVILA] Model not loaded. Call load_model() first.")
            return None, None

        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX

        # Build prompt
        history_tokens = "<image>\n" * (len(images) - 1)
        question = NAV_PROMPT_TEMPLATE.format(
            history_images=history_tokens,
            instruction=instruction,
        )
        conv = conv_templates[CONV_TYPE].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process images
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        # Analytically compute token ranges (no extra vision encoder call)
        ids_list = input_ids[0].cpu().tolist()
        num_placeholders = sum(1 for t in ids_list if t == IMAGE_TOKEN_INDEX)
        expanded_seq_len = (
            len(ids_list) - num_placeholders
            + num_placeholders * HIVLA_TOKENS_PER_IMAGE
        )
        token_ranges = compute_token_ranges(input_ids, expanded_seq_len, self.tokenizer)

        # Install inline hooks if token ranges are valid
        capture_dict = {}
        orig_forwards = {}
        if token_ranges.get('instr_start') is not None and target_heads:
            orig_forwards = install_inline_hooks(
                self.model, target_heads, token_ranges, capture_dict
            )

        # Stopping criteria
        stop_str = (
            conv.sep
            if conv.sep_style != SeparatorStyle.TWO
            else conv.sep2
        )
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        # Generate (hooks capture attention during prefill)
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=False,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if orig_forwards:
                remove_inline_hooks(self.model, orig_forwards)

        # Decode output
        output = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)]
        output = output.strip()

        # Convert capture_dict to [num_frames, num_instr_tokens] format
        frame_instr = self._build_frame_instr(capture_dict, token_ranges)

        print(f"[NaVILA] Output: {output}")
        return output, frame_instr

    def _build_frame_instr(self, capture_dict, token_ranges):
        """Convert raw capture_dict to {(l,h): np.ndarray [F, N]} format.

        capture_dict values are tensors of shape [num_instr_tokens, num_img_tokens].
        This reshapes them to [num_frames, num_instr_tokens] by averaging over
        each frame's image tokens (mirrors hivla_trainer._get_frame_instr_attention).
        """
        if not capture_dict or token_ranges is None:
            return None

        num_img = token_ranges['num_img_tokens']
        num_placeholders = token_ranges['num_img_placeholders']
        if num_placeholders == 0:
            return None

        # Handle CLS token offset (when num_img is not evenly divisible)
        cls_offset = 0
        if num_img % num_placeholders != 0:
            if (num_img - 1) % num_placeholders == 0:
                cls_offset = 1

        tokens_per_frame = (num_img - cls_offset) // num_placeholders
        if tokens_per_frame <= 0:
            return None

        frame_instr = {}
        for (l, h), instr_to_img in capture_dict.items():
            try:
                arr = instr_to_img.numpy()  # [instr_len, img_tokens]
                if cls_offset > 0 and arr.shape[1] > 0:
                    arr = arr[:, 1:]
                usable = tokens_per_frame * num_placeholders
                if arr.shape[1] < usable:
                    continue
                arr = arr[:, :usable]
                # [instr_len, num_frames, tokens_per_frame] → mean → transpose
                arr = arr.reshape(arr.shape[0], num_placeholders, tokens_per_frame)
                arr = arr.mean(axis=2)          # [instr_len, num_frames]
                frame_instr[(l, h)] = arr.T     # [num_frames, instr_len]
            except Exception as e:
                print(f"[NaVILA] Attention reshape error L{l}H{h}: {e}")
                continue

        return frame_instr if frame_instr else None
