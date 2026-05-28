"""
Token range computation utilities for NaVid.

NaVid-specific constants vs NaVILA:
  - IMAGE_TOKEN_INDEX = -200 (same)
  - nav_size = 4 tokens per video frame (Grid:2 → 8×8 pool on 16×16 → 4 per frame)
  - Separator between frames: 1 token (image_sep embedding re-inserted between frames)
  - Total image tokens for N frames: N×4 + (N-1) = 5N − 1
  - Prompt wraps instruction in single quotes: '{}' (not double quotes like NaVILA)
  - Prompt anchors:
      prefix = ". Your assigned task is: '"   (single quote)
      suffix = "'. Analyze this series of images"

Navigation mode offset (IMPORTANT):
  NaVid's PROMPT_TEMPLATE always triggers navigation mode (contains NAVIGATION_IDENTIFIER).
  In navigation mode, navid_arch.py inserts 64 extra nav_or_not tokens between
  is_tok (position P+2) and ie_tok (position P+3) in the expanded sequence.
  All tokens at input_ids positions > P+2 (including the instruction text) are
  shifted by an additional 64 positions relative to the plain image-expansion offset.
  Pass nav_or_not_size=64 (the default for Grid:2 navigation mode).
"""

import numpy as np
from navid.constants import IMAGE_TOKEN_INDEX

# Tokens per video frame after Grid:2 compression (8×8 avg-pool → 4 tokens per frame)
NAV_SIZE = 4

# Navigation mode: last frame is also encoded at full resolution (8×8 grid = 64 tokens)
# and inserted between is_tok and ie_tok in the expanded sequence.
NAV_OR_NOT_SIZE = 64


def compute_token_ranges_navid(input_ids, n_frames, tokenizer, nav_or_not_size=NAV_OR_NOT_SIZE):
    """
    Compute image and instruction token ranges for NaVid's video format.

    NaVid expands a single IMAGE_TOKEN_INDEX placeholder into:
        [frame0 × NAV_SIZE] [sep] [frame1 × NAV_SIZE] [sep] ... [frameN-1 × NAV_SIZE]
    Total tokens in image range: N × NAV_SIZE + (N-1) = 5N − 1

    In navigation mode (always active for NaVid's PROMPT_TEMPLATE), an additional
    nav_or_not_size tokens are inserted between is_tok and ie_tok after the visual
    tokens. These shift all instruction tokens by nav_or_not_size extra positions.

    Args:
        input_ids:        Input token IDs tensor [1, seq_len] (after special-token assembly).
        n_frames:         Number of accumulated video frames (= len(agent.rgb_list)).
        tokenizer:        Tokenizer instance.
        nav_or_not_size:  Extra tokens inserted in navigation mode (64 for Grid:2).
                          Pass 0 only if navigation mode is confirmed to be inactive.

    Returns:
        dict with keys:
          img_start, img_end         – image token range in expanded sequence
          num_img_tokens             – total tokens in image range (5N-1)
          n_frames                   – frame count
          instr_start, instr_end     – instruction token range (or None if not found)
          instr_token_labels         – list of decoded instruction token strings
    """
    ids_list = input_ids[0].cpu().tolist()

    img_positions = [i for i, t in enumerate(ids_list) if t == IMAGE_TOKEN_INDEX]
    if not img_positions:
        return {
            'img_start': None, 'img_end': None,
            'num_img_tokens': 0, 'n_frames': n_frames,
            'instr_start': None, 'instr_end': None,
            'instr_token_labels': [],
        }

    first_img_pos = img_positions[0]
    num_img_placeholders = len(img_positions)

    # NaVid video expansion: N frames × 4 tokens + (N-1) separator tokens
    n_sep = max(0, n_frames - 1)
    num_img_tokens = n_frames * NAV_SIZE + n_sep  # = 5*N - 1 for N >= 1

    img_start = first_img_pos
    img_end = first_img_pos + num_img_tokens

    # Offset: image expansion shifts all post-image tokens by (5N-2).
    # Additionally, nav_or_not_size tokens are inserted between is_tok and ie_tok
    # (both of which are at positions P+2 and P+3 in input_ids).
    # Instruction tokens are always at P+3 or beyond, so they pick up both shifts.
    offset = num_img_tokens - num_img_placeholders + nav_or_not_size

    # Anchor-based instruction detection
    # NaVid prompt: "...Your assigned task is: '{}'. Analyze this series of images..."
    prefix_text = ". Your assigned task is: '"   # ends with opening single quote
    suffix_text = "'. Analyze this series of images"  # starts with closing single quote

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

    p_anchor = prefix_ids[-6:] if len(prefix_ids) > 6 else prefix_ids
    s_anchor = suffix_ids[:8]  if len(suffix_ids) > 8  else suffix_ids

    def find_subseq(seq, subseq, start=0):
        n = len(subseq)
        for i in range(start, len(seq) - n + 1):
            if seq[i:i + n] == subseq:
                return i
        return -1

    # Search for instruction anchors AFTER the image placeholder
    last_img_idx = max(i for i, t in enumerate(ids_list) if t == IMAGE_TOKEN_INDEX)
    search_start = last_img_idx + 1

    instr_start = instr_end = None
    instr_token_labels = []

    p_idx = find_subseq(ids_list, p_anchor, search_start)
    if p_idx != -1:
        instr_start_orig = p_idx + len(p_anchor)
        s_idx = find_subseq(ids_list, s_anchor, instr_start_orig)
        if s_idx != -1:
            instr_end_orig = s_idx  # suffix begins with closing quote — exclude it

            instr_tokens = ids_list[instr_start_orig:instr_end_orig]

            # Apply offset from image expansion to get positions in expanded sequence
            instr_start = instr_start_orig + offset
            instr_end = instr_end_orig + offset

            for tid in instr_tokens:
                decoded = tokenizer.decode([tid]).strip()
                instr_token_labels.append(decoded if decoded else "_")

    return {
        'img_start': img_start,
        'img_end': img_end,
        'num_img_tokens': num_img_tokens,
        'num_img_placeholders': num_img_placeholders,
        'n_frames': n_frames,
        'instr_start': instr_start,
        'instr_end': instr_end,
        'instr_token_labels': instr_token_labels,
    }


def prepare_token_ranges_fast(input_ids, n_frames, tokenizer, nav_or_not_size=NAV_OR_NOT_SIZE):
    """
    Fast token range computation: no vision encoder call needed.

    Computes expanded_seq_len analytically from n_frames and NAV_SIZE,
    then delegates to compute_token_ranges_navid.

    Args:
        input_ids:        [1, seq_len] token IDs (before model expansion).
        n_frames:         len(agent.rgb_list) at current step.
        tokenizer:        Tokenizer instance.
        nav_or_not_size:  Extra nav tokens in navigation mode (64 for Grid:2).

    Returns:
        Same dict as compute_token_ranges_navid.
    """
    return compute_token_ranges_navid(input_ids, n_frames, tokenizer, nav_or_not_size)


def get_frame_instr_attention(attn_capture, token_ranges):
    """
    Convert raw captured attention into per-frame format [N, instr_len].

    NaVid's image region contains N frames × 4 tokens interleaved with
    (N-1) separator tokens. Frame k occupies columns [k*5 : k*5+4] in the
    image-region slice; separator at [k*5+4] is skipped.

    Args:
        attn_capture: dict {(layer, head): Tensor[instr_len, img_tokens]}
                      — raw instr→img attention captured by inline hooks.
        token_ranges: dict from compute_token_ranges_navid.

    Returns:
        dict {(layer, head): np.array[N_frames, instr_len]}  or  None.
    """
    if not attn_capture or token_ranges is None:
        return None

    n_frames = token_ranges.get('n_frames', 0)
    if n_frames <= 0:
        return None

    frame_instr = {}
    for (l, h), instr_to_img in attn_capture.items():
        try:
            arr = instr_to_img  # Tensor or ndarray [instr_len, img_tokens]
            if hasattr(arr, 'numpy'):
                arr = arr.numpy()
            arr = arr.astype(np.float32)  # [instr_len, 5N-1]

            per_frame = []
            for k in range(n_frames):
                col_start = k * 5
                col_end = col_start + NAV_SIZE  # 4 tokens per frame
                if col_end > arr.shape[1]:
                    break
                frame_cols = arr[:, col_start:col_end]  # [instr_len, 4]
                per_frame.append(frame_cols.mean(axis=1))  # [instr_len]

            if len(per_frame) == 0:
                continue

            frame_instr_arr = np.stack(per_frame, axis=0)  # [N, instr_len]
            frame_instr[(l, h)] = frame_instr_arr

        except Exception:
            continue

    return frame_instr if frame_instr else None
