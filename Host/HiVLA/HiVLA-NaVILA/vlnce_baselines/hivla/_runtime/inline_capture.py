"""
Inline attention capture via forward wrappers.

Instead of running a SEPARATE LLM forward pass to capture attention weights
(~700ms overhead), this module wraps self_attn.forward on target layers to
compute Q·K^T for only the target heads DURING model.generate().

How it works:
  1. The original Flash Attention forward runs normally (unchanged output).
  2. During the PREFILL step (seq_len > 1), the wrapper additionally
     computes Q·K^T for only the K target heads (e.g., 3 out of 1024).
  3. Only the sliced q_proj/k_proj weights for target heads are used,
     so the overhead is ~0.3% of a full forward pass.
  4. Captured attention is stored in a dict for post-processing.

This reduces capture overhead from ~700ms to ~2-5ms.
"""

import functools
import types

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def install_inline_hooks(model, target_heads, token_ranges, capture_dict):
    """
    Install lightweight attention capture wrappers on target layers.

    Args:
        model:        LLaVA model (model.llm.model.layers accessible).
        target_heads: set/list of (layer_idx, head_idx) tuples to capture.
        token_ranges: dict with instr_start/end, img_start/end keys.
        capture_dict: dict to store captured attention {(l,h): tensor}.

    Returns:
        orig_forwards: dict {layer_idx: original_forward} for restoration.
    """
    if not target_heads or token_ranges is None:
        return {}

    if token_ranges.get('instr_start') is None:
        return {}

    # Group heads by layer
    heads_by_layer = {}
    for l, h in target_heads:
        heads_by_layer.setdefault(l, []).append(h)

    orig_forwards = {}

    for layer_idx, heads in heads_by_layer.items():
        layer = model.llm.model.layers[layer_idx]
        attn = layer.self_attn
        orig_forwards[layer_idx] = attn.forward

        wrapper = _create_capture_wrapper(
            original_forward=attn.forward,
            layer_idx=layer_idx,
            target_heads=heads,
            token_ranges=token_ranges,
            capture_dict=capture_dict,
            attn_module=attn,
        )

        attn.forward = types.MethodType(wrapper, attn)

    return orig_forwards


def remove_inline_hooks(model, orig_forwards):
    """Restore original forward methods."""
    for layer_idx, orig_fwd in orig_forwards.items():
        model.llm.model.layers[layer_idx].self_attn.forward = orig_fwd


def _create_capture_wrapper(
    original_forward, layer_idx, target_heads,
    token_ranges, capture_dict, attn_module,
):
    """
    Create a closure that wraps the original Flash Attention forward.

    During prefill (first call with seq_len > 1), it additionally computes
    Q·K^T attention weights for only the specified target heads using
    sliced projection weights.  The original output is returned unchanged.
    """
    num_heads = attn_module.num_heads
    num_kv_heads = attn_module.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = attn_module.head_dim
    rotary_emb = attn_module.rotary_emb
    q_proj = attn_module.q_proj
    k_proj = attn_module.k_proj

    # Pre-slice projection weights for target heads (no copy — just views)
    q_weight_slices = {}
    q_bias_slices = {}
    k_kv_heads_needed = set()

    for h in target_heads:
        q_weight_slices[h] = q_proj.weight[h * head_dim : (h + 1) * head_dim]
        if q_proj.bias is not None:
            q_bias_slices[h] = q_proj.bias[h * head_dim : (h + 1) * head_dim]
        else:
            q_bias_slices[h] = None
        kv_h = h // num_kv_groups
        k_kv_heads_needed.add(kv_h)

    k_weight_slices = {}
    k_bias_slices = {}
    for kv_h in k_kv_heads_needed:
        k_weight_slices[kv_h] = k_proj.weight[kv_h * head_dim : (kv_h + 1) * head_dim]
        if k_proj.bias is not None:
            k_bias_slices[kv_h] = k_proj.bias[kv_h * head_dim : (kv_h + 1) * head_dim]
        else:
            k_bias_slices[kv_h] = None

    # Mutable flag — only capture once (during prefill)
    captured = [False]

    instr_s = token_ranges['instr_start']
    instr_e = token_ranges['instr_end']
    img_s = token_ranges['img_start']
    img_e = token_ranges['img_end']
    scale = 1.0 / (head_dim ** 0.5)

    def wrapper(self_attn, hidden_states, attention_mask=None,
                position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, **kwargs):
        # ── 1) Call original Flash Attention forward (unchanged) ──
        output = original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # ── 2) Only capture during prefill (first forward, seq_len > 1) ──
        bsz, q_len, _ = hidden_states.size()
        if captured[0] or q_len <= 1:
            return output
        captured[0] = True

        # ── 3) Lightweight Q·K^T for target heads only ──
        with torch.no_grad():
            # Compute K for needed KV heads (shared across query heads via GQA)
            k_states = {}
            for kv_h in k_kv_heads_needed:
                k_h = F.linear(hidden_states, k_weight_slices[kv_h], k_bias_slices[kv_h])
                k_h = k_h.unsqueeze(1)  # [B, 1, S, head_dim]
                k_states[kv_h] = k_h

            # RoPE: compute cos/sin once, reuse for all heads
            dummy = next(iter(k_states.values()))
            cos, sin = rotary_emb(dummy, position_ids)

            # Apply RoPE to K states
            for kv_h in k_states:
                _, k_states[kv_h] = apply_rotary_pos_emb(
                    k_states[kv_h], k_states[kv_h], cos, sin
                )

            # Causal mask (upper-triangular, shared across heads)
            causal_mask = torch.triu(
                torch.full((q_len, q_len), float('-inf'),
                           device=hidden_states.device, dtype=hidden_states.dtype),
                diagonal=1,
            )

            # Per-head Q·K^T
            for h in target_heads:
                kv_h = h // num_kv_groups

                # Q projection for this head
                q_h = F.linear(hidden_states, q_weight_slices[h], q_bias_slices[h])
                q_h = q_h.unsqueeze(1)  # [B, 1, S, head_dim]
                q_h, _ = apply_rotary_pos_emb(q_h, q_h, cos, sin)

                # Attention scores: [B, 1, S, S]
                attn_scores = torch.matmul(q_h, k_states[kv_h].transpose(-2, -1)) * scale
                attn_scores = attn_scores + causal_mask

                # Softmax (compute in float32 for numerical stability)
                attn_w = F.softmax(attn_scores.squeeze(1), dim=-1, dtype=torch.float32)
                # attn_w: [B, S, S]

                # Extract instr→img attention and store
                capture_dict[(layer_idx, h)] = (
                    attn_w[0, instr_s:instr_e, img_s:img_e]
                    .detach().cpu().float()
                )

                del q_h, attn_scores, attn_w

            del k_states

        return output

    return wrapper
