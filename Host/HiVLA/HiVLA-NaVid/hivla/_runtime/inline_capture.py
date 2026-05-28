"""
Inline attention capture for NaVid via forward wrappers.

Ported from NaVILA's hivla/_runtime/inline_capture.py.
Key change: NaVid uses model.model.layers[i] (not model.llm.model.layers[i]).

Installs lightweight Q·K^T wrappers on target attention layers that fire
during model.generate() prefill, capturing instr→img attention for only
the specified K heads (~2-5 ms overhead vs ~700 ms for a full separate pass).
"""

import types

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def install_inline_hooks(model, target_heads, token_ranges, capture_dict):
    """
    Install lightweight attention capture wrappers on target layers.

    Args:
        model:        NaVid LlavaLlamaAttForCausalLM (model.model.layers accessible).
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

    heads_by_layer = {}
    for l, h in target_heads:
        heads_by_layer.setdefault(l, []).append(h)

    orig_forwards = {}
    for layer_idx, heads in heads_by_layer.items():
        # NaVid: model.model.layers (NaVILA used model.llm.model.layers)
        layer = model.model.layers[layer_idx]
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
        # NaVid: model.model.layers
        model.model.layers[layer_idx].self_attn.forward = orig_fwd


def _create_capture_wrapper(
    original_forward, layer_idx, target_heads,
    token_ranges, capture_dict, attn_module,
):
    """
    Closure wrapping Flash Attention forward. During prefill (seq_len > 1),
    additionally computes Q·K^T for only the specified target heads using
    sliced projection weights. Returns original output unchanged.
    """
    num_heads     = attn_module.num_heads
    num_kv_heads  = attn_module.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim      = attn_module.head_dim
    rotary_emb    = attn_module.rotary_emb
    q_proj        = attn_module.q_proj
    k_proj        = attn_module.k_proj

    # Pre-slice projection weights for target heads (views, no copy)
    q_weight_slices = {}
    q_bias_slices   = {}
    k_kv_heads_needed = set()

    for h in target_heads:
        q_weight_slices[h] = q_proj.weight[h * head_dim : (h + 1) * head_dim]
        q_bias_slices[h]   = (
            q_proj.bias[h * head_dim : (h + 1) * head_dim]
            if q_proj.bias is not None else None
        )
        k_kv_heads_needed.add(h // num_kv_groups)

    k_weight_slices = {}
    k_bias_slices   = {}
    for kv_h in k_kv_heads_needed:
        k_weight_slices[kv_h] = k_proj.weight[kv_h * head_dim : (kv_h + 1) * head_dim]
        k_bias_slices[kv_h]   = (
            k_proj.bias[kv_h * head_dim : (kv_h + 1) * head_dim]
            if k_proj.bias is not None else None
        )

    captured = [False]  # mutable flag — only capture once (prefill)

    instr_s = token_ranges['instr_start']
    instr_e = token_ranges['instr_end']
    img_s   = token_ranges['img_start']
    img_e   = token_ranges['img_end']
    scale   = 1.0 / (head_dim ** 0.5)

    def wrapper(self_attn, hidden_states, attention_mask=None,
                position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, **kwargs):
        # 1) Run original Flash Attention forward unchanged
        output = original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # 2) Only capture during prefill
        bsz, q_len, _ = hidden_states.size()
        if captured[0] or q_len <= 1:
            return output
        captured[0] = True

        # 3) Lightweight Q·K^T for target heads only
        with torch.no_grad():
            k_states = {}
            for kv_h in k_kv_heads_needed:
                k_h = F.linear(hidden_states, k_weight_slices[kv_h], k_bias_slices[kv_h])
                k_states[kv_h] = k_h.unsqueeze(1)  # [B, 1, S, head_dim]

            seq_len = hidden_states.shape[1]
            dummy  = next(iter(k_states.values()))
            # transformers 4.31: rotary_emb(x, seq_len=int), not (x, position_ids)
            cos, sin = rotary_emb(dummy, seq_len=seq_len)

            for kv_h in k_states:
                _, k_states[kv_h] = apply_rotary_pos_emb(
                    k_states[kv_h], k_states[kv_h], cos, sin, position_ids
                )

            causal_mask = torch.triu(
                torch.full((q_len, q_len), float('-inf'),
                           device=hidden_states.device, dtype=hidden_states.dtype),
                diagonal=1,
            )

            for h in target_heads:
                kv_h = h // num_kv_groups

                q_h = F.linear(hidden_states, q_weight_slices[h], q_bias_slices[h])
                q_h = q_h.unsqueeze(1)  # [B, 1, S, head_dim]
                q_h, _ = apply_rotary_pos_emb(q_h, q_h, cos, sin, position_ids)

                attn_scores = torch.matmul(q_h, k_states[kv_h].transpose(-2, -1)) * scale
                attn_scores = attn_scores + causal_mask

                attn_w = F.softmax(attn_scores.squeeze(1), dim=-1, dtype=torch.float32)
                # attn_w: [B, S, S]

                # Store instr→img slice
                capture_dict[(layer_idx, h)] = (
                    attn_w[0, instr_s:instr_e, img_s:img_e]
                    .detach().cpu().float()
                )

                del q_h, attn_scores, attn_w

            del k_states

        return output

    return wrapper
