"""
Full-capture attention forward for NaVid (h_tmp extraction mode).

Ported from NaVILA's hivla/_runtime/capture_forward.py.
Used when extract_h_tmp=True: runs a SEPARATE LLM forward pass capturing
ALL heads. Installed as a method-replacement on model.model.layers[i].self_attn.

For inline (single-head) capture during generate(), use inline_capture.py instead.
"""

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def _capture_attn_forward(
    self, hidden_states, attention_mask=None, position_ids=None,
    past_key_value=None, output_attentions=False, use_cache=False,
    layer_idx=None, capture_dict=None, token_ranges=None, vis_heads=None,
    **kwargs
):
    """
    Capture-only attention forward.
    Captures instr→img attention weights for specified heads during prefill.
    Installed via functools.partial on model.model.layers[i].self_attn.forward.

    Args:
        layer_idx:    Current layer index (injected via functools.partial).
        capture_dict: Dict to store {(layer, head): Tensor[instr, img]}.
        token_ranges: Dict with instr_start/end, img_start/end.
        vis_heads:    Set of (l, h) to capture; None → capture all heads.
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            key_states   = torch.cat([past_key_value[0], key_states],   dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_kv_out = (key_states, value_states) if use_cache else None

    key_states   = repeat_kv(key_states,   self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    kv_seq_len = key_states.shape[2]
    scale = 1.0 / (self.head_dim ** 0.5)

    if attention_mask is not None:
        causal_mask = attention_mask.squeeze(1)
    else:
        causal_mask = torch.triu(
            torch.full((q_len, kv_seq_len), float('-inf'), device=hidden_states.device),
            diagonal=1,
        ).unsqueeze(0)

    is_prefill    = (q_len > 1)
    should_capture = (
        is_prefill
        and token_ranges is not None
        and token_ranges.get('instr_start') is not None
        and capture_dict is not None
    )

    attn_outputs = []
    for h in range(self.num_heads):
        q_h = query_states[:, h]
        k_h = key_states[:, h]
        v_h = value_states[:, h]

        attn_w = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
        attn_w = attn_w + causal_mask
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q_h.dtype)

        if should_capture and (vis_heads is None or (layer_idx, h) in vis_heads):
            instr_s = token_ranges['instr_start']
            instr_e = token_ranges['instr_end']
            img_s   = token_ranges['img_start']
            img_e   = token_ranges['img_end']
            capture_dict[(layer_idx, h)] = (
                attn_w[0, instr_s:instr_e, img_s:img_e]
                .detach().cpu().float()
            )

        attn_outputs.append(torch.matmul(attn_w, v_h))

    attn_output = torch.stack(attn_outputs, dim=1)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_kv_out
