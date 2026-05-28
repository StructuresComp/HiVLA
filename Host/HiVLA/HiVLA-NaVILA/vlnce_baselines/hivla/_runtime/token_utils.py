"""
Token range computation utilities.
Extract image and instruction token ranges in expanded embeddings.
"""

from llava.constants import IMAGE_TOKEN_INDEX


def compute_token_ranges(input_ids, expanded_seq_len, tokenizer):
    """
    Compute image and instruction token ranges in expanded embedding.
    
    Args:
        input_ids: Input token IDs [1, seq_len]
        expanded_seq_len: Sequence length after image expansion
        tokenizer: Tokenizer instance
        
    Returns:
        dict containing:
            - img_start, img_end: Image token range
            - num_img_tokens, num_img_placeholders: Image token counts
            - instr_start, instr_end: Instruction token range
            - instr_token_labels: List of instruction token strings
    """
    ids_list = input_ids[0].cpu().tolist()

    # Find image placeholder positions
    img_positions = [
        i for i, t in enumerate(ids_list) if t == IMAGE_TOKEN_INDEX
    ]
    num_img_placeholders = len(img_positions)
    first_img_pos = img_positions[0] if img_positions else 0
    
    # Calculate expanded image tokens
    num_img_tokens = expanded_seq_len - (
        len(ids_list) - num_img_placeholders
    )
    img_start = first_img_pos
    img_end = first_img_pos + num_img_tokens
    offset = num_img_tokens - num_img_placeholders

    # Anchor-based instruction detection (match exact prompt format)
    prefix_text = '. Your assigned task is: "' 
    suffix_text = ' Analyze this series of images'  # Space before Analyze

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

    # Use more tokens for accurate matching
    p_anchor = prefix_ids[-6:] if len(prefix_ids) > 6 else prefix_ids
    s_anchor = suffix_ids[:8] if len(suffix_ids) > 8 else suffix_ids

    def find_subseq(seq, subseq, start=0):
        n = len(subseq)
        for i in range(start, len(seq) - n + 1):
            if seq[i:i + n] == subseq:
                return i
        return -1

    # Start searching AFTER all image tokens
    if IMAGE_TOKEN_INDEX in ids_list:
        last_img_idx = max(i for i, t in enumerate(ids_list) if t == IMAGE_TOKEN_INDEX)
        search_start = last_img_idx + 1
    else:
        search_start = 0

    instr_start = instr_end = None
    instr_token_labels = []

    p_idx = find_subseq(ids_list, p_anchor, search_start)

    if p_idx != -1:
        instr_start_orig = p_idx + len(p_anchor)
        s_idx = find_subseq(ids_list, s_anchor, instr_start_orig)
        if s_idx != -1:
            instr_end_orig = s_idx

            # Extract instruction tokens
            instr_tokens = ids_list[instr_start_orig:instr_end_orig]

            # Remove trailing quote if present
            if instr_tokens:
                last_token = instr_tokens[-1]
                last_decoded = tokenizer.decode([last_token]).strip()
                if last_decoded == '"':
                    instr_tokens = instr_tokens[:-1]
                    instr_end_orig -= 1

            instr_start = instr_start_orig + offset
            instr_end = instr_end_orig + offset

            for tid in instr_tokens:
                decoded = tokenizer.decode([tid]).strip()
                if not decoded:
                    decoded = "_"
                instr_token_labels.append(decoded)

    return {
        'img_start': img_start,
        'img_end': img_end,
        'num_img_tokens': num_img_tokens,
        'num_img_placeholders': num_img_placeholders,
        'instr_start': instr_start,
        'instr_end': instr_end,
        'instr_token_labels': instr_token_labels,
    }
