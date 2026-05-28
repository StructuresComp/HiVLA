"""
Head selection utilities.
Load ranked heads from h_tmp importance JSON files.
"""

import json
import os


def load_ranked_heads(path, ratio):
    """
    Load top-k heads from ranked importance JSON.
    
    Args:
        path: Path to head importance JSON file
        ratio: Top ratio to select (e.g., 0.1 for top 10%)
        
    Returns:
        list of (layer, head) tuples
        dict of (layer,head) -> rank mapping
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Head importance file not found: {path}")
        
    with open(path) as f:
        data = json.load(f)
    
    ranked = data['ranked_heads']
    total = len(ranked)
    top_k = max(1, int(total * ratio))
    heads = [(e['layer'], e['head']) for e in ranked[:top_k]]
    
    # Store ranking info
    head_rankings = {}
    for e in ranked[:top_k]:
        head_rankings[(e['layer'], e['head'])] = e['rank']
    
    # Print summary
    print(f"[HeadSelection] Loaded {len(heads)} heads from {path}")
    for e in ranked[:min(10, top_k)]:
        score_key = 'temporal_score' if 'temporal_score' in e else 'importance'
        print(f"  #{e['rank']:3d} {e['label']:8s} "
              f"{score_key}={e.get(score_key, 0):.4f}")
    if top_k > 10:
        print(f"  ... ({top_k - 10} more)")
    
    return heads, head_rankings


