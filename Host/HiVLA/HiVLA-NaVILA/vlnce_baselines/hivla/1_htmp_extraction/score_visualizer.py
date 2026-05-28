"""
Score visualization utilities for head rankings and analysis plots.
"""

import numpy as np
import os


def plot_head_ranking(ranked_heads, output_path, metric_name='importance',
                     title_prefix='Head Ranking', num_episodes=None):
    """
    Generate and save bar chart of head rankings.
    
    Args:
        ranked_heads: List of dicts with 'label', metric_name, 'rank'
        output_path: Path to save PNG
        metric_name: Name of the metric ('importance', 'temporal_score')
        title_prefix: Title prefix for the chart
        num_episodes: Optional number of episodes (for subtitle)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Visualization] Warning: matplotlib not available")
        return
    
    top_n = min(50, len(ranked_heads))
    top = ranked_heads[:top_n]
    
    fig, ax = plt.subplots(figsize=(18, 6))
    labels = [e['label'] for e in top]
    values = [e[metric_name] for e in top]
    
    # Color scheme
    if 'temporal' in metric_name:
        colors = plt.cm.YlGnBu(np.linspace(0.9, 0.3, top_n))
    else:
        colors = plt.cm.OrRd(np.linspace(0.9, 0.3, top_n))
    
    ax.bar(range(top_n), values, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel('Head (ranked by ' + metric_name.replace('_', ' ') + ', highest first)')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    
    title = f'Top-{top_n} {title_prefix}'
    if num_episodes:
        title += f'\n({num_episodes} episodes)'
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualization] Saved ranking chart: {output_path}")


def save_ideal_pattern(pattern_list, output_dir):
    """
    Save ideal attention pattern (average of successful episodes).
    This is a visualization artifact for analysis.

    Args:
        pattern_list: List of [num_frames, num_instr] arrays
        output_dir: Output directory
    """
    import torch
    import torch.nn.functional as F
    
    # Normalize to fixed grid
    grid_size = 32
    normalized_patterns = []
    
    for pattern in pattern_list:
        pt = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            pt, size=(pattern.shape[0], grid_size),
            mode='bilinear', align_corners=True
        ).squeeze(0).squeeze(0).numpy()
        normalized_patterns.append(resized)
    
    # Average over all steps
    ideal_pattern = np.mean(normalized_patterns, axis=0)
    
    # Save as .npy
    pattern_path = os.path.join(output_dir, 'ideal_attention_pattern.npy')
    np.save(pattern_path, ideal_pattern)
    print(f"[Visualization] Saved ideal pattern: {ideal_pattern.shape} → {pattern_path}")
    
    # Save visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(ideal_pattern, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xlabel('Instruction Position (normalized)')
        ax.set_ylabel('Frame Index')
        ax.set_title('Ideal Attention Pattern (avg from successful episodes)')
        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.savefig(os.path.join(output_dir, 'ideal_attention_pattern.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualization] Saved: ideal_attention_pattern.png")
    except Exception as e:
        print(f"[Visualization] Warning: Could not save pattern visualization: {e}")
