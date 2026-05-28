"""
Head extraction for h_tmp (Temporal Focus+Flow).

h_tmp: Diagonal-Shifting Temporal Score
    - S_peak: Single-peak Gaussian focus validation (concentration + unimodality).
    - S_diag: Diagonal trajectory alignment (peak position vs ideal diagonal).
    - S_shift: Smooth forward shifting (monotonicity + step regularity).
    - I_tmp = S_peak * (0.60 * S_diag + 0.40 * S_shift)

Usage:
    extractor = HeadExtractor(num_layers=32, num_heads=32, head_dim=128)

    # h_tmp: accumulate per episode
    extractor.accumulate_h_tmp(episode_steps, spl, spl_threshold)

    # Save per-chunk results (multi-GPU)
    extractor.save_h_tmp_chunk(output_dir, chunk_idx)

    # Merge all chunks into final ranking
    HeadExtractor.merge_h_tmp_chunks(results_dir, num_chunks)
"""

import glob
import json
import os

import numpy as np

from .head_extraction import compute_temporal_score
from .score_visualizer import plot_head_ranking, save_ideal_pattern


class HeadExtractor:
    """
    Head extraction for h_tmp (Diagonal-Shifting Temporal Score).

    Measures how well each attention head tracks temporal navigation context:
    - S_peak: Single-peak Gaussian focus validation
    - S_diag: Diagonal trajectory alignment
    - S_shift: Smooth forward shifting
    - I_tmp = S_peak * (0.60*S_diag + 0.40*S_shift)
    """

    def __init__(self, num_layers=32, num_heads=32, head_dim=128):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # h_tmp accumulators
        self.temporal_scores = {}      # {(l,h): total_score}
        self.temporal_counts = {}      # {(l,h): count}
        self.analyzed_episodes = 0
        self.ideal_pattern_acc = []    # list of [num_frames, num_instr] arrays

    # ===========================================================
    # h_tmp: Temporal Focus+Flow Extraction
    # ===========================================================

    def accumulate_h_tmp(self, episode_steps, spl, spl_threshold=1.0):
        """
        Process a completed episode for h_tmp extraction.
        Only accumulates if SPL >= threshold (successful navigation).

        Args:
            episode_steps: List of dicts, each with:
                - 'frame_instr': dict {(l,h): np.array [num_frames, num_instr]}
                - 'num_frames': int (typically 8)
            spl: Episode SPL score (0.0 to 1.0)
            spl_threshold: Minimum SPL to include this episode

        Returns:
            True if episode was accepted, False otherwise
        """
        if spl < spl_threshold or len(episode_steps) == 0:
            return False

        for step_data in episode_steps:
            frame_instr = step_data['frame_instr']
            num_frames = step_data.get('num_frames', 8)

            # Compute Focus+Flow temporal scores per head
            scores = compute_temporal_score(
                frame_instr, num_frames=num_frames
            )
            for head, score in scores.items():
                if head not in self.temporal_scores:
                    self.temporal_scores[head] = 0.0
                    self.temporal_counts[head] = 0
                self.temporal_scores[head] += score
                self.temporal_counts[head] += 1

            # Accumulate ideal attention pattern (mean over all heads)
            step_maps = []
            for (l, h), attn in frame_instr.items():
                if attn is not None and attn.shape[0] >= num_frames:
                    attn_np = attn[:num_frames, :]
                    if hasattr(attn_np, 'numpy'):
                        attn_np = attn_np.numpy()
                    row_sums = attn_np.sum(axis=1, keepdims=True)
                    row_sums = np.maximum(row_sums, 1e-10)
                    attn_normed = attn_np / row_sums
                    step_maps.append(attn_normed)
            if step_maps:
                avg_map = np.mean(step_maps, axis=0)
                self.ideal_pattern_acc.append(avg_map)

        self.analyzed_episodes += 1
        return True

    # ===========================================================
    # Save & Merge: h_tmp
    # ===========================================================

    def save_h_tmp_chunk(self, output_dir, chunk_idx):
        """
        Save h_tmp raw scores for this chunk (multi-GPU support).

        Returns:
            Path to saved chunk JSON, or None if no data.
        """
        if self.analyzed_episodes == 0:
            return None

        chunk_data = {
            'scores': {
                f"{l},{h}": s
                for (l, h), s in self.temporal_scores.items()
            },
            'counts': {
                f"{l},{h}": c
                for (l, h), c in self.temporal_counts.items()
            },
            'analyzed_episodes': self.analyzed_episodes,
        }
        chunk_path = os.path.join(
            output_dir, f'temporal_head_raw_chunk{chunk_idx}.json'
        )
        with open(chunk_path, 'w') as f:
            json.dump(chunk_data, f)
        print(
            f"[HeadExtractor] Saved h_tmp chunk {chunk_idx}: "
            f"{self.analyzed_episodes} episodes -> {chunk_path}"
        )
        return chunk_path

    @staticmethod
    def merge_h_tmp_chunks(results_dir, num_chunks, spl_threshold=1.0):
        """
        Merge h_tmp raw chunks from multiple GPUs into final ranking.

        Reads temporal_head_raw_chunk*.json files, averages scores,
        and saves:
            - temporal_head_importance.json (ranked heads)
            - temporal_head_importance_ranked.png (bar chart)

        Returns:
            Path to saved JSON, or None if not all chunks ready.
        """
        chunk_files = sorted(glob.glob(
            os.path.join(results_dir, 'temporal_head_raw_chunk*.json')
        ))

        if len(chunk_files) < num_chunks:
            print(
                f"[HeadExtractor] Waiting for h_tmp chunks "
                f"({len(chunk_files)}/{num_chunks})"
            )
            return None

        print(f"[HeadExtractor] Merging {len(chunk_files)} h_tmp chunks...")

        merged_scores = {}
        merged_counts = {}
        total_episodes = 0

        for path in chunk_files:
            with open(path) as f:
                data = json.load(f)
            total_episodes += data['analyzed_episodes']
            for key, score in data['scores'].items():
                l, h = map(int, key.split(','))
                head = (l, h)
                merged_scores[head] = (
                    merged_scores.get(head, 0.0) + score
                )
                merged_counts[head] = (
                    merged_counts.get(head, 0) + data['counts'][key]
                )

        # Compute averaged scores
        ranked = []
        for head, total_score in merged_scores.items():
            count = merged_counts[head]
            avg_score = total_score / count if count > 0 else 0.0
            ranked.append({
                'layer': head[0],
                'head': head[1],
                'label': f'L{head[0]}H{head[1]}',
                'temporal_score': avg_score,
                'step_count': count,
            })

        ranked.sort(key=lambda x: x['temporal_score'], reverse=True)
        for i, entry in enumerate(ranked):
            entry['rank'] = i + 1

        # Save JSON
        output = {
            'metadata': {
                'head_type': 'h_tmp',
                'analyzed_episodes': total_episodes,
                'spl_threshold': spl_threshold,
                'total_heads_evaluated': len(ranked),
                'metric': 'diagonal_shifting',
                'description': (
                    'Diagonal-Shifting Temporal Score: '
                    'I_tmp = S_peak * (0.60*S_diag + 0.40*S_shift). '
                    'S_peak = single-peak Gaussian focus gate. '
                    'S_diag = diagonal trajectory alignment. '
                    'S_shift = smooth forward shifting (anti-stagnation).'
                ),
            },
            'ranked_heads': ranked,
        }
        json_path = os.path.join(
            results_dir, 'temporal_head_importance.json'
        )
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

        # Save ranking chart
        try:
            chart_path = os.path.join(
                results_dir, 'temporal_head_importance_ranked.png'
            )
            plot_head_ranking(
                ranked, chart_path,
                metric_name='temporal_score',
                title_prefix='Temporal Heads - Diagonal-Shifting',
                num_episodes=total_episodes,
            )
        except Exception as e:
            print(f"[HeadExtractor] Warning: chart error: {e}")

        print(f"[HeadExtractor] Saved h_tmp rankings: {json_path}")
        print(
            f"[HeadExtractor] Total: {total_episodes} episodes, "
            f"{len(ranked)} heads"
        )
        print("[HeadExtractor] Top-10 h_tmp heads:")
        for entry in ranked[:10]:
            print(
                f"  #{entry['rank']:2d}  {entry['label']:6s}  "
                f"temporal_score={entry['temporal_score']:.4f}  "
                f"(steps={entry['step_count']})"
            )

        return json_path

    # ===========================================================
    # Ideal Attention Pattern
    # ===========================================================

    def save_ideal_patterns(self, output_dir):
        """
        Save ideal attention pattern (average of successful episodes).
        """
        if len(self.ideal_pattern_acc) == 0:
            return
        save_ideal_pattern(self.ideal_pattern_acc, output_dir)
