"""
TemporalAnomalyLogger — Passive data collection for cognitive anomaly detection.

Runs during VLN-CE evaluation, recording per-head attention metrics from H_tmp
heads at every VLM prediction step. Output is streamed to JSONL for offline
threshold optimization (grid-search) without re-running the VLA model.

Output schema (per row / JSONL line):
    episode_id       (str)   - Episode identifier
    step             (int)   - VLM prediction step index
    action_name      (str)   - VLN-CE action taken at this step
                               ('STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT')
    action_prob      (float) - Probability [0,1] of the action-discriminating token
                               (e.g., "forward"/"left"/"right"/"stop") under the VLM's
                               softmax distribution. None if not available.
    agent_x          (float) - Agent X position (Habitat coords)
    agent_y          (float) - Agent Y position (height)
    agent_z          (float) - Agent Z position (Habitat coords)
    metrics_per_head (dict)  - Per-head metrics keyed by "layer,head"
        e_system     (float) - Systemic entropy (cognitive diffusion)
        s_peak       (float) - Peak normalized attention in current frame [0, 1]
                               High = focused; Low = dispersed
        entropy_list (list)  - Raw per-frame entropy [E_0, E_1, ..., E_{F-1}]
                               Allows offline computation of slope, drift, variance, etc.

Metric definitions (computed per individual head):
    E_system = (1/F) * sum_f [ -sum_i(A_fi * log(A_fi + eps)) / log(N) ]
    S_peak   = max_i(A_{F-1, i})  (peak attention in current/last frame)
    entropy_list = [E_0, E_1, ..., E_{F-1}]  (raw, for offline trend analysis)
        where E_f = -sum_i(A_fi * log(A_fi + eps)) / log(N)

    where A = single head's attention, normalized per frame.
    F = num_frames, N = num_instr_tokens.
"""

import json
import os
from typing import Dict, Optional, Set

import numpy as np


class TemporalAnomalyLogger:
    """
    Passive data collection logger for temporal attention anomaly detection.
    Computes metrics per individual head (not averaged) for offline analysis.

    Usage:
        logger = TemporalAnomalyLogger(
            output_path='anomaly_metrics.jsonl',
            h_tmp_heads={(11, 5), (13, 2), ...},
        )

        # Per episode:
        for step in range(num_steps):
            logger.log_step(
                frame_instr, episode_id, step,
                action_name='MOVE_FORWARD',
                agent_pos={"x": 1.0, "y": 0.0, "z": 2.0},
            )
        logger.end_episode()

        logger.close()
    """

    def __init__(self, output_path, h_tmp_heads=None, **kwargs):
        """
        Args:
            output_path: Path to output file (.jsonl).
            h_tmp_heads: Set of (layer, head) tuples defining H_tmp.
                         If None, computes metrics for ALL heads in attention_maps.
        """
        self.output_path = output_path
        self.h_tmp_heads = set(h_tmp_heads) if h_tmp_heads is not None else None

        self._episode_buffer = []
        self._total_steps = 0
        self._total_episodes = 0

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        self._file = open(output_path, 'w')

    # ===========================================================
    # Core metric computation
    # ===========================================================

    @staticmethod
    def compute_metrics(attention_maps, h_tmp_heads=None):
        """
        Compute per-head metrics: e_system, s_peak, entropy_list.
        """
        metrics_per_head = {}

        for (l, h), attn in attention_maps.items():
            if h_tmp_heads is not None and (l, h) not in h_tmp_heads:
                continue

            # Defensive: handle GPU tensors safely
            if hasattr(attn, 'detach'):
                attn = attn.detach().cpu().numpy()
            arr = np.asarray(attn, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 2:
                continue

            F, N = arr.shape

            # --- Normalize each frame to sum=1 ---
            row_sums = arr.sum(axis=1, keepdims=True)
            A = arr / np.maximum(row_sums, 1e-10)  # [F, N]

            # --- E_system: Systemic Entropy ---
            eps = 1e-10
            log_N = np.log(max(N, 2))
            E_f = -np.sum(A * np.log(A + eps), axis=1) / log_N  # [F]
            e_system = float(np.mean(E_f))

            # --- S_peak: Peak normalized attention in current frame ---
            s_peak = float(A[-1].max())

            # --- Raw Entropy List for Offline Analysis ---
            entropy_list = [round(float(x), 6) for x in E_f]

            key = f"{l},{h}"
            metrics_per_head[key] = {
                'e_system': round(e_system, 6),
                's_peak': round(s_peak, 6),
                'entropy_list': entropy_list,
            }

        if not metrics_per_head:
            return None

        return metrics_per_head

    # ===========================================================
    # Logging API
    # ===========================================================

    def log_step(self, attention_maps, episode_id, step,
                 action_name='MOVE_FORWARD', agent_pos=None, action_prob=None):
        """
        Record per-head metrics for one VLM prediction step.

        Args:
            attention_maps: Dict {(l, h): np.array [F, N]} — frame_instr format.
            episode_id: Episode identifier.
            step: VLM prediction step index.
            action_name: VLN-CE action taken ('STOP', 'MOVE_FORWARD',
                         'TURN_LEFT', 'TURN_RIGHT').
            agent_pos: Dict {"x", "y", "z"} — agent position.
                       If None, zeros are recorded.
            action_prob: Softmax probability [0,1] of the action-discriminating token.
                         None if scores were not captured.
        """
        metrics_per_head = self.compute_metrics(attention_maps, self.h_tmp_heads)
        if metrics_per_head is None:
            return

        if agent_pos is None:
            agent_pos = {"x": 0.0, "y": 0.0, "z": 0.0}

        row = {
            'episode_id': str(episode_id),
            'step': int(step),
            'action_name': str(action_name),
            'action_prob': round(float(action_prob), 6) if action_prob is not None else None,
            'agent_x': round(float(agent_pos.get('x', 0.0)), 4),
            'agent_y': round(float(agent_pos.get('y', 0.0)), 4),
            'agent_z': round(float(agent_pos.get('z', 0.0)), 4),
            'metrics_per_head': metrics_per_head,
        }
        self._episode_buffer.append(row)
        self._total_steps += 1

    def end_episode(self):
        """Finalize current episode: flush buffered rows to disk."""
        self._flush_buffer()
        self._episode_buffer = []
        self._total_episodes += 1

    # ===========================================================
    # I/O
    # ===========================================================

    def _flush_buffer(self):
        """Write buffered rows to file."""
        if not self._episode_buffer:
            return

        for row in self._episode_buffer:
            self._file.write(json.dumps(row) + '\n')

        self._file.flush()

    def close(self):
        """Flush remaining data and close file."""
        if self._episode_buffer:
            self._flush_buffer()
            self._episode_buffer = []

        if self._file and not self._file.closed:
            self._file.close()

        print(
            f"[AnomalyLogger] Closed: {self._total_episodes} episodes, "
            f"{self._total_steps} steps → {self.output_path}"
        )

    def __del__(self):
        try:
            if self._file and not self._file.closed:
                self.close()
        except Exception:
            pass
