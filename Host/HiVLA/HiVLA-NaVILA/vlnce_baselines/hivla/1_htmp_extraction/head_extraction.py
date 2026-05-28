"""
Head extraction utilities for h_act and h_tmp.
Compute Diagonal-Shifting Temporal Score for attention heads.
"""

import numpy as np


def compute_temporal_score(frame_instr, num_frames=8):
    """
    Diagonal-Shifting Temporal Score (I_tmp).

    Evaluates how well each attention head exhibits temporal instruction
    alignment: earlier frames should focus on earlier instruction tokens,
    and focus should shift smoothly along a diagonal as frames progress.

    === Methodology ===

    1. Single-Peak Gaussian Focus (S_peak):
       Each frame's attention distribution should form a concentrated,
       unimodal peak (like a Gaussian) rather than scattered attention.
       Measured via:
       (a) Narrowness: low std relative to uniform (the peak is sharp)
       (b) Mass near CoM: most mass within +/-20% of instruction around
           the center-of-mass (unimodal, not multi-peak)
       Averaged over ALL frames (including low-attention ones).

    2. Frame Energy Uniformity (S_uniform):
       All frames should have comparable attention energy. Penalizes
       heads where only edge frames (first/last) have strong attention
       while middle frames have vanishing attention.
       S_uniform = mean(energy_f / max_energy) across all frames.

    3. Diagonal Alignment (S_diag):
       The peak (center-of-mass) positions across frames should follow
       an ideal diagonal line proportional to temporal progress.
       Frame k (code: k=0 oldest, k=T-1 newest) at progress p_k = k/(T-1)
       should attend to instruction position p_k * (M-1).
       Ideal: i*_k = p_k * (M-1)  [paper equiv: (1 - k/(T-1))*(N-1) with k=0=current]
       Measured as 1 - mean(normalized_absolute_error).

    4. Smooth Shifting (S_shift):
       The peaks should shift forward smoothly without stagnation or
       sudden jumps. Measured via:
       (a) Monotonic forward progression (peaks keep moving right)
       (b) Step-size regularity (actual shift ~ expected shift)

    I_tmp = S_peak * S_uniform * (0.60 * S_diag + 0.40 * S_shift)

    S_peak and S_uniform act as multiplicative gates: heads without clear
    focus or without uniform frame-level energy get their trajectory
    score suppressed.

    Args:
        frame_instr: dict {(layer, head): np.array [num_frames, num_instr]}
        num_frames: number of video frames (typically 8)

    Returns:
        dict {(layer, head): float} - temporal score per head (0.0 to 1.0)
    """
    if not frame_instr:
        return {}

    temporal_scores = {}

    for (l, h), attn_map in frame_instr.items():
        if attn_map is None or attn_map.shape[0] < num_frames:
            continue

        num_instr = attn_map.shape[1]
        if num_instr < 2:
            continue

        positions = np.arange(num_instr, dtype=np.float64)

        # Uniform distribution std (theoretical maximum spread)
        max_std = (num_instr - 1) / np.sqrt(12.0)

        # =============================================================
        # Step 0: Frame Energy Uniformity (S_uniform)
        # =============================================================
        # Compute per-frame total attention energy.
        # Heads where only edge frames have strong attention while middle
        # frames have vanishing energy will be penalized.
        energies = np.array([
            float(attn_map[k, :].sum()) for k in range(num_frames)
        ])
        max_energy = energies.max()
        if max_energy < 1e-10:
            continue  # All frames have zero attention

        energy_ratios = energies / max_energy   # each in [0, 1]
        s_uniform = float(np.mean(energy_ratios))  # 1.0 = perfectly uniform

        peak_positions = []   # Center-of-mass per frame
        focus_scores = []     # Single-peak quality per frame

        # =============================================================
        # Step 1 & 2: Per-frame — Single-Peak Validation + Peak Position
        # =============================================================
        for k in range(num_frames):
            attn_dist = attn_map[k, :].astype(np.float64)
            attn_sum = attn_dist.sum()

            if attn_sum < 1e-10:
                peak_positions.append(None)
                focus_scores.append(0.0)
                continue

            attn_norm = attn_dist / attn_sum

            # --- Center of Mass (Gaussian mean) ---
            com = float(np.sum(positions * attn_norm))

            # --- Standard Deviation (Gaussian width) ---
            var = float(np.sum(((positions - com) ** 2) * attn_norm))
            std = np.sqrt(var + 1e-10)

            # --- Single-Peak Validation ---
            # (a) Narrowness: low std = sharp peak
            #     Normalized relative to uniform distribution std.
            #     Uniform -> 0.0, delta -> 1.0, Gaussian(std=2,N=30) -> ~0.76
            s_narrow = max(0.0, 1.0 - std / max_std)

            # (b) Unimodality: mass within +/-20% of instruction around CoM
            #     Single Gaussian: nearly all mass near CoM -> high
            #     Multi-peak: CoM falls between peaks, little mass -> low
            #     Uniform: only a fraction near CoM -> low
            window = max(1, int(num_instr * 0.2))
            com_int = int(round(com))
            lower_idx = max(0, com_int - window)
            upper_idx = min(num_instr, com_int + window + 1)
            mass_near_com = float(attn_norm[lower_idx:upper_idx].sum())

            # Combined focus score
            focus = 0.5 * s_narrow + 0.5 * mass_near_com

            peak_positions.append(com)
            focus_scores.append(focus)

        # --- Validation: enough valid frames? ---
        valid_frames = [
            k for k in range(num_frames) if peak_positions[k] is not None
        ]
        if len(valid_frames) < max(2, num_frames // 2):
            continue

        # =============================================================
        # S_peak: Average single-peak Gaussian focus (ALL frames)
        # =============================================================
        # Average over ALL frames, not just valid ones. Frames with
        # near-zero attention contribute focus_score=0.0, dragging down
        # the mean for heads that only focus on a few frames.
        s_peak = float(np.mean(focus_scores))

        # =============================================================
        # Step 3: Diagonal Trajectory Tracking
        # =============================================================
        # Ideal diagonal (code convention: k=0=oldest, k=T-1=current):
        #   i*_k = (k / (T-1)) * (N-1)
        # Equivalent to paper's i*_k = (1 - k/(T-1)) * (N-1) with k=0=current,
        # just with reversed frame index ordering.
        # No ρ scaling: full range [0, N-1].

        diag_errors = []
        actual_peaks = []
        ideal_peaks = []

        for k in valid_frames:
            progress = k / max(num_frames - 1, 1)
            ideal_pos = progress * (num_instr - 1)
            actual_pos = peak_positions[k]

            # Normalized absolute error (0 to 1)
            error = abs(actual_pos - ideal_pos) / max(num_instr - 1, 1)
            diag_errors.append(error)
            actual_peaks.append(actual_pos)
            ideal_peaks.append(ideal_pos)

        # =============================================================
        # S_diag: Diagonal Alignment Score
        # =============================================================
        s_diag = max(0.0, 1.0 - float(np.mean(diag_errors)))

        # =============================================================
        # S_shift: Smooth Diagonal Shifting
        # =============================================================
        if len(actual_peaks) >= 2:
            shifts = np.diff(actual_peaks)
            ideal_shifts = np.diff(ideal_peaks)

            # (a) Monotonic forward progression (anti-stagnation)
            #     Peaks should move to the right as frames progress
            positive_count = int(np.sum(np.array(shifts) > 0))
            s_monotonic = float(positive_count) / len(shifts)

            # (b) Step-size regularity
            #     Each shift should be close to the expected ideal shift.
            #     Gaussian penalty: perfect ratio=1.0 -> 1.0,
            #       stagnation (0.0) or jump (2.0) -> ~0.6
            ideal_step = float(np.mean(np.abs(ideal_shifts)))
            ideal_step = max(ideal_step, 1e-10)

            smoothness_scores = []
            for s in shifts:
                ratio = s / ideal_step
                # Penalize both stagnation (ratio~0) and jumps (ratio>>1)
                score = float(np.exp(-0.5 * (ratio - 1.0) ** 2))
                smoothness_scores.append(score)

            s_smooth = float(np.mean(smoothness_scores))

            s_shift = 0.5 * s_monotonic + 0.5 * s_smooth
        else:
            s_shift = 0.0

        # =============================================================
        # Step 5: Combined Temporal Score
        #   S_peak gates on single-peak focus quality.
        #   S_uniform gates on frame-level energy uniformity.
        #   S_diag + S_shift form the trajectory quality score.
        # =============================================================
        s_trajectory = 0.60 * s_diag + 0.40 * s_shift
        temporal_score = s_peak * s_uniform * s_trajectory

        temporal_scores[(l, h)] = float(temporal_score)

    return temporal_scores


def aggregate_temporal_scores(scores_list):
    """
    Aggregate Diagonal-Shifting temporal scores from multiple steps/episodes.

    Args:
        scores_list: List of score dicts from compute_temporal_score()

    Returns:
        dict {(layer, head): average_score}
    """
    aggregated = {}
    counts = {}

    for scores in scores_list:
        for head, score in scores.items():
            if head not in aggregated:
                aggregated[head] = 0.0
                counts[head] = 0
            aggregated[head] += score
            counts[head] += 1

    for head in aggregated:
        aggregated[head] /= counts[head]

    return aggregated
