#!/usr/bin/env python3
"""
Waypoint-based GT Labeling for VLN Anomaly Detection
═════════════════════════════════════════════════════

Generates per-step Normal/Anomaly ground truth labels from R2R reference paths.

Core Concept: Closest-Checkpoint Tracking
  At each step, track the closest forward waypoint (monotonic, no backtrack).
  Deviation = agent moving away from target for P consecutive forward steps.

State Transitions (one-way, no recovery):
  Normal -> Anomaly: P consecutive FORWARD steps with dist_delta > threshold
  Once in Anomaly:   ALL remaining steps are permanently Anomaly
  If normal resumes:  Episode is truncated (don't label normal as anomaly)

Episode Categories:
  only_normal       : All steps Normal
  only_anomaly      : Deviation within first P forward steps
  normal_to_anomaly : Normal navigation followed by permanent anomaly

Usage:
    python waypoint_gt_labeling.py --data_dir <anomaly_metrics_dir> \\
                                   --dataset_dir <R2R_VLNCE_dir> \\
                                   --split train
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROTATION_ACTIONS = {"TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"}
DEFAULT_DIST_DELTA_THRESH = 0.0
DEFAULT_PATIENCE = 3


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry
# ══════════════════════════════════════════════════════════════════════════════

def _point_to_segment_projection(p, a, b):
    """Project point p onto line segment a-b."""
    ab = b - a
    seg_len_sq = np.dot(ab, ab)
    if seg_len_sq < 1e-12:
        return a.copy(), 0.0, float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / seg_len_sq, 0.0, 1.0)
    proj = a + t * ab
    dist = float(np.linalg.norm(p - proj))
    return proj, float(t), dist


def compute_step_metrics(agent_pos, reference_path, target_idx, prev_dist):
    """Per-step metrics using closest-checkpoint waypoint tracking.

    Returns dict with: target_idx, dist_to_target, dist_delta, ctd, advanced.
    """
    n_wp = len(reference_path)
    target_idx = min(target_idx, n_wp - 1)

    best_idx = target_idx
    best_dist = float(np.linalg.norm(agent_pos - reference_path[target_idx]))

    for i in range(target_idx + 1, n_wp):
        d = float(np.linalg.norm(agent_pos - reference_path[i]))
        if d < best_dist:
            best_idx = i
            best_dist = d

    advanced = best_idx > target_idx

    # Full-path CTD: project onto ALL segments, take minimum distance
    ctd = float('inf')
    for seg_i in range(len(reference_path) - 1):
        _, _, seg_dist = _point_to_segment_projection(
            agent_pos, reference_path[seg_i], reference_path[seg_i + 1]
        )
        if seg_dist < ctd:
            ctd = seg_dist

    if advanced:
        dist_delta = 0.0
    else:
        dist_delta = best_dist - prev_dist if prev_dist is not None else 0.0

    return {
        'target_idx': best_idx,
        'dist_to_target': best_dist,
        'dist_delta': dist_delta,
        'ctd': ctd,
        'advanced': advanced,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Episode Classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_episode(labels):
    """Classify episode: only_normal / only_anomaly / normal_to_anomaly."""
    if not labels:
        return 'only_normal'
    has_anomaly = any(l == 1 for l in labels)
    if not has_anomaly:
        return 'only_normal'
    first_anom_idx = next(i for i, l in enumerate(labels) if l == 1)
    if first_anom_idx <= 1:
        return 'only_anomaly'
    return 'normal_to_anomaly'


# ══════════════════════════════════════════════════════════════════════════════
#  GT Labeling State Machine (one-way, no recovery)
# ══════════════════════════════════════════════════════════════════════════════

def build_gt_waypoint(episodes_raw, reference_paths,
                      dist_delta_thresh, patience):
    """Build GT labels using closest-checkpoint waypoint tracking.

    Returns:
        clean_episodes:     Dict[ep_id -> List[(step_dict, label)]]
        flat_steps:         List[(step_dict, label)]
        episode_metrics:    Dict[ep_id -> List[dict]] per-step metrics
        episode_categories: Dict[ep_id -> str]
    """
    clean_eps = defaultdict(list)
    flat_steps = []
    episode_metrics = {}
    episode_categories = {}

    n_missing_ref = 0
    n_missing_pos = 0

    for ep_id in sorted(episodes_raw):
        if ep_id not in reference_paths:
            n_missing_ref += 1
            continue

        ref_path = reference_paths[ep_id]
        steps = episodes_raw[ep_id]

        has_pos = any(
            s.get('agent_x') is not None and s.get('agent_z') is not None
            for s in steps
        )
        if not has_pos:
            n_missing_pos += 1
            continue

        # Per-step metrics
        step_metrics = []
        target_idx = 1
        prev_dist = None

        for step in steps:
            agent_pos = np.array([
                step.get('agent_x', 0.0),
                step.get('agent_y', 0.0),
                step.get('agent_z', 0.0),
            ])
            m = compute_step_metrics(agent_pos, ref_path, target_idx, prev_dist)
            target_idx = m['target_idx']
            prev_dist = m['dist_to_target']
            step_metrics.append({
                'target_idx': m['target_idx'],
                'dist_to_target': m['dist_to_target'],
                'dist_delta': m['dist_delta'],
                'ctd': m['ctd'],
                'advanced': m['advanced'],
                'is_deviated': m['dist_delta'] > dist_delta_thresh,
            })

        episode_metrics[ep_id] = step_metrics

        # State machine
        consec_bad = 0
        in_anom = False
        pending_bad = []
        pending_anom_good = []
        consec_anom_good = 0
        truncated = False

        for i, step in enumerate(steps):
            if truncated:
                break

            is_rot = step.get("action_name", "MOVE_FORWARD") in ROTATION_ACTIONS
            deviated = step_metrics[i]['is_deviated']

            if in_anom:
                if is_rot or deviated:
                    for ps in pending_anom_good:
                        clean_eps[ep_id].append((ps, 1))
                        flat_steps.append((ps, 1))
                    pending_anom_good = []
                    consec_anom_good = 0
                    clean_eps[ep_id].append((step, 1))
                    flat_steps.append((step, 1))
                else:
                    consec_anom_good += 1
                    pending_anom_good.append(step)
                    if consec_anom_good == patience:
                        truncated = True
            else:
                if is_rot:
                    continue
                elif not deviated:
                    for ps in pending_bad:
                        clean_eps[ep_id].append((ps, 0))
                        flat_steps.append((ps, 0))
                    pending_bad = []
                    consec_bad = 0
                    clean_eps[ep_id].append((step, 0))
                    flat_steps.append((step, 0))
                else:
                    consec_bad += 1
                    pending_bad.append(step)
                    if consec_bad == patience:
                        in_anom = True
                        for ps in pending_bad:
                            clean_eps[ep_id].append((ps, 1))
                            flat_steps.append((ps, 1))
                        pending_bad = []
                        consec_anom_good = 0
                        pending_anom_good = []

        if not in_anom and pending_bad:
            for ps in pending_bad:
                clean_eps[ep_id].append((ps, 0))
                flat_steps.append((ps, 0))
        elif in_anom and pending_anom_good and not truncated:
            for ps in pending_anom_good:
                clean_eps[ep_id].append((ps, 1))
                flat_steps.append((ps, 1))

        ep_labels = [l for _, l in clean_eps[ep_id]]
        episode_categories[ep_id] = classify_episode(ep_labels)

    if n_missing_ref > 0:
        print(f"  WARNING: {n_missing_ref} episodes missing reference path")
    if n_missing_pos > 0:
        print(f"  WARNING: {n_missing_pos} episodes missing position data")

    return clean_eps, flat_steps, episode_metrics, episode_categories


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_anomaly_metrics(data_dir):
    """Load anomaly/action-prob metrics from JSONL chunk files.

    Accepts both Step 2 output (anomaly_metrics_chunk*.jsonl) and
    Step 2b output (action_prob_chunk*.jsonl) from the same directory.
    """
    chunk_files = sorted(Path(data_dir).glob("anomaly_metrics_chunk*.jsonl"))
    if not chunk_files:
        # Fallback: step 2b lightweight files
        chunk_files = sorted(Path(data_dir).glob("action_prob_chunk*.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files in {data_dir}")

    episodes_raw = defaultdict(list)
    for cf in chunk_files:
        with open(cf) as fh:
            for line in fh:
                rec = json.loads(line)
                episodes_raw[rec["episode_id"]].append(rec)

    # Deduplicate steps per episode by step index (keep first occurrence)
    # Required when chunks overlap (e.g. 9-chunk + 4-chunk mixed runs)
    deduped = {}
    for ep_id, steps in episodes_raw.items():
        seen: set = set()
        deduped_steps = []
        for step in steps:
            key = step['step']
            if key not in seen:
                seen.add(key)
                deduped_steps.append(step)
        deduped[ep_id] = sorted(deduped_steps, key=lambda s: s['step'])
    return deduped


def load_reference_paths(dataset_dir, split):
    """Load reference paths from R2R dataset."""
    json_gz = Path(dataset_dir) / split / f"{split}.json.gz"
    if not json_gz.exists():
        raise FileNotFoundError(f"Dataset not found: {json_gz}")

    with gzip.open(json_gz, 'rt') as f:
        data = json.load(f)

    return {
        str(ep['episode_id']): np.array(ep['reference_path'], dtype=np.float64)
        for ep in data['episodes']
    }


def load_gt_trajectories(dataset_dir, split):
    """Load GT shortest-path trajectories (for position injection fallback)."""
    gt_gz = Path(dataset_dir) / split / f"{split}_gt.json.gz"
    if not gt_gz.exists():
        return {}
    with gzip.open(gt_gz, 'rt') as f:
        return json.load(f)


def load_episodes_meta(dataset_dir, split):
    """Load episode metadata (episode_id -> trajectory_id mapping)."""
    json_gz = Path(dataset_dir) / split / f"{split}.json.gz"
    if not json_gz.exists():
        return {}
    with gzip.open(json_gz, 'rt') as f:
        data = json.load(f)
    return {
        str(ep['episode_id']): {
            'trajectory_id': ep.get('trajectory_id', ep['episode_id']),
            'scene_id': ep.get('scene_id', ''),
        }
        for ep in data['episodes']
    }


def inject_positions_from_gt(episodes_raw, gt_trajectories, episodes_meta):
    """Inject position data from GT trajectories (fallback for legacy data)."""
    n_injected = 0
    n_missing = 0

    for ep_id, steps in episodes_raw.items():
        if steps and steps[0].get('agent_x') is not None:
            continue

        gt = gt_trajectories.get(str(ep_id))
        if gt is None:
            n_missing += 1
            continue

        gt_locations = gt['locations']
        for i, step in enumerate(steps):
            if i < len(gt_locations):
                loc = gt_locations[i]
                step['agent_x'] = loc[0]
                step['agent_y'] = loc[1]
                step['agent_z'] = loc[2]
                step['agent_heading'] = 0.0
                n_injected += 1

    print(f"  Position injection: {n_injected} steps, {n_missing} missing")
    return n_injected > 0


# ══════════════════════════════════════════════════════════════════════════════
#  Per-Episode Diagnostic Export
# ══════════════════════════════════════════════════════════════════════════════

def export_episode_diagnostics(episodes_raw, episode_metrics, clean_episodes,
                               episode_categories, reference_paths, output_dir):
    """Export per-step metrics and labels for visualization."""
    out_path = Path(output_dir) / "episode_diagnostics.jsonl"
    n_exported = 0

    with open(out_path, 'w') as f:
        for ep_id in sorted(episode_metrics):
            steps = episodes_raw[ep_id]
            metrics = episode_metrics[ep_id]
            labels = {s['step']: lbl for s, lbl in clean_episodes.get(ep_id, [])}
            category = episode_categories.get(ep_id, 'unknown')
            ref_path = reference_paths[ep_id]
            total_path_len = sum(
                float(np.linalg.norm(ref_path[i + 1] - ref_path[i]))
                for i in range(len(ref_path) - 1)
            )

            for i, step in enumerate(steps):
                m = metrics[i]
                action_prob = step.get('action_prob', None)
                record = {
                    'episode_id': ep_id,
                    'step': step['step'],
                    'action_name': step.get('action_name', ''),
                    'action_prob': round(float(action_prob), 6) if action_prob is not None else None,
                    'agent_x': step.get('agent_x', 0.0),
                    'agent_y': step.get('agent_y', 0.0),
                    'agent_z': step.get('agent_z', 0.0),
                    'target_wp': m['target_idx'],
                    'dist_to_target': round(m['dist_to_target'], 4),
                    'dist_delta': round(m['dist_delta'], 4),
                    'ctd': round(m['ctd'], 4),
                    'advanced': m['advanced'],
                    'is_deviated': m['is_deviated'],
                    'gt_label': labels.get(step['step'], -1),
                    'episode_category': category,
                    'n_waypoints': len(ref_path),
                    'total_path_length': round(total_path_len, 4),
                }
                f.write(json.dumps(record) + '\n')
                n_exported += 1

    print(f"  Episode diagnostics -> {out_path}  ({n_exported} steps)")


# ══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline: GT Labeling + Validation Statistics
# ══════════════════════════════════════════════════════════════════════════════

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()


def run_pipeline(data_dir, dataset_dir, split, output_dir,
                 dist_delta_thresh, gt_patience):
    """GT labeling pipeline: load data -> label -> export statistics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "waypoint_gt_analysis.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    # 1. Load Data
    print(f"\n{'=' * 74}\n  1. Data Loading\n{'=' * 74}")
    episodes_raw = load_anomaly_metrics(data_dir)
    print(f"  Raw episodes: {len(episodes_raw)}")
    print(f"  Total steps : {sum(len(s) for s in episodes_raw.values())}")

    reference_paths = load_reference_paths(dataset_dir, split)
    print(f"  Reference paths: {len(reference_paths)}")

    sample_ep = next(iter(episodes_raw.values()))
    has_pos = sample_ep[0].get('agent_x') is not None
    if not has_pos:
        print("  Position data missing - injecting from GT trajectories...")
        gt_traj = load_gt_trajectories(dataset_dir, split)
        episodes_meta = load_episodes_meta(dataset_dir, split)
        if gt_traj:
            has_pos = inject_positions_from_gt(episodes_raw, gt_traj, episodes_meta)
        if not has_pos:
            print("  ERROR: No position data available.")
            log_file.close()
            sys.stdout = sys.__stdout__
            return

    # 2. GT Labeling
    print(f"\n{'=' * 74}\n  2. GT Labeling (dist_delta_thresh={dist_delta_thresh}m, patience={gt_patience})\n{'=' * 74}")

    clean_episodes, flat_clean_steps, episode_metrics, episode_categories = \
        build_gt_waypoint(episodes_raw, reference_paths, dist_delta_thresh, gt_patience)

    n_only_normal = sum(1 for c in episode_categories.values() if c == 'only_normal')
    n_only_anomaly = sum(1 for c in episode_categories.values() if c == 'only_anomaly')
    n_normal_to_anomaly = sum(1 for c in episode_categories.values() if c == 'normal_to_anomaly')
    n_step_anom = sum(1 for _, l in flat_clean_steps if l == 1)
    n_step_norm = sum(1 for _, l in flat_clean_steps if l == 0)

    print(f"  Episodes: {len(clean_episodes)}")
    print(f"    only_normal      : {n_only_normal} ({n_only_normal/len(clean_episodes)*100:.1f}%)")
    print(f"    only_anomaly     : {n_only_anomaly} ({n_only_anomaly/len(clean_episodes)*100:.1f}%)")
    print(f"    normal_to_anomaly: {n_normal_to_anomaly} ({n_normal_to_anomaly/len(clean_episodes)*100:.1f}%)")
    print(f"  Steps: Normal={n_step_norm}  Anomaly={n_step_anom}")

    # 3. Path Distance Statistics (for GT validation table)
    norm_dd = []
    anom_dd = []
    for ep_id, steps_labels in clean_episodes.items():
        for (step, label) in steps_labels:
            step_idx = episodes_raw[ep_id].index(step)
            dd_val = episode_metrics[ep_id][step_idx]['ctd']
            if label == 1:
                anom_dd.append(dd_val)
            else:
                norm_dd.append(dd_val)

    if norm_dd and anom_dd:
        print(f"\n  Path distance statistics:")
        print(f"    Normal:  median={np.median(norm_dd):.2f}m  mean={np.mean(norm_dd):.4f}m")
        print(f"    Anomaly: median={np.median(anom_dd):.2f}m  mean={np.mean(anom_dd):.4f}m")

        # Cohen's d (path distance)
        import math
        n1, n2 = len(norm_dd), len(anom_dd)
        mu1, mu2 = np.mean(norm_dd), np.mean(anom_dd)
        s1, s2 = np.std(norm_dd, ddof=1), np.std(anom_dd, ddof=1)
        pooled = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        d = abs(mu1 - mu2) / pooled if pooled > 1e-10 else 0.0
        print(f"    Cohen's d: {d:.2f}")

    # 4. Export Diagnostics
    export_episode_diagnostics(
        episodes_raw, episode_metrics, clean_episodes,
        episode_categories, reference_paths, output_dir
    )

    print(f"\nDone. Log: {log_path}")
    log_file.close()
    sys.stdout = sys.__stdout__


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Waypoint GT Labeling for VLN Anomaly Detection")
    parser.add_argument("--data_dir", required=True, help="anomaly_metrics_chunk*.jsonl directory")
    parser.add_argument("--dataset_dir", required=True, help="R2R_VLNCE_v1-3_preprocessed directory")
    parser.add_argument("--split", required=True, choices=["train", "val_seen", "val_unseen"])
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: sibling 3_gt_labeling/ of data_dir)")
    parser.add_argument("--dist_delta_thresh", type=float, default=DEFAULT_DIST_DELTA_THRESH)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    args = parser.parse_args()

    # Default output: sibling 3_gt_labeling/outputs/ of data_dir
    if args.output_dir:
        output_dir = args.output_dir
    elif '2_data_extraction' in args.data_dir:
        output_dir = args.data_dir.replace(
            '2_data_extraction/outputs', '3_gt_labeling/outputs'
        ).replace(
            '2_data_extraction', '3_gt_labeling/outputs'
        )
    else:
        output_dir = str(Path(args.data_dir) / 'outputs')

    run_pipeline(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        split=args.split,
        output_dir=output_dir,
        dist_delta_thresh=args.dist_delta_thresh,
        gt_patience=args.patience,
    )


if __name__ == "__main__":
    main()
