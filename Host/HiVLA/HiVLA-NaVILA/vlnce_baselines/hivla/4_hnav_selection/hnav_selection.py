#!/usr/bin/env python3
"""
Step 4: H_nav Selection — Rank heads by Cohen's d on anomaly sensitivity.

Combines Step 2 (per-head entropy) + Step 3 (GT Normal/Anomaly labels)
to compute Cohen's d per head and produce H_nav head rankings.

Cohen's d = |mean(anomaly_entropy) - mean(normal_entropy)| / pooled_std

Usage:
    python hnav_selection.py
    python hnav_selection.py --entropy_dir ... --gt_dir ... --output_dir ...
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_entropy_data(entropy_dir):
    """
    Load Step 2 output: per-step s_peak per head.

    Cohen's d head ranking uses s_peak (not e_system).
    See 0302 anomaly_analysis.py build_heads().

    Returns:
        dict: {(episode_id, step): {head_key: s_peak}}
    """
    chunk_files = sorted(Path(entropy_dir).glob("anomaly_metrics_chunk*.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(f"No anomaly_metrics_chunk*.jsonl in {entropy_dir}")

    data = {}
    for cf in chunk_files:
        with open(cf) as f:
            for line in f:
                rec = json.loads(line)
                key = (str(rec['episode_id']), int(rec['step']))
                data[key] = {
                    hk: m['s_peak']
                    for hk, m in rec.get('metrics_per_head', {}).items()
                }
    print(f"  Loaded s_peak data: {len(data)} steps from {len(chunk_files)} chunks")
    return data


def load_gt_labels(gt_dir):
    """
    Load Step 3 output: per-step GT label (0=Normal, 1=Anomaly).

    Returns:
        dict: {(episode_id, step): gt_label}
    """
    diag_path = Path(gt_dir) / "episode_diagnostics.jsonl"
    if not diag_path.exists():
        raise FileNotFoundError(f"episode_diagnostics.jsonl not found in {gt_dir}")

    labels = {}
    with open(diag_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('gt_label', -1) == -1:
                continue
            key = (str(rec['episode_id']), int(rec['step']))
            labels[key] = int(rec['gt_label'])

    n_normal = sum(1 for v in labels.values() if v == 0)
    n_anomaly = sum(1 for v in labels.values() if v == 1)
    print(f"  Loaded GT labels: {len(labels)} steps  (Normal={n_normal}, Anomaly={n_anomaly})")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
#  Cohen's d
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    """Cohen's d between two groups a (anomaly) and b (normal)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    mu1, mu2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled < 1e-10:
        return 0.0
    return abs(mu1 - mu2) / pooled


# ══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(entropy_dir, gt_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print(f"\n{'=' * 66}\n  1. Loading Data\n{'=' * 66}")
    entropy_data = load_entropy_data(entropy_dir)
    gt_labels = load_gt_labels(gt_dir)

    # 2. Join on (episode_id, step)
    print(f"\n{'=' * 66}\n  2. Joining entropy + GT labels\n{'=' * 66}")
    head_normal = defaultdict(list)    # head_key -> [s_peak, ...]
    head_anomaly = defaultdict(list)

    n_matched = 0
    for key, label in gt_labels.items():
        if key not in entropy_data:
            continue
        n_matched += 1
        for head_key, s_peak in entropy_data[key].items():
            if label == 0:
                head_normal[head_key].append(s_peak)
            else:
                head_anomaly[head_key].append(s_peak)

    print(f"  Matched steps: {n_matched} / {len(gt_labels)}")
    print(f"  Heads found: {len(head_normal)}")

    # 3. Compute Cohen's d per head
    print(f"\n{'=' * 66}\n  3. Computing Cohen's d\n{'=' * 66}")
    ranked = []
    for head_key in head_normal:
        normal = head_normal[head_key]
        anomaly = head_anomaly.get(head_key, [])
        d = cohens_d(anomaly, normal)
        layer, head = map(int, head_key.split(','))
        ranked.append({
            'layer': layer,
            'head': head,
            'cohens_d': round(d, 6),
            'n_normal': len(normal),
            'n_anomaly': len(anomaly),
            'mean_normal': round(float(np.mean(normal)), 6) if normal else 0.0,
            'mean_anomaly': round(float(np.mean(anomaly)), 6) if anomaly else 0.0,
        })

    ranked.sort(key=lambda x: x['cohens_d'], reverse=True)
    for i, entry in enumerate(ranked):
        entry['rank'] = i + 1

    # 4. Save top-20 CSV
    top20 = ranked[:20]
    csv_path = output_dir / "head_importance.csv"
    fieldnames = ['rank', 'layer', 'head', 'cohens_d', 'n_normal', 'n_anomaly',
                  'mean_normal', 'mean_anomaly']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top20)

    print(f"\n  Saved: {csv_path}  (top 20 / {len(ranked)} heads)")
    print("\n  Top-20 H_nav heads:")
    for e in top20:
        print(
            f"    #{e['rank']:2d}  L{e['layer']:2d}H{e['head']:2d}  "
            f"d={e['cohens_d']:.4f}  "
            f"(N_norm={e['n_normal']}, N_anom={e['n_anomaly']})"
        )

    # 5. Auto-select K via elbow criterion (Delta% < 25%)
    print(f"\n{'=' * 66}\n  4. Auto K Selection (elbow: last Delta% > 25%)\n{'=' * 66}")
    cumul = 0.0
    selected_k = 1
    for i, e in enumerate(top20):
        cumul += e['cohens_d']
        delta_pct = e['cohens_d'] / cumul * 100
        mark = ""
        if delta_pct > 25.0:
            selected_k = i + 1
            mark = " ← K"
        print(f"    K={i+1:>2}  d={e['cohens_d']:.4f}  cumul={cumul:.4f}  "
              f"Delta%={delta_pct:.1f}%{mark}")
        if delta_pct < 15.0:
            break

    print(f"\n  Selected K={selected_k} (last rank with Delta% > 25%)")

    # Save config JSON for Step 5
    config = {
        "selected_k": selected_k,
        "head_order": [f"{e['layer']},{e['head']}" for e in top20],
        "top_k_heads": [[e['layer'], e['head']] for e in top20[:selected_k]],
        "elbow_criterion": "last rank where Delta% > 25%",
    }
    config_path = output_dir / "hnav_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="H_nav Selection: Cohen's d head ranking")
    parser.add_argument(
        "--entropy_dir",
        default="eval_out/hivla/2_data/train",
        help="Step 2 output dir (anomaly_metrics_chunk*.jsonl)",
    )
    parser.add_argument(
        "--gt_dir",
        default="eval_out/hivla/3_gt/train",
        help="Step 3 output dir (episode_diagnostics.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_out/hivla/4_hnav/outputs",
        help="Output directory for head_importance.csv",
    )
    args = parser.parse_args()

    run_pipeline(args.entropy_dir, args.gt_dir, args.output_dir)


if __name__ == "__main__":
    main()
