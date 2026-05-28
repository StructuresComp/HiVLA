#!/usr/bin/env python3
"""
Step 5: RelDiff Grid Search — Sweep (W, K, P, τ) for anomaly detection.

Grid search over all (W, K, P, τ) combinations. Best config selected by
max Gap at FER 5-10%. Ablation studies sweep one variable at a time
around the best config.

Combines:
  - Step 2 output (per-head entropy_list from anomaly_metrics)
  - Step 3 output (GT labels from episode_diagnostics.jsonl)
  - Step 4 output (head ranking from head_importance.csv)

Detection signal: RelDiff
  R_t = mean_entropy_t / rolling_mean(mean_entropy_{t-W..t-1})
  where mean_entropy_t = Mean(mean(entropy_list) over top-K heads)

Sweep: W=1..10 × K=1..10 × P=1..10 × τ (natural + fixed grid)

Usage:
    python grid_search.py
    python grid_search.py --entropy_dir ... --gt_dir ... --head_csv ... --output_dir ...
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_EPS = 1e-6
K_GROUP = 10
K_TOP_VALS = list(range(1, K_GROUP + 1))
PATIENCE_VALS = list(range(1, 11))
WINDOW_VALS = list(range(1, 11))
_T_ENT = [0.85, 0.90, 0.93, 0.95, 0.97, 1.00, 1.02, 1.05]

CATEGORIES = ['only_normal', 'only_anomaly', 'normal_to_anomaly']
CAT_LABELS = {
    'only_normal': 'Normal-Only',
    'only_anomaly': 'Anomaly-Only',
    'normal_to_anomaly': 'N→A',
    'all': 'All',
}


# ══════════════════════════════════════════════════════════════════════════════
#  Logging
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


def _banner(title="", width=74):
    print(f"\n{'=' * width}")
    if title:
        print(f"  {title}")
        print(f"{'=' * width}")


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_anomaly_metrics(entropy_dir):
    """Load Step 2 output: full per-step metrics (entropy_list, s_peak, etc.)."""
    chunk_files = sorted(Path(entropy_dir).glob("anomaly_metrics_chunk*.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(f"No anomaly_metrics_chunk*.jsonl in {entropy_dir}")

    episodes = defaultdict(list)
    for cf in chunk_files:
        with open(cf) as f:
            for line in f:
                rec = json.loads(line)
                episodes[str(rec['episode_id'])].append(rec)

    # Sort steps within each episode
    for ep_id in episodes:
        episodes[ep_id].sort(key=lambda r: int(r['step']))

    print(f"  Loaded: {len(episodes)} episodes, "
          f"{sum(len(s) for s in episodes.values())} steps "
          f"from {len(chunk_files)} chunks")
    return dict(episodes)


def load_gt_labels(gt_dir):
    """Load Step 3 output: per-step GT label + episode categories."""
    diag_path = Path(gt_dir) / "episode_diagnostics.jsonl"
    if not diag_path.exists():
        raise FileNotFoundError(f"episode_diagnostics.jsonl not found in {gt_dir}")

    # Per-step labels
    step_labels = {}  # {(episode_id, step): gt_label}
    with open(diag_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('gt_label', -1) == -1:
                continue
            key = (str(rec['episode_id']), int(rec['step']))
            step_labels[key] = int(rec['gt_label'])

    # Derive episode categories
    ep_labels = defaultdict(set)
    for (ep_id, _), label in step_labels.items():
        ep_labels[ep_id].add(label)

    episode_categories = {}
    for ep_id, labels in ep_labels.items():
        if labels == {0}:
            episode_categories[ep_id] = 'only_normal'
        elif labels == {1}:
            episode_categories[ep_id] = 'only_anomaly'
        else:
            episode_categories[ep_id] = 'normal_to_anomaly'

    n_n = sum(1 for v in episode_categories.values() if v == 'only_normal')
    n_a = sum(1 for v in episode_categories.values() if v == 'only_anomaly')
    n_na = sum(1 for v in episode_categories.values() if v == 'normal_to_anomaly')
    print(f"  GT labels: {len(step_labels)} steps, {len(episode_categories)} episodes "
          f"(N={n_n}, A={n_a}, N→A={n_na})")
    return step_labels, episode_categories


def load_head_ranking(head_csv):
    """Load Step 4 output: head ranking by Cohen's d."""
    rows = []
    with open(head_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(f"{row['layer']},{row['head']}")
    print(f"  Head ranking: {len(rows)} heads from {head_csv}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Build Joined Data
# ══════════════════════════════════════════════════════════════════════════════

def build_clean_episodes(episodes_raw, step_labels):
    """Join Step 2 metrics with Step 3 GT labels."""
    clean = {}
    for ep_id, steps in episodes_raw.items():
        joined = []
        for step in steps:
            key = (str(step['episode_id']), int(step['step']))
            if key not in step_labels:
                continue
            joined.append((step, step_labels[key]))
        if joined:
            clean[ep_id] = joined
    n_steps = sum(len(s) for s in clean.values())
    print(f"  Joined: {len(clean)} episodes, {n_steps} steps")
    return clean


def build_ep_mean_ent(clean_episodes, head_order):
    """Precompute per-episode entropy buffer using given head order.

    For each step, computes mean(entropy_list) per head → detection signal.
    """
    ep_mean_ent = {}
    for ep_id, steps_labels in clean_episodes.items():
        rows = []
        for step, label in steps_labels:
            mph = step["metrics_per_head"]
            vals = np.full(len(head_order), np.nan, dtype=np.float64)
            for j, h in enumerate(head_order):
                if h in mph:
                    el = mph[h].get("entropy_list", [])
                    if el:
                        vals[j] = float(np.mean(el))
            rows.append((label, vals))
        ep_mean_ent[ep_id] = rows
    return ep_mean_ent


# ══════════════════════════════════════════════════════════════════════════════
#  RelDiff Detection
# ══════════════════════════════════════════════════════════════════════════════

def _rel_diff_w(buf, w):
    if len(buf) < w + 1:
        return None
    return float(buf[-1]) / (float(np.mean(buf[-(w + 1):-1])) + _EPS)


def nat_thresh(ep_mean_ent, k_top, w):
    """Compute natural threshold = midpoint of normal vs anomaly RelDiff means."""
    nv, av = [], []
    for rows in ep_mean_ent.values():
        buf = []
        for label, all_vals in rows:
            valid = all_vals[:k_top][~np.isnan(all_vals[:k_top])]
            if len(valid) == 0:
                continue
            buf.append(float(np.mean(valid)))
            v = _rel_diff_w(buf, w)
            if v is None:
                continue
            (nv if label == 0 else av).append(v)
    if not nv or not av:
        return None, "<"
    nm, am = float(np.mean(nv)), float(np.mean(av))
    return (nm + am) / 2.0, (">" if am > nm else "<")


def eval_one(ep_mean_ent, k_top, threshold, patience, nat_dir, w):
    """Episode-level detection evaluation."""
    n_anom = n_norm = n_det = n_fp = 0
    lats = []
    for ep_id, rows in ep_mean_ent.items():
        onset = next((i for i, (l, _) in enumerate(rows) if l == 1), None)
        has_a = onset is not None
        if has_a:
            n_anom += 1
        else:
            n_norm += 1
        consec = 0; fired = False; buf = []
        for i, (label, all_vals) in enumerate(rows):
            if fired:
                break
            valid = all_vals[:k_top][~np.isnan(all_vals[:k_top])]
            if len(valid) == 0:
                continue
            buf.append(float(np.mean(valid)))
            score = _rel_diff_w(buf, w)
            if score is None:
                continue
            cond = score > threshold if nat_dir == ">" else score < threshold
            consec = consec + 1 if cond else 0
            if consec >= patience:
                fired = True
                if has_a and i >= onset:
                    n_det += 1; lats.append(i - onset)
                elif not has_a:
                    n_fp += 1
    return n_anom, n_norm, n_det, n_fp, lats


def step_confusion(ep_mean_ent, k_top, threshold, patience, nat_dir, w):
    """Step-level confusion matrix."""
    TT = TF = FT = FF = 0
    for ep_id, rows in ep_mean_ent.items():
        consec = 0; buf = []; alarm = False
        for i, (label, all_vals) in enumerate(rows):
            valid = all_vals[:k_top][~np.isnan(all_vals[:k_top])]
            if len(valid) == 0:
                continue
            buf.append(float(np.mean(valid)))
            score = _rel_diff_w(buf, w)
            if score is not None:
                cond = score > threshold if nat_dir == ">" else score < threshold
                consec = consec + 1 if cond else 0
                alarm = consec >= patience
            else:
                alarm = False
            if label == 1 and alarm:       TT += 1
            elif label == 1 and not alarm: TF += 1
            elif label == 0 and alarm:     FT += 1
            else:                          FF += 1
    return TT, TF, FT, FF


def _filter_ep(ep_mean_ent, episode_categories, cat):
    if cat == 'all':
        return ep_mean_ent
    return {eid: rows for eid, rows in ep_mean_ent.items()
            if episode_categories.get(eid) == cat}


def eval_by_category(ep_mean_ent, episode_categories, k_top, threshold, patience, nat_dir, w):
    results = {}
    for cat in CATEGORIES + ['all']:
        sub = _filter_ep(ep_mean_ent, episode_categories, cat)
        if not sub:
            results[cat] = None
            continue
        n_anom, n_norm, n_det, n_fp, lats = eval_one(sub, k_top, threshold, patience, nat_dir, w)
        TT, TF, FT, FF = step_confusion(sub, k_top, threshold, patience, nat_dir, w)
        edr = n_det / n_anom if n_anom > 0 else 0.0
        fer = n_fp / n_norm if n_norm > 0 else 0.0
        prec = TT / (TT + FT) if (TT + FT) > 0 else 0.0
        recall = TT / (TT + TF) if (TT + TF) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        lm = float(np.mean(lats)) if lats else float('nan')
        lmd = float(np.median(lats)) if lats else float('nan')
        results[cat] = {
            'n_eps': len(sub), 'n_anom': n_anom, 'n_norm': n_norm,
            'n_det': n_det, 'n_fp': n_fp,
            'EDR': edr, 'FER': fer, 'Gap': edr - fer,
            'LatMean': lm, 'LatMedian': lmd,
            'TT': TT, 'TF': TF, 'FT': FT, 'FF': FF,
            'precision': prec, 'recall': recall, 'f1': f1,
        }
    return results


def _print_elbow_summary(head_csv, selected_k):
    """Print Cohen's d elbow justification from head_importance.csv."""
    rows = []
    with open(head_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'rank': int(row['rank']),
                'layer': int(row['layer']),
                'head': int(row['head']),
                'd': float(row['cohens_d']),
            })
    print(f"  K selection: Cohen's d elbow criterion (Step 4)")
    # Show top K+2 heads with delta%
    show_n = min(selected_k + 3, len(rows))
    for i in range(show_n):
        r = rows[i]
        if i == 0:
            delta_s = "      "
        else:
            delta_pct = (rows[i-1]['d'] - r['d']) / rows[i-1]['d'] * 100
            delta_s = f"Δ={delta_pct:+.1f}%"
        marker = " ◀ elbow" if i == selected_k else ""
        print(f"    Rank {r['rank']:>2}: L{r['layer']}H{r['head']:>2}  "
              f"d={r['d']:.3f}  {delta_s}{marker}")
    if selected_k < len(rows):
        drop = (rows[selected_k-1]['d'] - rows[selected_k]['d']) / rows[selected_k-1]['d'] * 100
        print(f"  → Drop at rank {selected_k}→{selected_k+1}: {drop:.1f}% "
              f"(d={rows[selected_k-1]['d']:.3f} → {rows[selected_k]['d']:.3f})")
    print(f"  → Selected K={selected_k}")


def _print_cat_table(cat_results, header=""):
    if header:
        print(f"\n  {header}")
    fmt = ("    {cat:<16s} eps={n:>4d}  EDR={edr:>6.1f}%  FER={fer:>6.1f}%  "
           "Gap={gap:>+6.1f}%  Lat={lat:>5s}  Prec={prec:>5.1f}%  F1={f1:>5.1f}%")
    for cat in CATEGORIES + ['all']:
        r = cat_results.get(cat)
        if r is None:
            continue
        lm_s = f"{r['LatMean']:.1f}" if not math.isnan(r['LatMean']) else "-"
        print(fmt.format(
            cat=CAT_LABELS[cat], n=r['n_eps'],
            edr=r['EDR'] * 100, fer=r['FER'] * 100, gap=r['Gap'] * 100,
            lat=lm_s, prec=r['precision'] * 100, f1=r['f1'] * 100,
        ))


# ══════════════════════════════════════════════════════════════════════════════
#  Ablation Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ablation_row(cat_r):
    """Build a flat dict with per-category columns for CSV export."""
    r_all = cat_r['all']
    row = {
        "EDR": round(r_all['EDR'], 4), "FER": round(r_all['FER'], 4),
        "Gap": round(r_all['Gap'], 4),
        "LatMean": round(r_all['LatMean'], 2) if not math.isnan(r_all['LatMean']) else None,
        "N_det": r_all['n_det'], "N_anom": r_all['n_anom'],
        "N_fp": r_all['n_fp'], "N_norm": r_all['n_norm'],
        "TT": r_all['TT'], "TF": r_all['TF'], "FT": r_all['FT'], "FF": r_all['FF'],
        "precision": round(r_all['precision'], 4),
        "recall": round(r_all['recall'], 4),
        "f1": round(r_all['f1'], 4),
    }
    for cat in CATEGORIES:
        cr = cat_r[cat]
        prefix = cat.replace('normal_to_anomaly', 'na').replace('only_anomaly', 'oa').replace('only_normal', 'on')
        if cr:
            row[f"{prefix}_EDR"] = round(cr['EDR'], 4)
            row[f"{prefix}_FER"] = round(cr['FER'], 4)
            row[f"{prefix}_Gap"] = round(cr['Gap'], 4)
            row[f"{prefix}_Prec"] = round(cr['precision'], 4)
            row[f"{prefix}_F1"] = round(cr['f1'], 4)
            row[f"{prefix}_LatMean"] = round(cr['LatMean'], 2) if not math.isnan(cr['LatMean']) else None
            row[f"{prefix}_n_det"] = cr['n_det']
            row[f"{prefix}_n_anom"] = cr['n_anom']
        else:
            for sfx in ['EDR', 'FER', 'Gap', 'Prec', 'F1', 'LatMean', 'n_det', 'n_anom']:
                row[f"{prefix}_{sfx}"] = None
    return row


def _save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(entropy_dir, gt_dir, head_csv, output_dir,
                 ab_k=3, ab_w=10, ab_p=9, ab_tau=0.95, skip_sweep=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "grid_search.log"
    log_file = open(log_path, 'w')
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, log_file)

    try:
        _run(entropy_dir, gt_dir, head_csv, output_dir, ab_k, ab_w, ab_p, ab_tau, skip_sweep)
    finally:
        sys.stdout = old_stdout
        log_file.close()
        print(f"\nLog: {log_path}")


def _load_full_sweep(path):
    """Load existing full_sweep.csv."""
    results = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            results.append({
                "Window": int(r['Window']), "K_top": int(r['K_top']),
                "Thresh": float(r['Thresh']), "Patience": int(r['Patience']),
                "Natural": r['Natural'] == 'True',
                "EDR": float(r['EDR']), "FER": float(r['FER']),
                "Gap": float(r['Gap']),
                "LatMean": float(r['LatMean']) if r['LatMean'] not in ('', 'nan') else float('nan'),
                "LatMedian": float(r['LatMedian']) if r['LatMedian'] not in ('', 'nan') else float('nan'),
                "N_det": int(r['N_det']), "N_anom": int(r['N_anom']),
                "N_fp": int(r['N_fp']), "N_norm": int(r['N_norm']),
            })
    return results


def _run(entropy_dir, gt_dir, head_csv, output_dir,
         ab_k=3, ab_w=10, ab_p=9, ab_tau=0.95, skip_sweep=False):
    # 1. Load data
    _banner("1. Loading Data")
    episodes_raw = load_anomaly_metrics(entropy_dir)
    step_labels, episode_categories = load_gt_labels(gt_dir)
    head_order = load_head_ranking(head_csv)

    # 2. Join
    _banner("2. Joining metrics + GT labels")
    clean_episodes = build_clean_episodes(episodes_raw, step_labels)

    n_ep_anom = sum(1 for c in episode_categories.values() if c != 'only_normal')
    n_ep_norm = sum(1 for c in episode_categories.values() if c == 'only_normal')

    # 3. Build entropy buffer
    _banner("3. Precompute — cross-step mean entropy buffer")
    ep_mean_ent = build_ep_mean_ent(clean_episodes, head_order)
    print(f"  ep_mean_ent built — {len(ep_mean_ent)} episodes")

    # 4. RelDiff Sweep
    sweep_path = output_dir / "full_sweep.csv"
    if skip_sweep and sweep_path.exists():
        _banner("4. RelDiff Sweep — SKIPPED (loading existing full_sweep.csv)")
        all_results = _load_full_sweep(sweep_path)
        print(f"  Loaded: {len(all_results)} configs from {sweep_path}")
    else:
        _banner(f"4. RelDiff Sweep — W=1..{max(WINDOW_VALS)} × K=1..{K_GROUP} × P=1..{max(PATIENCE_VALS)}")
        print(f"  A_eps={n_ep_anom}  N_eps={n_ep_norm}")

        all_results = []
        for W in WINDOW_VALS:
            print(f"  [W={W}] sweeping K×thresh×P ...", end="", flush=True)
            w_count = 0
            for k_top in K_TOP_VALS:
                nat_t, nat_dir = nat_thresh(ep_mean_ent, k_top, W)
                thresh_grid = sorted(set(
                    ([round(nat_t, 6)] if nat_t is not None else []) + _T_ENT
                ))
                for threshold in thresh_grid:
                    for p in PATIENCE_VALS:
                        n_anom, n_norm, n_det, n_fp, lats = eval_one(
                            ep_mean_ent, k_top, threshold, p, nat_dir, W)
                        edr = n_det / n_anom if n_anom > 0 else 0.0
                        fer = n_fp / n_norm if n_norm > 0 else 0.0
                        lm = float(np.mean(lats)) if lats else float("nan")
                        lmd = float(np.median(lats)) if lats else float("nan")
                        all_results.append({
                            "Window": W, "K_top": k_top, "Thresh": threshold,
                            "Patience": p,
                            "Natural": nat_t is not None and abs(threshold - nat_t) < 1e-6,
                            "EDR": edr, "FER": fer, "Gap": edr - fer,
                            "LatMean": lm, "LatMedian": lmd,
                            "N_det": n_det, "N_anom": n_anom,
                            "N_fp": n_fp, "N_norm": n_norm,
                        })
                        w_count += 1
            print(f" {w_count} configs done")

        _save_csv(output_dir / "full_sweep.csv", all_results)

    # 5. Best Config — best (W, K, P, τ) from full grid, FER 5-10%, max Gap
    _banner("5. Best Config (FER 5-10%, max Gap)")
    best_filt = sorted(
        [r for r in all_results if 0.05 <= r["FER"] <= 0.10],
        key=lambda r: (-r["Gap"], r["FER"]))
    if not best_filt:
        best_filt = sorted(
            [r for r in all_results if 0 < r["FER"] <= 0.10],
            key=lambda r: (-r["Gap"], r["FER"]))

    if best_filt:
        BEST = best_filt[0]
        BEST_W, BEST_K, BEST_P = BEST["Window"], BEST["K_top"], BEST["Patience"]
        BEST_THRESH = BEST["Thresh"]
        print(f"  Best: W={BEST_W}, K={BEST_K}, P={BEST_P}, τ={BEST_THRESH:.6f}")
        print(f"  EDR={BEST['EDR']:.4f}  FER={BEST['FER']:.4f}  "
              f"Gap={BEST['Gap']:+.4f}")

        print(f"\n  Top-10 FER 5-10% (by Gap):")
        for rank, r in enumerate(best_filt[:10], 1):
            lm = f"{r['LatMean']:.1f}" if not math.isnan(r['LatMean']) else "-"
            print(f"    {rank:>2}. W={r['Window']:>2} K={r['K_top']:>2} "
                  f"P={r['Patience']:>2} T={r['Thresh']:.6f}  "
                  f"EDR={r['EDR']:.4f} FER={r['FER']:.4f} Gap={r['Gap']:+.4f} L={lm}")
    else:
        BEST_W, BEST_K, BEST_P = 10, 2, 10
        BEST_THRESH = 0.85
        print(f"  WARNING: No config with FER 5-10%. "
              f"Fallback: W={BEST_W} K={BEST_K} P={BEST_P}")

    # Ablation base = best config from grid search
    AB_K, AB_W, AB_P, AB_TAU = BEST_K, BEST_W, BEST_P, BEST_THRESH
    print(f"\n  Ablation base (from best config): W={AB_W}, K={AB_K}, P={AB_P}, τ={AB_TAU:.6f}")

    # Save best config for downstream steps (Step 6, 7, 8)
    # heads: top-K from head_order, converted to [layer, head] pairs
    top_k_heads = []
    for h_str in head_order[:AB_K]:
        l, h = h_str.split(",")
        top_k_heads.append([int(l), int(h)])

    best_cfg_path = output_dir / "best_config.json"
    with open(best_cfg_path, "w") as f:
        json.dump({
            "K": AB_K, "W": AB_W, "P": AB_P, "tau": round(AB_TAU, 6),
            "heads": top_k_heads,
            "EDR": round(BEST['EDR'], 4) if best_filt else None,
            "FER": round(BEST['FER'], 4) if best_filt else None,
            "Gap": round(BEST['Gap'], 4) if best_filt else None,
        }, f, indent=2)
    print(f"  Saved: {best_cfg_path}")

    # Cohen's d reference
    _print_elbow_summary(head_csv, ab_k)

    # Per-category breakdown for best operating point
    _banner("5Z. Best Operating Point — Per-Category Breakdown")
    _, nd = nat_thresh(ep_mean_ent, BEST_K, BEST_W)
    cat_r = eval_by_category(
        ep_mean_ent, episode_categories, BEST_K, BEST_THRESH, BEST_P, nd, BEST_W)
    _print_cat_table(cat_r, f"Best: W={BEST_W}, K={BEST_K}, P={BEST_P}, τ={BEST_THRESH:.6f}")

    # ── K Ablation: best-τ-per-K (fix W, P from best config) ──
    _banner(f"5K. K Ablation — best τ per K (W={AB_W}, P={AB_P} fixed, FER 5-10%, max Gap)")
    best_per_k = {}
    for r in all_results:
        if r['Window'] != AB_W or r['Patience'] != AB_P:
            continue
        k = r['K_top']
        if 0.05 <= r['FER'] <= 0.10:
            if k not in best_per_k or r['Gap'] > best_per_k[k]['Gap']:
                best_per_k[k] = r

    # Fallback: FER<=10% for K values with no 5-10% config
    for r in all_results:
        if r['Window'] != AB_W or r['Patience'] != AB_P:
            continue
        k = r['K_top']
        if k in best_per_k:
            continue
        if 0 < r['FER'] <= 0.10:
            if k not in best_per_k or r['Gap'] > best_per_k[k]['Gap']:
                best_per_k[k] = r

    kbest_rows = []
    for k in sorted(best_per_k.keys()):
        cfg = best_per_k[k]
        w, p, t = cfg['Window'], cfg['Patience'], cfg['Thresh']
        _, nd = nat_thresh(ep_mean_ent, k, w)
        cat_r = eval_by_category(ep_mean_ent, episode_categories, k, t, p, nd, w)
        r_all = cat_r['all']
        r_na = cat_r.get('normal_to_anomaly')
        lm_s = f"{r_all['LatMean']:.1f}" if not math.isnan(r_all['LatMean']) else "-"
        na_edr = r_na['EDR'] if r_na else 0.0
        na_lat = r_na['LatMean'] if r_na else float('nan')
        na_lat_s = f"{na_lat:.1f}" if not math.isnan(na_lat) else "-"
        marker = "  ◀ best" if k == AB_K else ""
        print(f"  K={k:>2}  τ={t:.4f}  "
              f"EDR={r_all['EDR']:.3f}  FER={r_all['FER']:.3f}  "
              f"Gap={r_all['Gap']:+.3f}  Lat={lm_s}  |  "
              f"N→A: EDR={na_edr:.3f}  Lat={na_lat_s}{marker}")
        row = {"K": k, "W": w, "P": p, "thresh": t}
        row.update(_ablation_row(cat_r))
        kbest_rows.append(row)
    _save_csv(output_dir / "ablation_K_bestgap.csv", kbest_rows)

    # ── Fixed-threshold ablations (one variable at a time) ──
    # Base: from best config (AB_K, AB_W, AB_P, AB_TAU)

    # Determine nat_dir for the base config
    _, ab_nat_dir = nat_thresh(ep_mean_ent, AB_K, AB_W)

    # 6. K Ablation (fix W, P, τ)
    _banner(f"6. Ablation: K sweep (W={AB_W}, P={AB_P}, τ={AB_TAU:.4f})")
    k_rows = []
    for k in range(1, K_GROUP + 1):
        _, nd_k = nat_thresh(ep_mean_ent, k, AB_W)
        cat_r = eval_by_category(
            ep_mean_ent, episode_categories, k, AB_TAU, AB_P, nd_k, AB_W)
        r_all = cat_r['all']
        r_na = cat_r['normal_to_anomaly']
        lm_s = f"{r_all['LatMean']:.1f}" if not math.isnan(r_all['LatMean']) else "-"
        na_edr = r_na['EDR'] if r_na else 0.0
        na_lat = r_na['LatMean'] if r_na else float('nan')
        na_lat_s = f"{na_lat:.1f}" if not math.isnan(na_lat) else "-"
        print(f"  K={k:>2}  EDR={r_all['EDR']:.3f}  FER={r_all['FER']:.3f}  "
              f"Gap={r_all['Gap']:+.3f}  Lat={lm_s}  Prec={r_all['precision']:.3f}  |  "
              f"N→A: EDR={na_edr:.3f}  Lat={na_lat_s}")
        row = {"K": k, "W": AB_W, "P": AB_P, "thresh": AB_TAU}
        row.update(_ablation_row(cat_r))
        k_rows.append(row)
    _save_csv(output_dir / "ablation_K.csv", k_rows)

    # 7. W Ablation (fix K, P, τ)
    _banner(f"7. Ablation: W sweep (K={AB_K}, P={AB_P}, τ={AB_TAU:.4f})")
    w_rows = []
    for w in WINDOW_VALS:
        _, nd_w = nat_thresh(ep_mean_ent, AB_K, w)
        cat_r = eval_by_category(
            ep_mean_ent, episode_categories, AB_K, AB_TAU, AB_P, nd_w, w)
        r_all = cat_r['all']
        r_na = cat_r['normal_to_anomaly']
        lm_s = f"{r_all['LatMean']:.1f}" if not math.isnan(r_all['LatMean']) else "-"
        na_edr = r_na['EDR'] if r_na else 0.0
        na_lat = r_na['LatMean'] if r_na else float('nan')
        na_lat_s = f"{na_lat:.1f}" if not math.isnan(na_lat) else "-"
        print(f"  W={w:>2}  EDR={r_all['EDR']:.3f}  FER={r_all['FER']:.3f}  "
              f"Gap={r_all['Gap']:+.3f}  Lat={lm_s}  |  N→A: EDR={na_edr:.3f}  Lat={na_lat_s}")
        row = {"W": w, "K": AB_K, "P": AB_P, "thresh": AB_TAU}
        row.update(_ablation_row(cat_r))
        w_rows.append(row)
    _save_csv(output_dir / "ablation_W.csv", w_rows)

    # 8. P Ablation (fix K, W, τ)
    _banner(f"8. Ablation: P sweep (K={AB_K}, W={AB_W}, τ={AB_TAU:.4f})")
    p_rows = []
    for p in range(1, 11):
        cat_r = eval_by_category(
            ep_mean_ent, episode_categories, AB_K, AB_TAU, p, ab_nat_dir, AB_W)
        r_all = cat_r['all']
        r_na = cat_r['normal_to_anomaly']
        lm_s = f"{r_all['LatMean']:.1f}" if not math.isnan(r_all['LatMean']) else "-"
        na_edr = r_na['EDR'] if r_na else 0.0
        na_lat = r_na['LatMean'] if r_na else float('nan')
        na_lat_s = f"{na_lat:.1f}" if not math.isnan(na_lat) else "-"
        print(f"  P={p:>2}  EDR={r_all['EDR']:.3f}  FER={r_all['FER']:.3f}  "
              f"Gap={r_all['Gap']:+.3f}  Lat={lm_s}  |  N→A: EDR={na_edr:.3f}  Lat={na_lat_s}")
        row = {"P": p, "W": AB_W, "K": AB_K, "thresh": AB_TAU}
        row.update(_ablation_row(cat_r))
        p_rows.append(row)
    _save_csv(output_dir / "ablation_P.csv", p_rows)

    # 9. Threshold Sweep (fix K, W, P)
    _banner(f"9. Threshold sweep (K={AB_K}, W={AB_W}, P={AB_P})")
    tau_grid = sorted(set(
        [round(x, 4) for x in np.arange(0.85, 1.051, 0.005)] +
        [AB_TAU, BEST_THRESH]
    ))
    tau_rows = []
    for tau in tau_grid:
        cat_r = eval_by_category(
            ep_mean_ent, episode_categories, AB_K, tau, AB_P, ab_nat_dir, AB_W)
        r_all = cat_r['all']
        if r_all is None:
            continue
        r_na = cat_r['normal_to_anomaly']
        lm_s = f"{r_all['LatMean']:.1f}" if not math.isnan(r_all['LatMean']) else "-"
        is_nat = abs(tau - AB_TAU) < 1e-6
        row = {
            "threshold": tau, "is_natural": is_nat,
            "K": AB_K, "W": AB_W, "P": AB_P,
        }
        row.update(_ablation_row(cat_r))
        tau_rows.append(row)
    _save_csv(output_dir / "threshold_sweep.csv", tau_rows)
    print(f"  {len(tau_rows)} thresholds swept")

    # Summary
    _banner("SUMMARY")
    print(f"  Best Config (FER 5-10%, max Gap):")
    print(f"    W={BEST_W}, K={BEST_K}, P={BEST_P}, τ={BEST_THRESH:.6f}")
    if best_filt:
        print(f"    EDR={BEST['EDR']:.4f}  FER={BEST['FER']:.4f}  "
              f"Gap={BEST['Gap']:+.4f}")
    print(f"  Ablation base = best config above")
    print(f"  Cohen's d elbow (Step 4): K={ab_k}")
    print(f"  Head order (top-{K_GROUP}): {head_order[:K_GROUP]}")
    print(f"\n  Output dir: {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 5: RelDiff Grid Search")
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
        "--head_csv",
        default="eval_out/hivla/4_hnav/outputs/head_importance.csv",
        help="Step 4 output (head_importance.csv)",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_out/hivla/5_grid/outputs",
        help="Output directory",
    )
    parser.add_argument("--ab_k", type=int, default=None, help="Ablation base K (auto from Step 4 if omitted)")
    parser.add_argument("--ab_w", type=int, default=10, help="Ablation base W (default: 10)")
    parser.add_argument("--ab_p", type=int, default=9, help="Ablation base P (default: 9)")
    parser.add_argument("--ab_tau", type=float, default=0.95, help="Ablation base τ (default: 0.95)")
    parser.add_argument("--skip_sweep", action="store_true",
                        help="Skip full sweep; load existing full_sweep.csv")
    args = parser.parse_args()

    # Auto-load K from Step 4's hnav_config.json if not specified
    ab_k = args.ab_k
    if ab_k is None:
        config_path = Path(args.head_csv).parent / "hnav_config.json"
        if config_path.exists():
            with open(config_path) as f:
                hnav_cfg = json.load(f)
            ab_k = hnav_cfg["selected_k"]
            print(f"  Auto K={ab_k} from {config_path}")
        else:
            ab_k = 3
            print(f"  hnav_config.json not found, using default K={ab_k}")

    run_pipeline(args.entropy_dir, args.gt_dir, args.head_csv, args.output_dir,
                 ab_k=ab_k, ab_w=args.ab_w, ab_p=args.ab_p, ab_tau=args.ab_tau,
                 skip_sweep=args.skip_sweep)


if __name__ == "__main__":
    main()
