#!/usr/bin/env python3
"""
Comparison Study: Stagnation vs Act.Failure vs NaVILA (Ours)
═════════════════════════════════════════════════════════════

Reads pre-computed GT labels from episode_diagnostics.jsonl (Step 3 output)
instead of recomputing GT from reference paths.  This makes it possible to
run the comparison on any split once Steps 2+3 have been run for that split.

GT label convention in episode_diagnostics.jsonl:
    gt_label = -1  → rotation step, excluded from evaluation
    gt_label =  0  → Normal forward step
    gt_label =  1  → Anomaly forward step

Methods compared:
    Stagnation   : no position change over W consecutive steps
    Act.Failure  : MOVE_FORWARD with position delta < 0.01 m
    Ours (NaVILA): attention-entropy RelDiff on H_nav heads
                   (requires anomaly_metrics_chunk*.jsonl from Step 2)

Usage:
    # Just stagnation + act.failure (no attention metrics needed):
    python compare_from_gt.py \\
        --gt_file eval_out/hivla/3_gt/val_seen/episode_diagnostics.jsonl \\
        --split val_seen

    # All three methods (with NaVILA entropy):
    python compare_from_gt.py \\
        --gt_file  eval_out/hivla_3_gt/val_seen/episode_diagnostics.jsonl \\
        --metrics_dir eval_out/hivla_2_data/val_seen \\
        --split val_seen

    # Both splits (pass separate --gt_file per invocation, or use --gt_seen / --gt_unseen):
    python compare_from_gt.py --gt_seen <path> --gt_unseen <path>
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROTATION_ACTIONS = {"TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"}
CATEGORIES = ['only_normal', 'only_anomaly', 'normal_to_anomaly']
_EPS = 1e-6


# ══════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════

def load_episode_diagnostics(jsonl_path):
    """Load episode_diagnostics.jsonl → {ep_id: [step_dict, ...]}."""
    episodes = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            episodes[str(r['episode_id'])].append(r)
    # Deduplicate by step index (guards against mixed-chunk episode_diagnostics)
    result = {}
    for ep_id, steps in episodes.items():
        seen: dict = {}
        for s in steps:
            seen[s['step']] = s  # last occurrence wins
        result[ep_id] = sorted(seen.values(), key=lambda s: s['step'])
    return result


def load_anomaly_metrics(data_dir):
    """Load anomaly_metrics_chunk*.jsonl → {ep_id: [step_dict, ...]}."""
    chunk_files = sorted(Path(data_dir).glob("anomaly_metrics_chunk*.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(f"No anomaly_metrics_chunk*.jsonl in {data_dir}")
    episodes = defaultdict(list)
    for cf in chunk_files:
        with open(cf) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                episodes[str(r['episode_id'])].append(r)
    for ep_id in episodes:
        episodes[ep_id].sort(key=lambda s: s['step'])
    return dict(episodes)


# ══════════════════════════════════════════════════════════════
#  Build per-episode labeled sequences from episode_diagnostics
# ══════════════════════════════════════════════════════════════

def build_labeled_episodes(episodes_diag):
    """
    Returns:
        labeled_episodes: {ep_id: [(step_dict, gt_label), ...]}
            Only non-rotation steps (gt_label != -1) are included.
        episode_categories: {ep_id: str}
    """
    labeled = {}
    categories = {}
    for ep_id, steps in episodes_diag.items():
        seq = [(s, s['gt_label']) for s in steps if s.get('gt_label', -1) != -1]
        if not seq:
            continue
        labeled[ep_id] = seq
        # Use precomputed episode_category; fall back to inferring it
        cat = steps[-1].get('episode_category', None)
        if cat is None:
            labels = [l for _, l in seq]
            has_anom = any(l == 1 for l in labels)
            if not has_anom:
                cat = 'only_normal'
            else:
                fa = next(i for i, l in enumerate(labels) if l == 1)
                cat = 'only_anomaly' if fa <= 1 else 'normal_to_anomaly'
        categories[ep_id] = cat
    return labeled, categories


# ══════════════════════════════════════════════════════════════
#  Motion-based Baselines
# ══════════════════════════════════════════════════════════════

def build_stagnation_scores(labeled_episodes, window=5):
    """Score = 1 if max displacement over last W steps < 0.5 m."""
    result = {}
    for ep_id, seq in labeled_episodes.items():
        labels, scores = [], []
        pos_buf = []
        for step, label in seq:
            labels.append(label)
            pos = np.array([step.get('agent_x', 0.0),
                            step.get('agent_y', 0.0),
                            step.get('agent_z', 0.0)])
            pos_buf.append(pos)
            if len(pos_buf) < window + 1:
                scores.append(0.0)
            else:
                window_pos = pos_buf[-(window + 1):]
                max_disp = max(float(np.linalg.norm(p - window_pos[0])) for p in window_pos[1:])
                scores.append(1.0 if max_disp < 0.5 else 0.0)
        result[ep_id] = (labels, scores)
    return result


def build_act_failure_scores(labeled_episodes):
    """Score = 1 for MOVE_FORWARD with position delta < 0.01 m."""
    result = {}
    for ep_id, seq in labeled_episodes.items():
        labels, scores = [], []
        prev_pos = None
        for step, label in seq:
            labels.append(label)
            pos = np.array([step.get('agent_x', 0.0),
                            step.get('agent_y', 0.0),
                            step.get('agent_z', 0.0)])
            action = step.get('action_name', 'MOVE_FORWARD')
            if action == 'MOVE_FORWARD' and prev_pos is not None:
                dist = float(np.linalg.norm(pos - prev_pos))
                scores.append(1.0 if dist < 0.01 else 0.0)
            else:
                scores.append(0.0)
            prev_pos = pos
        result[ep_id] = (labels, scores)
    return result


# ══════════════════════════════════════════════════════════════
#  Action-Probability Baseline
# ══════════════════════════════════════════════════════════════

def build_action_prob_scores(labeled_episodes, threshold=0.5):
    """Score = 1 - action_prob  (high score = low confidence = possible anomaly).

    Only steps that have a valid action_prob are scored; steps with None get 0.0
    (treated as confident / not-alarm), which is conservative.

    Detection rule: score > (1 - threshold)  →  model was less than `threshold`
    confident in its chosen action.  Default threshold=0.5 means action_prob < 0.5
    triggers an alarm.
    """
    result = {}
    for ep_id, seq in labeled_episodes.items():
        labels, scores = [], []
        for step, label in seq:
            labels.append(label)
            prob = step.get('action_prob', None)
            if prob is not None:
                scores.append(1.0 - float(prob))   # low prob → high score
            else:
                scores.append(0.0)                  # unknown → not alarming
        result[ep_id] = (labels, scores)
    return result


# ══════════════════════════════════════════════════════════════
#  NaVILA Entropy-based Method (Ours)
# ══════════════════════════════════════════════════════════════

def build_navila_scores(labeled_episodes, anomaly_metrics, head_order, k_top, w):
    """RelDiff of mean entropy on top-K H_nav heads over window W."""
    # Build ep → step → metrics lookup from anomaly_metrics
    metrics_lookup = {}
    for ep_id, steps in anomaly_metrics.items():
        metrics_lookup[ep_id] = {s['step']: s for s in steps}

    result = {}
    for ep_id, seq in labeled_episodes.items():
        if ep_id not in metrics_lookup:
            continue
        ep_metrics = metrics_lookup[ep_id]
        labels, scores = [], []
        buf = []
        for step, label in seq:
            labels.append(label)
            step_idx = step['step']
            mph = ep_metrics.get(step_idx, {}).get('metrics_per_head', {})
            vals = np.full(len(head_order), np.nan, dtype=np.float64)
            for j, h in enumerate(head_order):
                if h in mph:
                    el = mph[h].get('entropy_list', [])
                    if el:
                        vals[j] = float(np.mean(el))
            valid = vals[:k_top][~np.isnan(vals[:k_top])]
            if len(valid) > 0:
                buf.append(float(np.mean(valid)))
            if len(buf) < w + 1:
                scores.append(None)
            else:
                scores.append(float(buf[-1]) / (float(np.mean(buf[-(w + 1):-1])) + _EPS))
        result[ep_id] = (labels, scores)
    return result


def infer_navila_threshold_dir(precomputed):
    """Infer threshold direction from normal vs anomaly score means."""
    nv, av = [], []
    for labels, scores in precomputed.values():
        for label, score in zip(labels, scores):
            if score is None:
                continue
            if label == 0:
                nv.append(score)
            else:
                av.append(score)
    if not nv or not av:
        return None, ">"
    nm, am = float(np.mean(nv)), float(np.mean(av))
    threshold = (nm + am) / 2.0
    direction = ">" if am > nm else "<"
    return threshold, direction


# ══════════════════════════════════════════════════════════════
#  Unified Evaluation
# ══════════════════════════════════════════════════════════════

def eval_detector(precomputed, episode_categories, threshold, patience, direction=">"):
    """
    Evaluate a detector on all episodes.

    precomputed: {ep_id: (labels, scores)}
        labels: list[int] — 0=normal, 1=anomaly
        scores: list[float|None] — detection score per step
    threshold: alarm fires when score crosses this value
    patience: number of consecutive threshold crossings before alarm fires
    direction: ">" means alarm when score > threshold
    """
    keys = ['all'] + CATEGORIES
    acc = {k: {'n_anom': 0, 'n_norm': 0, 'n_det': 0, 'n_fp': 0,
               'lats': [], 'TT': 0, 'TF': 0, 'FT': 0, 'FF': 0}
           for k in keys}

    for ep_id, (labels, scores) in precomputed.items():
        cat = episode_categories.get(ep_id, 'only_normal')
        buckets = [acc['all'], acc[cat]]

        onset = next((i for i, l in enumerate(labels) if l == 1), None)
        has_a = onset is not None
        for b in buckets:
            if has_a:
                b['n_anom'] += 1
            else:
                b['n_norm'] += 1

        n_steps = len(labels)
        consec = 0
        fired = False
        step_alarms = [False] * n_steps

        for i in range(n_steps):
            score = scores[i]
            if score is not None:
                cond = score > threshold if direction == ">" else score < threshold
                consec = consec + 1 if cond else 0
                if consec >= patience:
                    if not fired:
                        fired = True
                        if has_a and i >= onset:
                            for b in buckets:
                                b['n_det'] += 1
                                b['lats'].append(i - onset)
                        elif not has_a:
                            for b in buckets:
                                b['n_fp'] += 1
                    for j in range(max(0, i - patience + 1), i + 1):
                        step_alarms[j] = True

        # Propagate alarm to end of episode
        latched = False
        for i in range(n_steps):
            if step_alarms[i]:
                latched = True
            if latched:
                step_alarms[i] = True

        for lab, alm in zip(labels, step_alarms):
            if lab == 1 and alm:
                for b in buckets: b['TT'] += 1
            elif lab == 1 and not alm:
                for b in buckets: b['TF'] += 1
            elif lab == 0 and alm:
                for b in buckets: b['FT'] += 1
            else:
                for b in buckets: b['FF'] += 1

    results = {}
    for k in keys:
        a = acc[k]
        edr = a['n_det'] / a['n_anom'] if a['n_anom'] > 0 else 0.0
        fer = a['n_fp'] / a['n_norm'] if a['n_norm'] > 0 else 0.0
        tt, tf, ft, ff = a['TT'], a['TF'], a['FT'], a['FF']
        prec = tt / (tt + ft) if (tt + ft) > 0 else 0.0
        rec = tt / (tt + tf) if (tt + tf) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        lm = float(np.mean(a['lats'])) if a['lats'] else float('nan')
        results[k] = {
            'EDR': edr, 'FER': fer, 'Gap': edr - fer, 'LatMean': lm,
            'precision': prec, 'recall': rec, 'f1': f1,
            'n_anom': a['n_anom'], 'n_norm': a['n_norm'],
        }
    return results


# ══════════════════════════════════════════════════════════════
#  Output formatting
# ══════════════════════════════════════════════════════════════

def print_table(split_name, rows):
    """rows: list of (method_name, results_dict)"""
    print(f"\n  {'Method':<35s} {'EDR':>6s} {'FER':>6s} {'Gap':>6s} {'Lat':>6s}  "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s}  n_anom/n_norm")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  "
          f"{'-'*6} {'-'*6} {'-'*6}  {'-'*14}")
    for name, r in rows:
        lat = f"{r['LatMean']:.1f}" if not math.isnan(r['LatMean']) else '-'
        print(f"  {name:<35s} {r['EDR']*100:5.1f}% {r['FER']*100:5.1f}% "
              f"{r['Gap']*100:+5.1f}% {lat:>6s}  "
              f"{r['precision']*100:5.1f}% {r['recall']*100:5.1f}% {r['f1']*100:5.1f}%"
              f"  {r['n_anom']}/{r['n_norm']}")


def print_latex_block(results_by_split, method_names):
    print("\n" + "=" * 80)
    print("  LaTeX-ready summary  (normal_to_anomaly subset)")
    print("=" * 80)
    hdr = f"  {'Method':<25s}"
    for split in ['val_seen', 'val_unseen']:
        hdr += f" | {'EDR':>5s} {'FER':>5s} {'Gap':>5s} {'F1':>5s}"
    print(hdr)
    print(f"  {'-'*25}" + (" | " + "-" * 25) * 2)
    for method in method_names:
        row = f"  {method:<25s}"
        for split in ['val_seen', 'val_unseen']:
            r = results_by_split.get(split, {}).get(method)
            if r is None:
                row += " | {:>5s} {:>5s} {:>5s} {:>5s}".format('N/A', 'N/A', 'N/A', 'N/A')
            else:
                row += (f" | {r['EDR']*100:5.1f} {r['FER']*100:5.1f} "
                        f"{r['Gap']*100:+5.1f} {r['f1']*100:5.1f}")
        print(row)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def run_split(split_name, gt_file, metrics_dir, head_order, k_top, w, tau,
              stag_window, stag_patience, act_patience, navila_patience,
              navila_direction, action_prob_threshold=0.5, action_prob_patience=1):
    """Run all methods for a single split. Returns {method: result_dict}."""
    print(f"\n{'='*60}")
    print(f"  Split: {split_name}")
    if not Path(gt_file).exists():
        print(f"  [SKIP] GT file not found: {gt_file}")
        return {}
    print(f"  GT: {gt_file}")
    print(f"{'='*60}")

    episodes_diag = load_episode_diagnostics(gt_file)
    labeled_eps, ep_cats = build_labeled_episodes(episodes_diag)

    from collections import Counter
    cat_counts = Counter(ep_cats.values())
    print(f"  Episodes: {len(labeled_eps)}  "
          f"(N→A: {cat_counts['normal_to_anomaly']}, "
          f"OA: {cat_counts['only_anomaly']}, "
          f"ON: {cat_counts['only_normal']})")

    rows = []
    split_results = {}

    # ── Stagnation ──
    stag_scores = build_stagnation_scores(labeled_eps, window=stag_window)
    r_stag = eval_detector(stag_scores, ep_cats, threshold=0.5,
                           patience=stag_patience, direction=">")
    key = f"Stagnation (W={stag_window},P={stag_patience})"
    rows.append((key, r_stag['all']))
    split_results['Stagnation'] = r_stag['normal_to_anomaly']

    # ── Act.Failure ──
    act_scores = build_act_failure_scores(labeled_eps)
    r_act = eval_detector(act_scores, ep_cats, threshold=0.5,
                          patience=act_patience, direction=">")
    key = f"Act.Failure (P={act_patience})"
    rows.append((key, r_act['all']))
    split_results['Act.Failure'] = r_act['normal_to_anomaly']

    # ── Uncertainty (action_prob) ──
    # Score = 1 - action_prob; alarm when score > 0.25 (action_prob < 0.75)
    r_ap = None
    ap_scores = build_action_prob_scores(labeled_eps, threshold=action_prob_threshold)
    n_with_prob = sum(
        1 for seq in labeled_eps.values()
        for step, _ in seq if step.get('action_prob') is not None
    )
    if n_with_prob > 0:
        r_ap = eval_detector(ap_scores, ep_cats,
                             threshold=0.50,
                             patience=action_prob_patience, direction=">")
        key = f"Uncertainty (action_prob,P={action_prob_patience})"
        rows.append((key, r_ap['all']))
        split_results['Uncertainty'] = r_ap['normal_to_anomaly']
        print(f"  [Uncertainty] {n_with_prob} steps with action_prob")
    else:
        print(f"  [Uncertainty] Skipped (no action_prob in GT file — run step 2b→3b first)")

    # ── NaVILA (Ours) ──
    if metrics_dir and Path(metrics_dir).exists() and head_order:
        try:
            anomaly_metrics = load_anomaly_metrics(metrics_dir)
            nav_scores = build_navila_scores(labeled_eps, anomaly_metrics,
                                             head_order, k_top, w)
            # Determine threshold if not given
            if tau is None or navila_direction is None:
                inferred_tau, inferred_dir = infer_navila_threshold_dir(nav_scores)
                use_tau = tau if tau is not None else inferred_tau
                use_dir = navila_direction if navila_direction is not None else inferred_dir
                print(f"  [Ours] Inferred tau={use_tau:.4f} dir={use_dir}")
            else:
                use_tau, use_dir = tau, navila_direction
            if use_tau is not None:
                r_nav = eval_detector(nav_scores, ep_cats, threshold=use_tau,
                                      patience=navila_patience, direction=use_dir)
                key = f"Ours (K={k_top},W={w},P={navila_patience},tau={use_tau:.3f})"
                rows.append((key, r_nav['all']))
                split_results['Ours'] = r_nav['normal_to_anomaly']
        except Exception as e:
            print(f"  [Ours] Skipped: {e}")
    else:
        print(f"  [Ours] Skipped (no metrics_dir or head_order)")

    print_table(split_name, rows)

    # ── Normal→Anomaly breakdown ──
    print(f"\n  Normal→Anomaly breakdown:")
    na_rows = []
    for name, r_all in rows:
        if name.startswith('Stagnation'):
            na_rows.append((name, r_stag['normal_to_anomaly']))
        elif name.startswith('Act.Failure'):
            na_rows.append((name, r_act['normal_to_anomaly']))
        elif name.startswith('Uncertainty') and 'Uncertainty' in split_results:
            na_rows.append((name, split_results['Uncertainty']))
        elif name.startswith('Ours') and 'Ours' in split_results:
            na_rows.append((name, split_results['Ours']))
    if na_rows:
        print_table(split_name + " [N→A]", na_rows)

    return split_results, rows


def main():
    parser = argparse.ArgumentParser(description="Comparison Study from pre-computed GT labels")

    # GT inputs — single split or dual-split mode
    parser.add_argument('--gt_file',
                        default=None,
                        help='episode_diagnostics.jsonl (single-split mode)')
    parser.add_argument('--split',
                        default='train',
                        help='Split name for --gt_file mode')
    parser.add_argument('--gt_seen',
                        default=None,
                        help='episode_diagnostics.jsonl for val_seen')
    parser.add_argument('--gt_unseen',
                        default=None,
                        help='episode_diagnostics.jsonl for val_unseen')

    # Anomaly metrics (for NaVILA entropy method)
    parser.add_argument('--metrics_dir',
                        default=None,
                        help='Dir with anomaly_metrics_chunk*.jsonl (single-split)')
    parser.add_argument('--metrics_seen', default=None,
                        help='anomaly_metrics dir for val_seen')
    parser.add_argument('--metrics_unseen', default=None,
                        help='anomaly_metrics dir for val_unseen')

    # NaVILA config (from Steps 4+5)
    parser.add_argument('--head_config',
                        default='eval_out/hivla/4_hnav/outputs/hnav_config.json',
                        help='hnav_config.json from Step 4')
    parser.add_argument('--best_config',
                        default='eval_out/hivla/5_grid/outputs/best_config.json',
                        help='best_config.json from Step 5')
    parser.add_argument('--best_k', type=int, default=None)
    parser.add_argument('--best_w', type=int, default=None)
    parser.add_argument('--best_p', type=int, default=None)
    parser.add_argument('--best_tau', type=float, default=None)
    parser.add_argument('--navila_dir', default=None,
                        help='Direction for NaVILA score (> or <). Auto-inferred if omitted.')

    # Baseline hyperparameters
    parser.add_argument('--stag_window', type=int, default=5)
    parser.add_argument('--stag_patience', type=int, default=1)
    parser.add_argument('--act_patience', type=int, default=1)

    # ActionProb hyperparameters
    parser.add_argument('--ap_threshold', type=float, default=0.5,
                        help='Action-prob threshold: alarm if action_prob < this value (default 0.5)')
    parser.add_argument('--ap_patience', type=int, default=1,
                        help='ActionProb patience (consecutive low-prob steps to trigger)')

    # Output
    parser.add_argument('--output_dir',
                        default='eval_out/hivla/7_comparison/outputs',
                        help='Output directory')
    args = parser.parse_args()

    # ── Load NaVILA config ──
    head_order, k_top, w, tau, navila_patience = [], None, None, None, 1
    try:
        with open(args.head_config) as f:
            hnav_cfg = json.load(f)
        head_order = hnav_cfg['head_order']
        with open(args.best_config) as f:
            best_cfg = json.load(f)
        k_top = args.best_k or best_cfg.get('K')
        w = args.best_w or best_cfg.get('W')
        tau = args.best_tau or best_cfg.get('tau')
        navila_patience = args.best_p or best_cfg.get('P', 1)
        print(f"NaVILA config: K={k_top}, W={w}, P={navila_patience}, tau={tau}")
        print(f"Heads (top-{k_top}): {head_order[:k_top]}")
    except Exception as e:
        print(f"[warn] Could not load NaVILA config: {e}")

    navila_dir = args.navila_dir  # may be None → auto-infer

    # ── Build split list ──
    splits = []
    if args.gt_seen or args.gt_unseen:
        if args.gt_seen:
            splits.append(('val_seen', args.gt_seen, args.metrics_seen))
        if args.gt_unseen:
            splits.append(('val_unseen', args.gt_unseen, args.metrics_unseen))
    elif args.gt_file:
        splits.append((args.split, args.gt_file, args.metrics_dir))
    else:
        # Default: use the standard step-3 output
        splits.append((
            'train',
            'eval_out/hivla/3_gt/val_seen/episode_diagnostics.jsonl',
            'eval_out/hivla/2_data/val_seen',
        ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_csv_rows = []
    results_by_split = {}
    all_method_names = ['Stagnation', 'Act.Failure', 'Uncertainty', 'Ours']

    for split_name, gt_file, metrics_dir in splits:
        ret = run_split(
            split_name=split_name,
            gt_file=gt_file,
            metrics_dir=metrics_dir,
            head_order=head_order,
            k_top=k_top,
            w=w,
            tau=tau,
            stag_window=args.stag_window,
            stag_patience=args.stag_patience,
            act_patience=args.act_patience,
            navila_patience=navila_patience,
            navila_direction=navila_dir,
            action_prob_threshold=args.ap_threshold,
            action_prob_patience=args.ap_patience,
        )
        if not ret:
            continue
        split_results, rows = ret
        results_by_split[split_name] = split_results
        for method, r_na in split_results.items():
            all_csv_rows.append({
                'split': split_name, 'method': method,
                'EDR': r_na['EDR'], 'FER': r_na['FER'], 'Gap': r_na['Gap'],
                'Lat': r_na['LatMean'] if not math.isnan(r_na['LatMean']) else None,
                'na_Prec': r_na['precision'], 'na_Rec': r_na['recall'], 'na_F1': r_na['f1'],
                'n_anom': r_na['n_anom'], 'n_norm': r_na['n_norm'],
            })

    # ── Save CSV ──
    out_csv = output_dir / 'comparison_from_gt.csv'
    if all_csv_rows:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\nSaved → {out_csv}")

    # ── LaTeX summary (if both splits) ──
    if len(results_by_split) >= 2:
        print_latex_block(results_by_split, all_method_names)


if __name__ == '__main__':
    main()
