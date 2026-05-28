#!/usr/bin/env python3
"""
Comparison Study: Stagnation vs Act.Failure vs NaVid (Ours)
════════════════════════════════════════════════════════════

Reads pre-computed GT labels from episode_diagnostics.jsonl (Step 3 output)
and compares three path-deviation detectors:

    Stagnation  : no position change over W consecutive steps
    Act.Failure : MOVE_FORWARD with position delta < 0.01 m
    Ours (NaVid): attention-entropy RelDiff on H_nav heads
                  (requires anomaly_metrics_chunk*.jsonl from Step 2)

Usage:
    # Single split (val_seen):
    python hivla/7_comparison/compare_from_gt.py \\
        --gt_file   eval_out/hivla/3_gt/val_seen/episode_diagnostics.jsonl \\
        --metrics_dir eval_out/hivla/2_data/val_seen \\
        --head_config eval_out/hivla/4_hnav/outputs/hnav_config.json \\
        --best_config eval_out/hivla/5_grid/outputs/best_config.json \\
        --split val_seen

    # Both splits (val_seen + val_unseen):
    python hivla/7_comparison/compare_from_gt.py \\
        --gt_seen   eval_out/hivla/3_gt/val_seen/episode_diagnostics.jsonl \\
        --gt_unseen eval_out/hivla/3_gt/val_unseen/episode_diagnostics.jsonl \\
        --metrics_seen   eval_out/hivla/2_data/val_seen \\
        --metrics_unseen eval_out/hivla/2_data/val_unseen \\
        --head_config eval_out/hivla/4_hnav/outputs/hnav_config.json \\
        --best_config eval_out/hivla/5_grid/outputs/best_config.json
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
#  Build labeled episodes from episode_diagnostics
# ══════════════════════════════════════════════════════════════

def build_labeled_episodes(episodes_diag):
    """
    Returns:
        labeled_episodes: {ep_id: [(step_dict, gt_label), ...]}  (rotation steps excluded)
        episode_categories: {ep_id: str}
    """
    labeled = {}
    categories = {}
    for ep_id, steps in episodes_diag.items():
        seq = [(s, s['gt_label']) for s in steps if s.get('gt_label', -1) != -1]
        if not seq:
            continue
        labeled[ep_id] = seq
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
    """Score = 1 when previous action was MOVE_FORWARD but position delta < 0.01 m.

    Position is recorded BEFORE action execution, so delta(pos[N] - pos[N-1])
    reflects the result of action[N-1], not action[N].
    """
    result = {}
    for ep_id, seq in labeled_episodes.items():
        labels, scores = [], []
        prev_pos = None
        prev_action = None
        for step, label in seq:
            labels.append(label)
            pos = np.array([step.get('agent_x', 0.0),
                            step.get('agent_y', 0.0),
                            step.get('agent_z', 0.0)])
            if prev_action == 'MOVE_FORWARD' and prev_pos is not None:
                dist = float(np.linalg.norm(pos - prev_pos))
                scores.append(1.0 if dist < 0.01 else 0.0)
            else:
                scores.append(0.0)
            prev_pos = pos
            prev_action = step.get('action_name', '')
        result[ep_id] = (labels, scores)
    return result


# ══════════════════════════════════════════════════════════════
#  Uncertainty (action_prob) Baseline
# ══════════════════════════════════════════════════════════════

def build_uncertainty_scores(labeled_episodes):
    """Score = 1 - action_prob, read directly from episode_diagnostics step dict.

    episode_diagnostics already embeds action_prob (written by Step 3).
    Steps without action_prob get score 0 (treated as confident / no alarm).
    """
    result = {}
    for ep_id, seq in labeled_episodes.items():
        labels, scores = [], []
        for step, label in seq:
            labels.append(label)
            ap = step.get('action_prob', None)
            scores.append(1.0 - float(ap) if ap is not None else 0.0)
        result[ep_id] = (labels, scores)
    return result


# ══════════════════════════════════════════════════════════════
#  NaVid Attention-Entropy Method (Ours)
# ══════════════════════════════════════════════════════════════

def build_navid_scores(labeled_episodes, anomaly_metrics, head_order, k_top, w):
    """RelDiff of mean entropy on top-K H_nav heads over window W."""
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


def infer_threshold_dir(precomputed):
    """Infer threshold and direction from normal vs anomaly score means."""
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
#  Evaluation
# ══════════════════════════════════════════════════════════════

def eval_detector(precomputed, episode_categories, threshold, patience, direction=">"):
    """
    Evaluate a detector on all episodes.

    precomputed: {ep_id: (labels, scores)}
    threshold:   alarm fires when score crosses this value
    patience:    consecutive crossings before alarm fires
    direction:   ">" or "<"
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

        # Latch alarm through end of episode
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
#  Output Formatting
# ══════════════════════════════════════════════════════════════

def print_table(split_name, rows):
    print(f"\n  {'Method':<40s} {'EDR':>6s} {'FER':>6s} {'Gap':>6s} {'Lat':>6s}  "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s}  n_anom/n_norm")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*6} {'-'*6} {'-'*6}  {'-'*14}")
    for name, r in rows:
        lat = f"{r['LatMean']:.1f}" if not math.isnan(r['LatMean']) else '-'
        print(f"  {name:<40s} {r['EDR']*100:5.1f}% {r['FER']*100:5.1f}% "
              f"{r['Gap']*100:+5.1f}% {lat:>6s}  "
              f"{r['precision']*100:5.1f}% {r['recall']*100:5.1f}% {r['f1']*100:5.1f}%"
              f"  {r['n_anom']}/{r['n_norm']}")


def print_latex_block(results_by_split, method_names):
    print("\n" + "=" * 80)
    print("  LaTeX-ready summary  (normal_to_anomaly subset)")
    print("=" * 80)
    hdr = f"  {'Method':<28s}"
    for split in ['val_seen', 'val_unseen']:
        hdr += f" | {'EDR':>5s} {'FER':>5s} {'Gap':>6s} {'F1':>5s}"
    print(hdr)
    print(f"  {'-'*28}" + (" | " + "-" * 26) * 2)
    for method in method_names:
        row = f"  {method:<28s}"
        for split in ['val_seen', 'val_unseen']:
            r = results_by_split.get(split, {}).get(method)
            if r is None:
                row += " | {:>5s} {:>5s} {:>6s} {:>5s}".format('N/A', 'N/A', 'N/A', 'N/A')
            else:
                row += (f" | {r['EDR']*100:5.1f} {r['FER']*100:5.1f} "
                        f"{r['Gap']*100:+6.1f} {r['f1']*100:5.1f}")
        print(row)


# ══════════════════════════════════════════════════════════════
#  Per-split runner
# ══════════════════════════════════════════════════════════════

def run_split(split_name, gt_file, metrics_dir, head_order, k_top, w, tau,
              stag_window, stag_patience, act_patience, navid_patience,
              navid_direction, output_dir):
    print(f"\n{'='*66}")
    print(f"  Split: {split_name}")
    if not Path(gt_file).exists():
        print(f"  [SKIP] GT file not found: {gt_file}")
        return {}
    print(f"  GT: {gt_file}")
    print(f"{'='*66}")

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
    split_results['Stagnation'] = r_stag.get('normal_to_anomaly', r_stag['all'])

    # ── Act.Failure ──
    act_scores = build_act_failure_scores(labeled_eps)
    r_act = eval_detector(act_scores, ep_cats, threshold=0.5,
                          patience=act_patience, direction=">")
    key = f"Act.Failure (P={act_patience})"
    rows.append((key, r_act['all']))
    split_results['Act.Failure'] = r_act.get('normal_to_anomaly', r_act['all'])

    # ── Uncertainty (action_prob) ──
    # Score = 1 - action_prob; alarm when score > 0.25 (action_prob < 0.75).
    # action_prob is embedded in episode_diagnostics by Step 3 — read directly.
    r_unc = None
    n_with_prob = sum(
        1 for seq in labeled_eps.values()
        for step, _ in seq if step.get('action_prob') is not None
    )
    if n_with_prob > 0:
        unc_scores = build_uncertainty_scores(labeled_eps)
        r_unc = eval_detector(unc_scores, ep_cats, threshold=0.5,
                              patience=1, direction=">")
        rows.append(("Uncertainty (action_prob)", r_unc['all']))
        split_results['Uncertainty'] = r_unc.get('normal_to_anomaly', r_unc['all'])
        print(f"  [Uncertainty] {n_with_prob} steps with action_prob")
    else:
        print("  [Uncertainty] Skipped (action_prob=null in GT; re-run Step 2 → Step 3)")

    # ── NaVid (Ours) ──
    if metrics_dir and Path(metrics_dir).exists() and head_order:
        try:
            anomaly_metrics = load_anomaly_metrics(metrics_dir)
            nav_scores = build_navid_scores(labeled_eps, anomaly_metrics,
                                            head_order, k_top, w)
            if tau is None or navid_direction is None:
                inferred_tau, inferred_dir = infer_threshold_dir(nav_scores)
                use_tau = tau if tau is not None else inferred_tau
                use_dir = navid_direction if navid_direction is not None else inferred_dir
                print(f"  [Ours] Inferred tau={use_tau:.4f} dir={use_dir}")
            else:
                use_tau, use_dir = tau, navid_direction

            if use_tau is not None:
                r_nav = eval_detector(nav_scores, ep_cats, threshold=use_tau,
                                      patience=navid_patience, direction=use_dir)
                key = f"Ours (K={k_top},W={w},P={navid_patience},tau={use_tau:.3f})"
                rows.append((key, r_nav['all']))
                split_results['Ours'] = r_nav.get('normal_to_anomaly', r_nav['all'])
        except Exception as e:
            print(f"  [Ours] Skipped: {e}")
    else:
        print(f"  [Ours] Skipped (no metrics_dir or head_order)")

    print_table(split_name, rows)

    # Per-category breakdown
    print(f"\n  Normal→Anomaly breakdown:")
    na_rows = []
    for name, r_all in rows:
        if name.startswith('Stagnation'):
            na_rows.append((name, r_stag.get('normal_to_anomaly', r_stag['all'])))
        elif name.startswith('Act.Failure'):
            na_rows.append((name, r_act.get('normal_to_anomaly', r_act['all'])))
        elif name.startswith('Uncertainty') and 'Uncertainty' in split_results:
            na_rows.append((name, split_results['Uncertainty']))
        elif name.startswith('Ours') and 'Ours' in split_results:
            na_rows.append((name, split_results['Ours']))
    if na_rows:
        print_table(split_name + " [N→A]", na_rows)

    # Save per-split CSV
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    extra_method_dicts = [('Uncertainty', r_unc)] if r_unc is not None else []
    for cat in ['all', 'only_normal', 'only_anomaly', 'normal_to_anomaly']:
        for mname, r_all_dict in [
            ('Stagnation', r_stag),
            ('Act.Failure', r_act),
        ] + extra_method_dicts:
            r = r_all_dict.get(cat)
            if r:
                csv_rows.append({
                    'split': split_name, 'method': mname, 'category': cat,
                    **{k: round(v, 4) if not (isinstance(v, float) and math.isnan(v)) else None
                       for k, v in r.items()},
                })
        if 'Ours' in split_results and metrics_dir:
            for cat2 in ['all']:
                try:
                    if nav_scores:
                        r = eval_detector(nav_scores, ep_cats, threshold=use_tau,
                                         patience=navid_patience, direction=use_dir).get(cat2)
                        if r:
                            csv_rows.append({
                                'split': split_name, 'method': 'Ours', 'category': cat2,
                                **{k: round(v, 4) if not (isinstance(v, float) and math.isnan(v)) else None
                                   for k, v in r.items()},
                            })
                except Exception:
                    pass

    if csv_rows:
        csv_path = out_dir / f"comparison_{split_name}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Saved: {csv_path}")

    return split_results, rows


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NaVid Comparison: Stagnation vs Act.Failure vs Ours")

    parser.add_argument('--gt_file', default=None,
                        help='episode_diagnostics.jsonl (single-split mode)')
    parser.add_argument('--split', default='val_seen',
                        help='Split name for --gt_file mode')
    parser.add_argument('--gt_seen', default=None,
                        help='episode_diagnostics.jsonl for val_seen')
    parser.add_argument('--gt_unseen', default=None,
                        help='episode_diagnostics.jsonl for val_unseen')

    parser.add_argument('--metrics_dir', default=None,
                        help='Dir with anomaly_metrics_chunk*.jsonl (single-split)')
    parser.add_argument('--metrics_seen', default=None,
                        help='anomaly_metrics dir for val_seen')
    parser.add_argument('--metrics_unseen', default=None,
                        help='anomaly_metrics dir for val_unseen')

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
    parser.add_argument('--navid_dir', default=None,
                        help='Score direction for Ours (> or <). Auto-inferred if omitted.')

    parser.add_argument('--stag_window', type=int, default=5)
    parser.add_argument('--stag_patience', type=int, default=1)
    parser.add_argument('--act_patience', type=int, default=1)

    parser.add_argument('--output_dir', default='eval_out/hivla/7_comparison/outputs',
                        help='Output directory')
    args = parser.parse_args()

    # Load config from Steps 4+5
    head_order, k_top, w, tau, navid_patience = [], None, None, None, 1
    try:
        with open(args.head_config) as f:
            hnav_cfg = json.load(f)
        head_order = hnav_cfg['head_order']
        with open(args.best_config) as f:
            best_cfg = json.load(f)
        k_top = args.best_k or best_cfg.get('K')
        w = args.best_w or best_cfg.get('W')
        tau = args.best_tau or best_cfg.get('tau')
        navid_patience = args.best_p or best_cfg.get('P', 1)
        print(f"NaVid config: K={k_top}, W={w}, P={navid_patience}, tau={tau}")
        print(f"Heads (top-{k_top}): {head_order[:k_top]}")
    except Exception as e:
        print(f"[warn] Could not load NaVid config: {e}")

    navid_dir = args.navid_dir

    # Build split list
    splits = []
    if args.gt_seen or args.gt_unseen:
        if args.gt_seen:
            splits.append(('val_seen', args.gt_seen, args.metrics_seen))
        if args.gt_unseen:
            splits.append(('val_unseen', args.gt_unseen, args.metrics_unseen))
    elif args.gt_file:
        splits.append((args.split, args.gt_file, args.metrics_dir))
    else:
        splits.append((
            'val_seen',
            'eval_out/hivla/3_gt/val_seen/episode_diagnostics.jsonl',
            'eval_out/hivla/2_data/val_seen',
        ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_split = {}
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
            navid_patience=navid_patience,
            navid_direction=navid_dir,
            output_dir=args.output_dir,
        )
        if ret:
            split_results, _ = ret
            results_by_split[split_name] = split_results

    if len(results_by_split) >= 2:
        print_latex_block(results_by_split, ['Stagnation', 'Act.Failure', 'Uncertainty', 'Ours'])


if __name__ == '__main__':
    main()
