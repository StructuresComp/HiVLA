#!/usr/bin/env python3
"""
Step 6: Generate publication-quality figures from Step 5 grid search outputs.

Produces:
  - fig1a_head_discriminability.png  (Cohen's d bar chart, top-20)
  - fig_K_step_na.png                (Step-level K ablation for N→A, best config per K)
  - fig_K_episode.png                (Episode-level K ablation, best config per K)
  - fig2_ablation_K3.png             (2x2: K, W, P+Latency, τ ablation, fixed-param)

Usage:
    python generate_figures.py
    python generate_figures.py --data_dir ... --head_csv ... --output_dir ...
"""

import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── Style ──
matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 200,
})

C_EDR  = '#2980B9'   # blue
C_FER  = '#E74C3C'   # red
C_GAP  = '#27AE60'   # green
C_LAT  = '#BFBFBF'   # light gray for latency bars
C_BEST = '#E67E22'   # orange for best/selected markers
C_REST = '#AED6F1'   # light blue for remaining heads


def _clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _safe_float(val, default=np.nan):
    if val is None or val in ('', 'nan', 'None'):
        return default
    return float(val)


def _get_gap(row):
    """Get Gap from row, computing EDR-FER if Gap column is missing."""
    if 'Gap' in row and row['Gap'] not in ('', 'nan', 'None', None):
        return float(row['Gap'])
    return float(row['EDR']) - float(row['FER'])


def _derive_recall(prec, f1):
    """Derive recall from precision and F1: R = F1*P / (2P - F1)."""
    denom = 2 * prec - f1
    if denom <= 0 or prec <= 0 or f1 <= 0:
        return 0.0
    return f1 * prec / denom


def _add_legend(ax, has_gap=True):
    lines = [
        plt.Line2D([0], [0], color=C_EDR, marker='o', lw=1.8, ms=4, label='EDR'),
        plt.Line2D([0], [0], color=C_FER, marker='o', lw=1.8, ms=4, label='FER'),
    ]
    if has_gap:
        lines.append(plt.Line2D([0], [0], color=C_GAP, marker='o', lw=1.4, ms=4,
                                linestyle='--', label='Gap'))
    ax.legend(handles=lines, loc='best', fontsize=7.5)


# ══════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════

def load_data(data_dir, head_csv):
    data_dir = Path(data_dir)

    head_imp   = _load_csv(head_csv)
    abl_K_best = _load_csv(data_dir / "ablation_K_bestgap.csv")
    abl_W      = _load_csv(data_dir / "ablation_W.csv")
    abl_P      = _load_csv(data_dir / "ablation_P.csv")
    thresh     = _load_csv(data_dir / "threshold_sweep.csv")

    return head_imp, abl_K_best, abl_W, abl_P, thresh


# ══════════════════════════════════════════════════════════════════════════
#  Figure 1a: Head Discriminability (Cohen's d bar chart)
# ══════════════════════════════════════════════════════════════════════════

def fig1a_head_discriminability(head_imp, selected_k, output_dir):
    print("  Generating fig1a_head_discriminability.png ...")

    ranks  = np.array([int(r['rank']) for r in head_imp])
    labels = [f"L{r['layer']}H{r['head']}" for r in head_imp]
    d_vals = np.array([float(r['cohens_d']) for r in head_imp])

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=200)

    colors = [C_BEST if rank <= selected_k else C_REST for rank in ranks]
    ax.bar(ranks, d_vals, color=colors, edgecolor='white', linewidth=0.3, width=0.7)

    ax.set_xlabel("Head Rank", fontsize=10)
    ax.set_ylabel("Cohen's $d$", fontsize=12)
    ax.set_title("Attention Head Discriminability (Cohen's $d$, top-20)",
                 fontsize=11, fontweight='bold')
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"{r}\n{labels[i]}" for i, r in enumerate(ranks)],
                        fontsize=6, rotation=45, ha='right')
    ax.set_xlim(0.2, 20.8)
    ax.set_ylim(0, d_vals.max() * 1.15)
    _clean_ax(ax)

    legend_elements = [
        Patch(facecolor=C_BEST, label=f'Top-{selected_k} (selected)'),
        Patch(facecolor=C_REST, label=f'Rank {selected_k+1}-20'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    out = Path(output_dir) / "fig1a_head_discriminability.png"
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure: K Step-level N→A (best config per K from grid search)
# ══════════════════════════════════════════════════════════════════════════

def fig_K_step_na(abl_K_best, selected_k, output_dir):
    print("  Generating fig_K_step_na.png ...")

    K_vals   = np.array([int(float(r['K'])) for r in abl_K_best])
    na_prec  = np.array([_safe_float(r.get('na_Prec', 0)) for r in abl_K_best])
    na_f1    = np.array([_safe_float(r.get('na_F1', 0)) for r in abl_K_best])
    na_recall = np.array([_derive_recall(p, f) for p, f in zip(na_prec, na_f1)])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    ax.plot(K_vals, na_prec * 100, '^-', color=C_EDR, lw=2, ms=7,
            label='Precision', markerfacecolor=C_EDR)
    ax.plot(K_vals, na_f1 * 100, 'D-', color=C_BEST, lw=2, ms=7,
            label='F1', markerfacecolor=C_BEST)
    ax.plot(K_vals, na_recall * 100, 'v-', color=C_FER, lw=2, ms=7,
            label='Recall', markerfacecolor=C_FER)

    # Highlight selected K
    ax.axvline(x=selected_k, color=C_BEST, linestyle='--', lw=1.5, alpha=0.6)

    # Y-axis: zoom in to show differences
    all_vals = np.concatenate([na_prec * 100, na_f1 * 100, na_recall * 100])
    y_min = max(0, np.min(all_vals) - 5)
    y_max = min(100, np.max(all_vals) + 3)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel('Ensemble Size $K$', fontsize=12)
    ax.set_ylabel('Step-level Metric (%)', fontsize=12)
    ax.set_title(r'Step-level $K$ Ablation (best $\tau$ per $K$, FER 5-10%, N$\rightarrow$A)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(K_vals)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    _clean_ax(ax)

    plt.tight_layout()
    out = Path(output_dir) / "fig_K_step_na.png"
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure: K Episode-level (best config per K from grid search)
# ══════════════════════════════════════════════════════════════════════════

def _plot_K_episode(ax, abl_K_best, selected_k, ab_w, ab_p,
                    title=None, xtick_step=None,
                    label_fontsize=10, title_fontsize=10):
    """Plot K episode-level ablation on a single axis."""
    K_vals = np.array([int(float(r['K'])) for r in abl_K_best])
    EDR    = np.array([float(r['EDR']) for r in abl_K_best])
    FER    = np.array([float(r['FER']) for r in abl_K_best])
    Gap    = np.array([_get_gap(r) for r in abl_K_best])

    ax.plot(K_vals, EDR * 100, 'o-', color=C_EDR, lw=2, ms=5, label='EDR')
    ax.plot(K_vals, FER * 100, 's-', color=C_FER, lw=2, ms=5, label='FER')
    ax.fill_between(K_vals, FER * 100, EDR * 100, alpha=0.08, color=C_GAP)
    ax.plot(K_vals, Gap * 100, 'D--', color=C_GAP, lw=1.5, ms=5, label='Gap')
    ax.axvline(x=selected_k, color=C_BEST, linestyle='--', lw=1.5, alpha=0.8)

    # Label at selected K
    gap_at_k = Gap[K_vals == selected_k]
    if len(gap_at_k) > 0:
        ax.text(selected_k + 0.3, gap_at_k[0] * 100 + 0.8,
                f'$K$={selected_k}', color=C_BEST, fontsize=8, fontweight='bold')

    if xtick_step:
        ax.set_xticks(range(1, max(K_vals) + 1, xtick_step))
    else:
        ax.set_xticks(K_vals)
    ax.set_xlim(0.5, max(K_vals) + 0.5)
    ax.grid(True, alpha=0.3)
    _clean_ax(ax)

    ax.set_ylabel('Rate (%)', fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

    return ax


def fig_K_episode(abl_K_best, selected_k, ab_w, ab_p, output_dir):
    print("  Generating fig_K_episode.png ...")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    _plot_K_episode(
        ax, abl_K_best, selected_k, ab_w, ab_p,
        title=f'Episode-level $K$ Ablation ($W$={ab_w}, $P$={ab_p}, '
              f'best $\\tau$ per $K$, FER 5-10%)',
        label_fontsize=12, title_fontsize=13)
    ax.set_xlabel('Ensemble Size $K$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    out = Path(output_dir) / "fig_K_episode.png"
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure 2: 2x2 Ablation (K, W, P+Latency, τ)
# ══════════════════════════════════════════════════════════════════════════

def fig2_ablation(abl_K_best, abl_W, abl_P, thresh, selected_k, ab_w, ab_p, ab_tau, output_dir):
    print("  Generating fig2_ablation_K3.png ...")

    # Parse W ablation
    W_vals = np.array([int(float(r['W'])) for r in abl_W])
    W_EDR  = np.array([float(r['EDR']) for r in abl_W])
    W_FER  = np.array([float(r['FER']) for r in abl_W])
    W_Gap  = np.array([_get_gap(r) for r in abl_W])

    # Parse P ablation
    P_vals = np.array([int(float(r['P'])) for r in abl_P])
    P_EDR  = np.array([float(r['EDR']) for r in abl_P])
    P_FER  = np.array([float(r['FER']) for r in abl_P])
    P_Gap  = np.array([_get_gap(r) for r in abl_P])
    P_Lat  = np.array([_safe_float(r.get('LatMean')) for r in abl_P])

    # Parse threshold sweep
    T_thresh = np.array([float(r['threshold']) for r in thresh])
    T_EDR    = np.array([float(r['EDR']) for r in thresh])
    T_FER    = np.array([float(r['FER']) for r in thresh])
    T_Gap    = np.array([_get_gap(r) for r in thresh])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=200)

    # -- (a) K Ablation --
    _plot_K_episode(
        axes[0, 0], abl_K_best, selected_k, ab_w, ab_p,
        title=f'(a) $K$ Ablation ($W$={ab_w}, $P$={ab_p}, '
              f'best $\\tau$ per $K$)',
        xtick_step=5)
    axes[0, 0].set_xlabel('Ensemble Size $K$', fontsize=10)
    _add_legend(axes[0, 0])

    # -- (b) W Ablation --
    ax = axes[0, 1]
    ax.plot(W_vals, W_EDR * 100, 'o-', color=C_EDR, lw=1.8, ms=5)
    ax.plot(W_vals, W_FER * 100, 'o-', color=C_FER, lw=1.8, ms=5)
    ax.fill_between(W_vals, W_FER * 100, W_EDR * 100, alpha=0.08, color=C_GAP)
    ax.plot(W_vals, W_Gap * 100, 'o--', color=C_GAP, lw=1.4, ms=5)
    ax.axvline(x=ab_w, color=C_BEST, linestyle='--', lw=1.5, alpha=0.8)
    ax.text(ab_w - 1.8, max(W_EDR * 100) * 0.92,
            f'$W$={ab_w}', color=C_BEST, fontsize=8, fontweight='bold')
    ax.set_xlabel('Window Size $W$', fontsize=10)
    ax.set_ylabel('Rate (%)', fontsize=10)
    ax.set_title(f'(b) $W$ Ablation ($K$={selected_k}, $P$={ab_p}, '
                 f'$\\tau$={ab_tau})', fontsize=10, fontweight='bold')
    ax.set_xticks(W_vals)
    _clean_ax(ax)
    _add_legend(ax)

    # -- (c) P Ablation + Latency --
    ax = axes[1, 0]
    ax.plot(P_vals, P_EDR * 100, 'o-', color=C_EDR, lw=1.8, ms=5, zorder=3)
    ax.plot(P_vals, P_FER * 100, 'o-', color=C_FER, lw=1.8, ms=5, zorder=3)
    ax.fill_between(P_vals, P_FER * 100, P_EDR * 100, alpha=0.08, color=C_GAP, zorder=1)
    ax.plot(P_vals, P_Gap * 100, 'o--', color=C_GAP, lw=1.4, ms=5, zorder=3)
    ax.axvline(x=ab_p, color=C_BEST, linestyle='--', lw=1.5, alpha=0.8)
    ax.text(ab_p + 0.2, max(P_EDR * 100) * 0.92,
            f'$P$={ab_p}', color=C_BEST, fontsize=8, fontweight='bold')

    # Latency on secondary axis
    ax2 = ax.twinx()
    valid_mask = ~np.isnan(P_Lat)
    if np.any(valid_mask):
        ax2.bar(P_vals[valid_mask], P_Lat[valid_mask], width=0.35,
                color=C_LAT, alpha=0.5, zorder=0, label='Latency')
        ax2.set_ylabel('Latency (steps)', fontsize=9, color='#666666')
        ax2.tick_params(axis='y', labelcolor='#666666', labelsize=7)
        ax2.set_ylim(0, np.nanmax(P_Lat) * 1.6)

    ax.set_xlabel('Patience $P$', fontsize=10)
    ax.set_ylabel('Rate (%)', fontsize=10)
    ax.set_title(f'(c) $P$ Ablation + Latency ($K$={selected_k}, $W$={ab_w}, '
                 f'$\\tau$={ab_tau})', fontsize=10, fontweight='bold')
    ax.set_xticks(P_vals)
    ax.spines['top'].set_visible(False)

    # Combined legend
    lines_c = [
        plt.Line2D([0], [0], color=C_EDR, marker='o', lw=1.8, ms=4, label='EDR'),
        plt.Line2D([0], [0], color=C_FER, marker='o', lw=1.8, ms=4, label='FER'),
        plt.Line2D([0], [0], color=C_GAP, marker='o', lw=1.4, ms=4,
                    linestyle='--', label='Gap'),
        Patch(facecolor=C_LAT, alpha=0.5, label='Latency'),
    ]
    ax.legend(handles=lines_c, loc='upper left', fontsize=7)

    # -- (d) Threshold Ablation --
    ax = axes[1, 1]
    ax.plot(T_thresh, T_EDR * 100, 'o-', color=C_EDR, lw=1.8, ms=4)
    ax.plot(T_thresh, T_FER * 100, 'o-', color=C_FER, lw=1.8, ms=4)
    ax.fill_between(T_thresh, T_FER * 100, T_EDR * 100, alpha=0.08, color=C_GAP)
    ax.plot(T_thresh, T_Gap * 100, 'o--', color=C_GAP, lw=1.4, ms=4)
    ax.axvline(x=ab_tau, color=C_BEST, linestyle='--', lw=1.5, alpha=0.8)
    ax.text(ab_tau + 0.002, max(T_EDR * 100) * 0.85,
            f'$\\tau$={ab_tau}', color=C_BEST, fontsize=8, fontweight='bold')
    ax.set_xlabel(r'Threshold $\tau$', fontsize=10)
    ax.set_ylabel('Rate (%)', fontsize=10)
    ax.set_title(f'(d) $\\tau$ Ablation ($K$={selected_k}, $W$={ab_w}, $P$={ab_p})',
                 fontsize=10, fontweight='bold')
    _clean_ax(ax)
    _add_legend(ax)

    plt.tight_layout()
    out = Path(output_dir) / f"fig2_ablation_K{selected_k}.png"
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 6: Generate figures from grid search")
    parser.add_argument(
        "--data_dir",
        default="vlnce_baselines/hivla/5_grid_search/outputs",
        help="Step 5 output dir",
    )
    parser.add_argument(
        "--head_csv",
        default="vlnce_baselines/hivla/4_hnav_selection/outputs/head_importance.csv",
        help="Step 4 head_importance.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="vlnce_baselines/hivla/6_figures/outputs",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    # Load best config from Step 5
    best_cfg_path = Path(args.data_dir) / "best_config.json"
    with open(best_cfg_path) as f:
        best_cfg = json.load(f)
    selected_k = best_cfg["K"]
    ab_w = best_cfg["W"]
    ab_p = best_cfg["P"]
    ab_tau = best_cfg["tau"]

    print(f"Config (from {best_cfg_path}): K={selected_k}, W={ab_w}, P={ab_p}, tau={ab_tau}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    head_imp, abl_K_best, abl_W, abl_P, thresh = load_data(
        args.data_dir, args.head_csv)

    # Generate figures
    fig1a_head_discriminability(head_imp, selected_k, output_dir)
    fig_K_step_na(abl_K_best, selected_k, output_dir)
    fig_K_episode(abl_K_best, selected_k, ab_w, ab_p, output_dir)
    fig2_ablation(abl_K_best, abl_W, abl_P, thresh, selected_k, ab_w, ab_p, ab_tau, output_dir)

    print("\nDone! All figures generated.")


if __name__ == "__main__":
    main()
