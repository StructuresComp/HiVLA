# HiVLA — NaVILA Adapter

**HiVLA** (Hierarchical VLN Anomaly Detection) is a plug-in module that adds path-deviation detection to [NaVILA](https://github.com/BishengLi0327/NaVILA) without modifying the model.

It works by analyzing attention-entropy patterns from the VLM backbone at inference time to detect when the agent has deviated from the intended path.

---

## Prerequisites

Install and verify that **NaVILA** runs correctly on your machine before proceeding:

```
https://github.com/BishengLi0327/NaVILA
```

Confirm that the standard NaVILA evaluation works end-to-end before adding HiVLA.

---

## Installation

Clone this repo, then copy the files into your `NaVILA/evaluation/` directory:

```bash
git clone https://github.com/<your-org>/HiVLA-NaVILA.git

# Copy everything into your NaVILA/evaluation directory
cp -r HiVLA-NaVILA/vlnce_baselines/   /path/to/NaVILA/evaluation/
cp    HiVLA-NaVILA/eval_navila_hivla.sh /path/to/NaVILA/evaluation/
```

This adds:
- `eval_navila_hivla.sh` — pipeline entry point (root level)
- `vlnce_baselines/hivla/` — HiVLA Python package
- `vlnce_baselines/config/r2r_baselines/navila_hivla.yaml` — HiVLA trainer config

**Additional Python dependencies** (if not already installed):

```bash
pip install scipy pandas
```

---

## Configuration

Open `scripts/eval/r2r_hivla.sh` and edit the top section:

```bash
MODEL_PATH="checkpoints/navila-llama3-8b-8f"
GPU_LIST="0,1,2,3"    # GPU IDs to use (one per chunk)
TOTAL_CHUNKS=4        # number of parallel chunks (= number of GPUs)
```

All other paths are relative to the `NaVILA/evaluation/` root and do not need to be changed.

---

## Pipeline

Run all commands from the **`NaVILA/evaluation/` directory**.

### Phase 1 — Train split (find H_nav heads and best hyperparameters)

```bash
# Step 1: Extract H_tmp heads from successful train episodes (SPL = 1.0)
bash eval_navila_hivla.sh 1

# Step 2: Run full attention-metric extraction on train split
bash eval_navila_hivla.sh 2 train

# Step 3: Assign GT labels (Normal / Anomaly) from reference paths
bash eval_navila_hivla.sh 3 train

# Step 4: Rank heads by Cohen's d → produces H_nav config
bash eval_navila_hivla.sh 4 train

# Step 5: Grid search over W × K × P × τ → produces best_config.json
bash eval_navila_hivla.sh 5 train
```

### Phase 2 — Validation splits (evaluate with found config)

```bash
bash eval_navila_hivla.sh 2 val_seen
bash eval_navila_hivla.sh 3 val_seen

bash eval_navila_hivla.sh 2 val_unseen
bash eval_navila_hivla.sh 3 val_unseen

# Step 7: Compare Stagnation / Act.Failure / Uncertainty / Ours
bash eval_navila_hivla.sh 7 val_seen val_unseen
```

Results are saved under `eval_out/hivla_7_comparison/outputs/` (or the path configured in the script).

---

## Output Files

| Path | Contents |
|---|---|
| `eval_out/hivla_1_htmp/outputs/temporal_head_importance.json` | H_tmp head ranking |
| `eval_out/hivla_2_data/<split>/anomaly_metrics_chunk*.jsonl` | Per-step attention metrics |
| `eval_out/hivla_3_gt/<split>/episode_diagnostics.jsonl` | GT-labeled episodes |
| `eval_out/hivla_4_hnav/outputs/hnav_config.json` | H_nav head config |
| `eval_out/hivla_5_grid/outputs/best_config.json` | Best W/K/P/τ config |
| `eval_out/hivla_7_comparison/outputs/` | Comparison CSVs and LaTeX table |

---

## File Structure

```
(NaVILA/evaluation/ root after install)
├── scripts/eval/r2r_hivla.sh          # Pipeline orchestration script
└── vlnce_baselines/hivla/
    ├── _runtime/                      # Attention capture hooks (injected at eval time)
    ├── 1_htmp_extraction/             # Step 1: H_tmp head ranking
    ├── 2_data_extraction/             # Step 2: attention metric logger
    ├── 2b_action_prob/                # Step 2b: lightweight action-prob extraction
    ├── 3_gt_labeling/                 # Step 3: GT labeling from reference paths
    ├── 4_hnav_selection/              # Step 4: H_nav selection (Cohen's d)
    ├── 5_grid_search/                 # Step 5: hyperparameter grid search
    ├── 6_figures/                     # Step 6: figure generation
    ├── 7_comparison/                  # Step 7: baseline comparison
    └── 8_replanning/                  # Step 8: active replanning module
```
