# HiVLA — NaVid Adapter

**HiVLA** (Hierarchical VLN Anomaly Detection) is a plug-in module that adds path-deviation detection to [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE) without modifying the model.

It works by analyzing attention-entropy patterns from the VLM backbone at inference time to detect when the agent has deviated from the intended path.

---

## Prerequisites

Install and verify that **NaVid** runs correctly on your machine before proceeding:

```
https://github.com/jzhzhang/NaVid-VLN-CE
```

Confirm that the standard NaVid evaluation works end-to-end before adding HiVLA.

---

## Installation

Clone this repo, then copy the files into your existing NaVid-VLN-CE root:

```bash
git clone https://github.com/<your-org>/HiVLA-NaVid.git
NAVID_ROOT=/path/to/NaVid-VLN-CE

# New files (added to root)
cp HiVLA-NaVid/agent_navid_hivla.py  $NAVID_ROOT/
cp HiVLA-NaVid/run_hivla.py          $NAVID_ROOT/
cp HiVLA-NaVid/eval_navid_hivla.sh   $NAVID_ROOT/
cp -r HiVLA-NaVid/hivla/             $NAVID_ROOT/

# Patch files (overwrite existing NaVid-VLN-CE files)
cp HiVLA-NaVid/VLN_CE/habitat_extensions/maps.py \
   $NAVID_ROOT/VLN_CE/habitat_extensions/maps.py
cp HiVLA-NaVid/VLN_CE/habitat_extensions/config/vlnce_task_navid_r2r.yaml \
   $NAVID_ROOT/VLN_CE/habitat_extensions/config/vlnce_task_navid_r2r.yaml
cp HiVLA-NaVid/VLN_CE/habitat_extensions/config/vlnce_task_navid_rxr.yaml \
   $NAVID_ROOT/VLN_CE/habitat_extensions/config/vlnce_task_navid_rxr.yaml
```

**Additional Python dependencies** (if not already installed):

```bash
pip install scipy pandas
```

---

## Configuration

Open `eval_navid_hivla.sh` and edit the top section:

```bash
MODEL_PATH="model_zoo/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split"
CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
DATASET_DIR="data/datasets/R2R_VLNCE_v1-3_preprocessed"
GPU_LIST="0,1,2,3"    # GPU IDs to use (one per chunk)
TOTAL_CHUNKS=4        # number of parallel chunks (= number of GPUs)
```

Also update the data paths in the patched YAML files to point to your dataset location:

```yaml
# VLN_CE/habitat_extensions/config/vlnce_task_navid_r2r.yaml
NDTW:
  GT_PATH: data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz
DATASET:
  DATA_PATH: data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz
  SCENES_DIR: data/scene_datasets/
```

All output paths (`eval_out/hivla/...`) are relative to the NaVid-VLN-CE root and do not need to be changed.

---

## Pipeline

Run all commands from the **NaVid-VLN-CE root directory**.

### Option A — Full pipeline (all at once)

```bash
bash eval_navid_hivla.sh          # train → val_seen → val_unseen → step 7
```

### Option B — Step by step

**Phase 1 — Train split** (find H_nav heads and best hyperparameters)

```bash
bash eval_navid_hivla.sh train 1  # Step 1: extract H_tmp heads (SPL = 1.0 episodes only)
bash eval_navid_hivla.sh train 2  # Step 2: attention-metric extraction
bash eval_navid_hivla.sh train 3  # Step 3: GT labeling (Normal / Anomaly)
bash eval_navid_hivla.sh train 4  # Step 4: rank heads by Cohen's d → H_nav config
bash eval_navid_hivla.sh train 5  # Step 5: grid search W × K × P × τ → best_config.json
```

**Phase 2 — Validation splits** (evaluate with found config)

```bash
bash eval_navid_hivla.sh val_seen 2
bash eval_navid_hivla.sh val_seen 3

bash eval_navid_hivla.sh val_unseen 2
bash eval_navid_hivla.sh val_unseen 3

bash eval_navid_hivla.sh 7        # Step 7: compare vs baselines
```

Results are saved under `eval_out/hivla/7_comparison/outputs/`.

---

## Output Files

| Path | Contents |
|---|---|
| `eval_out/hivla/1_htmp/outputs/temporal_head_importance.json` | H_tmp head ranking |
| `eval_out/hivla/2_data/<split>/anomaly_metrics_chunk*.jsonl` | Per-step attention metrics |
| `eval_out/hivla/3_gt/<split>/episode_diagnostics.jsonl` | GT-labeled episodes |
| `eval_out/hivla/4_hnav/outputs/hnav_config.json` | H_nav head config |
| `eval_out/hivla/5_grid/outputs/best_config.json` | Best W/K/P/τ config |
| `eval_out/hivla/7_comparison/outputs/` | Comparison CSVs and LaTeX table |

---

## File Structure

```
HiVLA-NaVid/
├── agent_navid_hivla.py                        # HiVLA-aware NaVid agent  (→ NaVid root)
├── run_hivla.py                                # Evaluation runner, Steps 1–2  (→ NaVid root)
├── eval_navid_hivla.sh                         # Pipeline orchestration  (→ NaVid root)
├── hivla/                                      # HiVLA modules  (→ NaVid root)
│   ├── _runtime/                               #   Attention capture hooks
│   ├── 1_htmp_extraction/                      #   Step 1: H_tmp head ranking
│   ├── 2_data_extraction/                      #   Step 2: attention metric logger
│   ├── 3_gt_labeling/                          #   Step 3: GT labeling
│   ├── 4_hnav_selection/                       #   Step 4: H_nav selection (Cohen's d)
│   ├── 5_grid_search/                          #   Step 5: hyperparameter grid search
│   ├── 6_figures/                              #   Step 6: figure generation
│   └── 7_comparison/                           #   Step 7: baseline comparison
└── VLN_CE/                                     # Patches to NaVid-VLN-CE
    └── habitat_extensions/
        ├── maps.py                             #   Bug fix: bounds check in map rendering
        └── config/
            ├── vlnce_task_navid_r2r.yaml       #   MAX_EPISODE_STEPS: 200, relative data paths
            └── vlnce_task_navid_rxr.yaml       #   Matching RxR config
```
