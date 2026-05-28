#!/bin/bash
# NaVid-HiVLA Pipeline Runner
#
# Usage:
#   bash eval_navid_hivla.sh                  # all splits, full pipeline (1→7)
#   bash eval_navid_hivla.sh train             # train only (steps 1→5)
#   bash eval_navid_hivla.sh val_seen          # val_seen only (steps 2→3)
#   bash eval_navid_hivla.sh val_unseen        # val_unseen only (steps 2→3)
#   bash eval_navid_hivla.sh 7                 # step 7 only (val_seen + val_unseen)
#   bash eval_navid_hivla.sh train 1           # step 1 only
#   bash eval_navid_hivla.sh train 2           # step 2 on train only
#   bash eval_navid_hivla.sh val_seen 2        # step 2 on val_seen only
#   bash eval_navid_hivla.sh val_seen 3        # step 3 on val_seen only
#
# Steps:
#   1   H_tmp Extraction     (train split, SPL = 1.0 only)
#   2   Data Extraction      (attention metrics + action_prob → JSONL)
#   3   GT Labeling          (Normal / Anomaly from reference path)
#   4   H_nav Selection      (Cohen's d head ranking)
#   5   Grid Search          (W×K×P×τ sweep → best config)
#   7   Comparison           (Stagnation / Act.Failure / Uncertainty / Ours)
# ============================================================

export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore"
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"
export TRANSFORMERS_VERBOSITY=error

# ──────────────────────────────────────────────────────────────
# Configuration — edit these for your setup
# ──────────────────────────────────────────────────────────────
MODEL_PATH="model_zoo/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split"
CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
DATASET_DIR="data/datasets/R2R_VLNCE_v1-3_preprocessed"
GPU_LIST="0,1,2,3"          # comma-separated GPU IDs for parallel chunking
TOTAL_CHUNKS=4              # set > 1 for multi-GPU
IDX_START=0
SPL_THRESHOLD=1.0           # Step 1: only perfectly successful episodes (SPL=1) for H_tmp
TRAIN_EPISODES=1000         # Steps 1+2: total train episodes (split evenly across chunks; val=-1=all)
HEAD_RATIO=0.1              # Step 2: top-ratio of H_tmp heads to monitor

# Output directories (all under eval_out/hivla/)
OUT_BASE="eval_out/hivla"
DIR_HTMP="${OUT_BASE}/1_htmp"        # Step 1 outputs
DIR_DATA="${OUT_BASE}/2_data"        # Step 2 per-split anomaly_metrics
DIR_GT="${OUT_BASE}/3_gt"            # Step 3 per-split episode_diagnostics
DIR_HNAV="${OUT_BASE}/4_hnav"        # Step 4 head_importance.csv + hnav_config.json
DIR_GRID="${OUT_BASE}/5_grid"        # Step 5 full_sweep.csv + best_config.json
DIR_FIG="${OUT_BASE}/6_figures"      # Step 6 publication figures
DIR_CMP="${OUT_BASE}/7_comparison"   # Step 7 comparison CSVs
# ──────────────────────────────────────────────────────────────

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

# Individual step runners
_step1() {
    local EPS_PER_CHUNK=$(( TRAIN_EPISODES / CHUNKS ))
    echo "=== Step 1: H_tmp Extraction (train, SPL>=${SPL_THRESHOLD}, ${TRAIN_EPISODES} eps) ==="
    mkdir -p "${DIR_HTMP}/outputs"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CHUNK_IDX=$((IDX + IDX_START))
        echo "  Chunk ${CHUNK_IDX}/${TOTAL_CHUNKS} on GPU ${GPULIST[$IDX]}"
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run_hivla.py \
            --exp-config    $CONFIG_PATH \
            --split         train \
            --split-num     $TOTAL_CHUNKS \
            --split-id      $CHUNK_IDX \
            --model-path    $MODEL_PATH \
            --result-path   "${DIR_HTMP}/results" \
            --mode          htmp_extraction \
            --spl-threshold   $SPL_THRESHOLD \
            --max-episodes    $EPS_PER_CHUNK \
            --htmp-output-dir "${DIR_HTMP}/outputs" &
    done
    wait

    echo ""
    echo "--- Merging H_tmp chunks (${TOTAL_CHUNKS} chunk(s)) ---"
    python -c "
import sys, importlib
sys.path.insert(0, '.')
HeadExtractor = importlib.import_module('hivla.1_htmp_extraction').HeadExtractor
HeadExtractor.merge_h_tmp_chunks('${DIR_HTMP}/outputs', ${TOTAL_CHUNKS})
"
    echo "  Head ranking → ${DIR_HTMP}/outputs/temporal_head_importance.json"
}

_step2() {
    local SPLIT="${1:-val_seen}"
    local EPS_PER_CHUNK
    if [ "$SPLIT" = "train" ]; then
        EPS_PER_CHUNK=$(( TRAIN_EPISODES / CHUNKS ))
    else
        EPS_PER_CHUNK=0   # 0 = no limit (all episodes)
    fi
    echo "=== Step 2: Data Extraction (${SPLIT}, eps/chunk=${EPS_PER_CHUNK:-all}) ==="
    mkdir -p "${DIR_DATA}/${SPLIT}"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CHUNK_IDX=$((IDX + IDX_START))
        OUTPUT_PATH="${DIR_DATA}/${SPLIT}/anomaly_metrics_chunk${CHUNK_IDX}.jsonl"
        echo "  Chunk ${CHUNK_IDX}/${TOTAL_CHUNKS} on GPU ${GPULIST[$IDX]}"
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run_hivla.py \
            --exp-config    $CONFIG_PATH \
            --split         $SPLIT \
            --split-num     $TOTAL_CHUNKS \
            --split-id      $CHUNK_IDX \
            --model-path    $MODEL_PATH \
            --result-path   "${DIR_DATA}/${SPLIT}" \
            --mode          data_extraction \
            --output-path   "$OUTPUT_PATH" \
            --head-path     "${DIR_HTMP}/outputs/temporal_head_importance.json" \
            --head-ratio    $HEAD_RATIO \
            --max-episodes  $EPS_PER_CHUNK &
    done
    wait
    echo "  Metrics → ${DIR_DATA}/${SPLIT}/"
}

_step3() {
    local SPLIT="${1:-val_seen}"
    echo "=== Step 3: GT Labeling (${SPLIT}) ==="
    python hivla/3_gt_labeling/waypoint_gt_labeling.py \
        --data_dir    "${DIR_DATA}/${SPLIT}" \
        --dataset_dir $DATASET_DIR \
        --split       $SPLIT \
        --output_dir  "${DIR_GT}/${SPLIT}"
    echo "  GT → ${DIR_GT}/${SPLIT}/episode_diagnostics.jsonl"
}

_step4() {
    local SPLIT="${1:-train}"
    echo "=== Step 4: H_nav Selection (Cohen's d on ${SPLIT}) ==="
    python hivla/4_hnav_selection/hnav_selection.py \
        --entropy_dir "${DIR_DATA}/${SPLIT}" \
        --gt_dir      "${DIR_GT}/${SPLIT}" \
        --output_dir  "${DIR_HNAV}/outputs"
    echo "  H_nav config → ${DIR_HNAV}/outputs/hnav_config.json"
}

_step5() {
    local SPLIT="${1:-train}"
    echo "=== Step 5: Grid Search (${SPLIT}) ==="
    python hivla/5_grid_search/grid_search.py \
        --entropy_dir "${DIR_DATA}/${SPLIT}" \
        --gt_dir      "${DIR_GT}/${SPLIT}" \
        --head_csv    "${DIR_HNAV}/outputs/head_importance.csv" \
        --output_dir  "${DIR_GRID}/outputs"
    echo "  Best config → ${DIR_GRID}/outputs/best_config.json"
}

_step6() {
    echo "=== Step 6: Figure Generation ==="
    python hivla/6_figures/generate_figures.py \
        --data_dir  "${DIR_GRID}/outputs" \
        --head_csv  "${DIR_HNAV}/outputs/head_importance.csv" \
        --output_dir "${DIR_FIG}/outputs"
    echo "  Figures → ${DIR_FIG}/outputs/"
}

_step7() {
    echo "=== Step 7: Comparison (Stagnation / Act.Failure / Uncertainty / Ours) ==="
    local ARGS=""

    if [ -f "${DIR_GT}/val_seen/episode_diagnostics.jsonl" ]; then
        ARGS="$ARGS --gt_seen     ${DIR_GT}/val_seen/episode_diagnostics.jsonl"
        ARGS="$ARGS --metrics_seen ${DIR_DATA}/val_seen"
        echo "  val_seen  GT: ${DIR_GT}/val_seen/episode_diagnostics.jsonl"
    else
        echo "  [WARN] val_seen GT not found — run: bash eval_navid_hivla.sh val_seen 3"
    fi

    if [ -f "${DIR_GT}/val_unseen/episode_diagnostics.jsonl" ]; then
        ARGS="$ARGS --gt_unseen     ${DIR_GT}/val_unseen/episode_diagnostics.jsonl"
        ARGS="$ARGS --metrics_unseen ${DIR_DATA}/val_unseen"
        echo "  val_unseen GT: ${DIR_GT}/val_unseen/episode_diagnostics.jsonl"
    else
        echo "  [WARN] val_unseen GT not found — run: bash eval_navid_hivla.sh val_unseen 3"
    fi

    python hivla/7_comparison/compare_from_gt.py \
        $ARGS \
        --head_config "${DIR_HNAV}/outputs/hnav_config.json" \
        --best_config "${DIR_GRID}/outputs/best_config.json" \
        --output_dir  "${DIR_CMP}/outputs"
    echo "  Results → ${DIR_CMP}/outputs/"
}

# Pipeline definitions
_pipeline_train() {
    _step1
    _step2 train
    _step3 train
    _step4 train
    _step5 train
    _step6
}

_pipeline_val_seen() {
    _step2 val_seen
    _step3 val_seen
}

_pipeline_val_unseen() {
    _step2 val_unseen
    _step3 val_unseen
}

_pipeline_all() {
    _pipeline_train
    _pipeline_val_seen
    _pipeline_val_unseen
    _step7
}

# ──────────────────────────────────────────────────────────────
# Argument parsing
#   $1 = split (train / val_seen / val_unseen) or step number or empty
#   $2 = step number (optional; runs only that step for the given split)
# ──────────────────────────────────────────────────────────────
ARG1="${1:-}"
ARG2="${2:-}"

if [ -z "$ARG1" ]; then
    echo "=== HiVLA Full Pipeline (train → val_seen → val_unseen → step 7) ==="
    _pipeline_all

elif [ "$ARG1" = "7" ]; then
    _step7

elif [ "$ARG1" = "train" ]; then
    if [ -z "$ARG2" ]; then
        echo "=== HiVLA Train Pipeline (steps 1→6) ==="
        _pipeline_train
    else
        case "$ARG2" in
            1) _step1 ;;
            2) _step2 train ;;
            3) _step3 train ;;
            4) _step4 train ;;
            5) _step5 train ;;
            6) _step6 ;;
            7) _step7 ;;
            *) echo "Unknown step: $ARG2"; exit 1 ;;
        esac
    fi

elif [ "$ARG1" = "val_seen" ]; then
    if [ -z "$ARG2" ]; then
        echo "=== HiVLA val_seen Pipeline (steps 2→3) ==="
        _pipeline_val_seen
    else
        case "$ARG2" in
            2) _step2 val_seen ;;
            3) _step3 val_seen ;;
            7) _step7 ;;
            *) echo "Unknown step: $ARG2"; exit 1 ;;
        esac
    fi

elif [ "$ARG1" = "val_unseen" ]; then
    if [ -z "$ARG2" ]; then
        echo "=== HiVLA val_unseen Pipeline (steps 2→3) ==="
        _pipeline_val_unseen
    else
        case "$ARG2" in
            2) _step2 val_unseen ;;
            3) _step3 val_unseen ;;
            7) _step7 ;;
            *) echo "Unknown step: $ARG2"; exit 1 ;;
        esac
    fi

else
    echo "Unknown argument: $ARG1"
    echo ""
    echo "Usage:"
    echo "  bash eval_navid_hivla.sh                  # full pipeline (all splits)"
    echo "  bash eval_navid_hivla.sh train             # train pipeline (steps 1→5)"
    echo "  bash eval_navid_hivla.sh val_seen          # val_seen pipeline (steps 2→3)"
    echo "  bash eval_navid_hivla.sh val_unseen        # val_unseen pipeline (steps 2→3)"
    echo "  bash eval_navid_hivla.sh 7                 # step 7 only"
    echo "  bash eval_navid_hivla.sh train 1           # step 1 only"
    echo "  bash eval_navid_hivla.sh train 2           # step 2 on train"
    echo "  bash eval_navid_hivla.sh val_seen 2        # step 2 on val_seen"
    echo "  bash eval_navid_hivla.sh val_seen 3        # step 3 on val_seen"
    exit 1
fi

echo ""
echo "Done."
