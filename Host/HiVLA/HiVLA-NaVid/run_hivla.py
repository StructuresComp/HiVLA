#!/usr/bin/env python3
"""
NaVid-HiVLA Evaluation Runner.

Runs the NaVid evaluation loop with the HiVLA path deviation detection
pipeline enabled.  Output is compatible with NaVILA's offline analysis
scripts (Steps 3–5 are model-agnostic and can be reused directly).

Example — data extraction mode (Step 2):
    python run_hivla.py \
        --exp-config VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml \
        --split-num 1 --split-id 0 \
        --model-path model_zoo/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split \
        --result-path tmp/hivla_results \
        --mode data_extraction \
        --output-path tmp/hivla_results/anomaly_metrics_chunk0.jsonl

Example — h_tmp extraction mode (Step 1):
    python run_hivla.py ... --mode htmp_extraction --spl-threshold 1.0

After data extraction, run the offline analysis pipeline:
    Step 3  python hivla/3_gt_labeling/waypoint_gt_labeling.py  ...
    Step 4  python hivla/4_hnav_selection/hnav_selection.py     ...
    Step 5  python hivla/5_grid_search/grid_search.py           ...
"""

import argparse
import json
import os

import numpy as np
from habitat import Env
from habitat.datasets import make_dataset
from tqdm import trange

from VLN_CE.vlnce_baselines.config.default import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-config',   type=str, required=True)
    parser.add_argument('--split-num',    type=int, required=True)
    parser.add_argument('--split-id',     type=int, required=True)
    parser.add_argument('--model-path',   type=str, required=True)
    parser.add_argument('--result-path',  type=str, required=True)

    # HiVLA options
    parser.add_argument(
        '--mode', type=str, default='data_extraction',
        choices=['data_extraction', 'htmp_extraction'],
        help='data_extraction: log attention metrics (Step 2); '
             'htmp_extraction: compute I_diag scores (Step 1)',
    )
    parser.add_argument(
        '--output-path', type=str, default='',
        help='Path for anomaly_metrics JSONL (data_extraction mode). '
             'Defaults to <result-path>/anomaly_metrics_chunk<split-id>.jsonl',
    )
    parser.add_argument(
        '--head-path', type=str, default='',
        help='Path to temporal_head_importance.json for target-head filtering '
             '(optional; if not provided, ALL heads are logged)',
    )
    parser.add_argument(
        '--head-ratio', type=float, default=0.5,
        help='Top-ratio of heads to use from head JSON file (default: 0.5)',
    )
    parser.add_argument(
        '--spl-threshold', type=float, default=1.0,
        help='Min SPL to include episode in h_tmp extraction (default: 1.0)',
    )
    parser.add_argument(
        '--max-episodes', type=int, default=0,
        help='Cap episodes per chunk (0 = no limit). '
             'e.g. 200 with 5 chunks = 1000 total train episodes.',
    )
    parser.add_argument(
        '--htmp-output-dir', type=str, default='',
        help='Directory to save temporal_head_raw_chunk*.json files',
    )
    parser.add_argument(
        '--split', type=str, default='',
        help='Override dataset split (e.g. train, val_seen, val_unseen). '
             'If not set, the value from --exp-config is used.',
    )
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Default paths
    if not args.output_path:
        args.output_path = os.path.join(
            args.result_path,
            f'anomaly_metrics_chunk{args.split_id}.jsonl'
        )
    if not args.htmp_output_dir:
        args.htmp_output_dir = os.path.join(args.result_path, 'htmp_outputs')

    run_exp(args)


def run_exp(args) -> None:
    opts = list(args.opts) if args.opts else []
    if args.split:
        opts += ['TASK_CONFIG.DATASET.SPLIT', args.split]
        opts += ['TASK_CONFIG.TASK.NDTW.SPLIT', args.split]
    config = get_config(args.exp_config, opts if opts else None)

    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(42)

    dataset_split = dataset.get_splits(args.split_num)[args.split_id]

    if args.max_episodes and args.max_episodes < len(dataset_split.episodes):
        dataset_split.episodes = dataset_split.episodes[:args.max_episodes]

    evaluate_agent(config, args, dataset_split)


def evaluate_agent(config, args, dataset) -> None:
    from agent_navid_hivla import NaVidHiVLA_Agent

    # Build hivla_config based on run mode
    hivla_config = {}
    if args.mode == 'data_extraction':
        hivla_config['DATA_EXTRACTION'] = {
            'ENABLED':    True,
            'OUTPUT_PATH': args.output_path,
            'HEAD_PATH':  args.head_path,
            'HEAD_RATIO': args.head_ratio,
        }
        hivla_config['HTMP_EXTRACTION'] = {'ENABLED': False}
    elif args.mode == 'htmp_extraction':
        hivla_config['DATA_EXTRACTION'] = {'ENABLED': False}
        hivla_config['HTMP_EXTRACTION'] = {
            'ENABLED':        True,
            'SPL_THRESHOLD':  args.spl_threshold,
            'OUTPUT_DIR':     args.htmp_output_dir,
        }

    env   = Env(config.TASK_CONFIG, dataset)
    agent = NaVidHiVLA_Agent(args.model_path, args.result_path, hivla_config)

    num_episodes        = len(env.episodes)
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS    = config.EVAL.EARLY_STOP_STEPS
    target_keys         = {'distance_to_goal', 'success', 'spl', 'path_length', 'oracle_success'}

    os.makedirs(os.path.join(args.result_path, 'log'), exist_ok=True)

    for _ in trange(num_episodes, desc=f'HiVLA-{args.mode}-{args.split_id}'):
        obs = env.reset()
        agent.reset()

        iter_step              = 0
        continuse_rotation_count = 0
        last_dtg               = 999.0

        while not env.episode_over:
            info = env.get_metrics()
            pos = env._sim.get_agent_state().position
            info['agent_position'] = pos  # np.array [x, y, z]

            if info['distance_to_goal'] != last_dtg:
                last_dtg = info['distance_to_goal']
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1

            action = agent.act(obs, info, env.current_episode.episode_id)

            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                action = {'action': 0}

            iter_step += 1
            obs = env.step(action)

        info    = env.get_metrics()
        results = {k: info[k] for k in target_keys if k in info}
        results['id'] = env.current_episode.episode_id

        spl = float(results.get('spl', 0.0))
        agent.finish_episode(spl=spl)

        log_path = os.path.join(
            args.result_path, 'log',
            f'stats_{env.current_episode.episode_id}.json'
        )
        with open(log_path, 'w') as f:
            json.dump(results, f, indent=4)

    # Finalize
    agent.close()

    if args.mode == 'htmp_extraction':
        chunk_path = agent.save_htmp_chunk(chunk_idx=args.split_id)
        if chunk_path:
            print(f'[run_hivla] H_tmp chunk saved: {chunk_path}')

    print(f'[run_hivla] Done. Results in: {args.result_path}')


if __name__ == '__main__':
    main()
