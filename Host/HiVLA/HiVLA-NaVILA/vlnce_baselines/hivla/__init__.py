"""
HiVLA: Hierarchical VLN Anomaly Detection Pipeline for NaVILA.

Pipeline:
  1_htmp_extraction/    H_tmp head extraction (Diagonal-Shifting)
  2_data_extraction/    Per-step attention metrics + action_prob (eval)
  3_gt_labeling/        Waypoint GT labeling (offline)
  4_hnav_selection/     H_nav ranking by Cohen's d (offline)
  5_grid_search/        Hyperparameter sweep over (K, W, P, tau)
  6_figures/            Figure generation
  7_comparison/         Baseline comparison
  _runtime/             Trainer runtime utilities
"""

__version__ = "1.0.0"

import importlib as _il

from vlnce_baselines.hivla.hivla_trainer import HiVLATrainer

_htmp = _il.import_module("vlnce_baselines.hivla.1_htmp_extraction")
_data = _il.import_module("vlnce_baselines.hivla.2_data_extraction")

HeadExtractor = _htmp.HeadExtractor
TemporalAnomalyLogger = _data.TemporalAnomalyLogger

__all__ = ["HiVLATrainer", "HeadExtractor", "TemporalAnomalyLogger"]
