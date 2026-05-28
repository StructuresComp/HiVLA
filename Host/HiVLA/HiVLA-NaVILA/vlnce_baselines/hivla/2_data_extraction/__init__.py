"""
Step 2: Data Extraction - Collect per-step attention metrics via evaluation.

Requires: Step 1 output (temporal_head_importance.json) to know which heads to log.
Output: anomaly_metrics_chunk*.jsonl

Usage:
    Set in navila_hivla.yaml:
        HIVLA.DATA_EXTRACTION.ENABLED: true
    Then run:
        bash scripts/eval/r2r_hivla.sh
"""

from .anomaly_logger import TemporalAnomalyLogger

__all__ = ['TemporalAnomalyLogger']
