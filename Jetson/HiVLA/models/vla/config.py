# models/vla/config.py

import os

# ==============================================================================
# NaVILA VLA Configuration
# ==============================================================================

_DEFAULT_NAVILA_ROOT = os.path.join(os.path.expanduser("~"), "HiVLA", "third_party", "NaVILA")

# Model checkpoint path
NAVILA_MODEL_PATH = os.environ.get(
    "NAVILA_MODEL_PATH",
    os.path.join(_DEFAULT_NAVILA_ROOT, "checkpoints", "navila-llama3-8b-8f")
)

# NaVILA repo path (for importing llava modules)
NAVILA_REPO_PATH = os.environ.get(
    "NAVILA_REPO_PATH",
    _DEFAULT_NAVILA_ROOT
)

# Inference settings
NUM_VIDEO_FRAMES = 8        # 7 historical + 1 current
MAX_NEW_TOKENS = 32         # Short action descriptions
TEMPERATURE = 0.0           # Greedy decoding (deterministic)

# NaVILA evaluation path (for importing vlnce_baselines modules)
NAVILA_EVAL_PATH = os.path.join(NAVILA_REPO_PATH, "evaluation")

# HiVLA replanning
HIVLA_TOKENS_PER_IMAGE = 196  # SigLIP 384/14=27 → pad 28×28 → 2×2 downsample → 14×14=196
HIVLA_BEST_CONFIG_PATH = os.environ.get(
    "HIVLA_BEST_CONFIG_PATH",
    os.path.join(
        _DEFAULT_NAVILA_ROOT, "evaluation", "vlnce_baselines",
        "hivla", "5_grid_search", "outputs", "best_config.json"
    ),
)

# Conversation template
CONV_TYPE = "llama_3"

# Navigation prompt template (from NaVILA paper)
NAV_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video "
    "of historical observations {history_images}, and current observation <image>\n. "
    'Your assigned task is: "{instruction}" '
    "Analyze this series of images to decide your next action, which could be turning left "
    "or right by a specific degree, moving forward a certain distance, or stop if the task "
    "is completed."
)
