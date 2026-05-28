# models/policy/inference.py

import torch
import os
import sys

# ==============================================================================
# 1. Custom Module Imports
# ==============================================================================
# Import config and network definition from the same policy module
from .config import DEVICE 
from .network import HiVLANavigationPolicy 

# ==============================================================================
# 2. RL Navigator Class
# ==============================================================================

class RLNavigator:
    """
    A wrapper class for loading and running inference on the HiVLA Navigation Policy.
    """
    def __init__(self, checkpoint_path):
        """
        Initializes the model and loads weights from a checkpoint file.

        :param checkpoint_path: Path to the PyTorch checkpoint file (.pt).
        """
        self.device = DEVICE
        
        # Action Dim = 2 (Linear Velocity: v, Angular Velocity: w)
        self.model = HiVLANavigationPolicy(action_dim=2).to(self.device)
        self.model.eval() # Set model to inference mode
        self._load_checkpoint(checkpoint_path)
        print(f"[RL] Navigator initialized on {self.device}")

    def _load_checkpoint(self, path):
        """Loads the state dictionary from the specified checkpoint path."""
        if not os.path.exists(path):
            # Use sys.stderr for critical errors
            print(f"[RL] Error: Checkpoint not found at: {path}", file=sys.stderr)
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
            
        print(f"[RL] Loading weights from {path}...")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            # Handle standard RL checkpoint formats (e.g., policy key or raw dict)
            state_dict = checkpoint.get("policy", checkpoint)
            
            # Remove 'module.' prefix if the model was saved using DataParallel/DistributedDataParallel
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
            print("[RL] Model weights loaded successfully.")
        except Exception as e:
            print(f"[RL] Error loading checkpoint: {e}", file=sys.stderr)
            raise e

    def get_action(self, costmap_tensor, goal_vector):
        """
        Performs a forward pass to get navigation actions (v, w).

        :param costmap_tensor: Raw costmap tensor from BatchGPULocalCostmapCore.
                               Expected shape: (H, W) or (1, H, W) or (1, 1, H, W)
                               Raw Orientation: Right is Front (+X), Up is Left (+Y)
        :param goal_vector: (..., 2) Tensor [goal_x_local, goal_y_local] normalized.
        :return: Tuple (v, w)
        """
        with torch.no_grad():
            obs = costmap_tensor.float().to(self.device)
            state_in = goal_vector.float().to(self.device)

            if obs.dim() == 2:   # (H, W) -> (1, 1, H, W)
                obs = obs.unsqueeze(0).unsqueeze(0)
            elif obs.dim() == 3: # (C, H, W) or (B, H, W)
                if obs.size(0) == 1: # (1, H, W) -> (1, 1, H, W)
                    obs = obs.unsqueeze(0)
                else:                # (B, H, W) -> (B, 1, H, W)
                    obs = obs.unsqueeze(1)

            actions = self.model(obs, state_in)
            
            return actions[0, 0].item(), actions[0, 1].item()