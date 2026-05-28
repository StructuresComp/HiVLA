# models/policy/network.py

import torch
import torch.nn as nn

# ==============================================================================
# HiVLA Navigation Policy Network Architecture
# ==============================================================================

class HiVLANavigationPolicy(nn.Module):
    """
    A multi-input policy network for navigation, combining visual (costmap) 
    observations and state (goal vector) observations through a fusion layer.
    """
    def __init__(self, action_dim=2):
        super().__init__()
        
        # 1. Visual Feature Extractor (CNN for Costmap/Image Input)
        # Input: (B, 1, 128, 128) - Assuming costmap dimensions
        self.net_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Calculated output size for a 128x128 input after the final convolution/flatten (15x15x64)
        self.cnn_out_size = 9216
        
        # 2. State Feature Extractor (MLP for Goal Vector Input)
        # Input: (B, 2) - typically [distance to goal, angle to goal]
        self.net_state = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), 
            nn.Linear(64, 64), nn.ReLU()
        )
        self.state_out_size = 64
        
        # 3. Fusion and Action Prediction Head (MLP)
        # Input size is the sum of visual and state features: 9216 + 64
        self.net_fusion = nn.Sequential(
            nn.Linear(self.cnn_out_size + self.state_out_size, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Output: (v, w)
        )

    def forward(self, visual_obs, state_obs):
        """
        Forward pass through the network.

        :param visual_obs: Visual input tensor (Costmap).
        :param state_obs: State input tensor (Goal vector).
        :return: Action tensor (Linear and Angular velocity).
        """
        # 1. Extract features independently
        feat_visual = self.net_cnn(visual_obs)
        feat_state = self.net_state(state_obs)
        
        # 2. Concatenate features along the feature dimension (dim=1)
        fused = torch.cat([feat_visual, feat_state], dim=1)
        
        # 3. Predict action
        return self.net_fusion(fused)