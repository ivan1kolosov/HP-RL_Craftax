import torch as th
import torch.nn as nn
from torch import Tensor
from gym import spaces
from typing import Dict, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        attn_weights = self.attention(x)
        return x * attn_weights

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        map_shape = observation_space['map'].shape
        local_map_shape = observation_space['local_map'].shape
        numeric_dims = observation_space['numeric'].shape[0]
        
        self.map_cnn = nn.Sequential(
            nn.Conv2d(map_shape[0], 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            SimpleCNN(64, 64),
            SpatialAttention(64),
            nn.Flatten()
        )
        
        self.local_map_cnn = nn.Sequential(
            nn.Conv2d(local_map_shape[0], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            SimpleCNN(64, 64),
            SpatialAttention(64),
            nn.Flatten()
        )
        
        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        with th.no_grad():
            map_sample = th.randn(1, *map_shape)
            map_out_dim = self.map_cnn(map_sample).shape[1]

            local_map_sample = th.randn(1, *local_map_shape)
            local_map_out_dim = self.local_map_cnn(local_map_sample).shape[1]

            numeric_sample = th.randn(1, numeric_dims)
            numeric_out_dim = self.numeric_mlp(numeric_sample).shape[1]
        
        combined_dim = map_out_dim + local_map_out_dim + numeric_out_dim
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        map_features = self.map_cnn(observations["map"])

        local_map_features = self.local_map_cnn(observations["local_map"])
        
        numeric_features = self.numeric_mlp(observations["numeric"])
        
        combined = th.cat([map_features, local_map_features, numeric_features], dim=1)
        return self.combine(combined)