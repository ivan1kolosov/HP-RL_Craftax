import torch as th
import torch.nn as nn
from torch import Tensor
from gym import spaces
from typing import Dict, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, 
                 expansion: int = 4, dropout: float = 0.1):
        super().__init__()

        assert out_channels % expansion == 0, "out_channels must be divisible by expansion"
        self.expansion = expansion
        inner_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_channels)
        
        self.conv2 = nn.Conv2d(
            inner_channels, inner_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size//2,
            groups=inner_channels,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(inner_channels)
        
        self.conv3 = nn.Conv2d(inner_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        
        return out

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.ln2(out)
        
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        
        return out

class AttentionPooling(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()

        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.query = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(th.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, C, H, W = x.shape
        
        if H * W > 64:
            return x
            
        Q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        K = self.key(x).view(batch_size, -1, H * W)
        V = self.value(x).view(batch_size, -1, H * W)
        
        attention = th.bmm(Q, K)
        attention = th.softmax(attention, dim=-1)
        
        out = th.bmm(V, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)
        
        map_shape = observation_space['map'].shape
        local_map_shape = observation_space['local_map'].shape
        numeric_dims = observation_space['numeric'].shape[0]
        
        self.map_cnn = nn.Sequential(
            nn.Conv2d(map_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualCNNBlock(64, 128, stride=2),
            ResidualCNNBlock(128, 128),
            ResidualCNNBlock(128, 256, stride=2),
            ResidualCNNBlock(256, 256),
            AttentionPooling(256),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.local_map_cnn = nn.Sequential(
            nn.Conv2d(local_map_shape[0], 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            ResidualCNNBlock(32, 64, stride=2),
            ResidualCNNBlock(64, 64),
            AttentionPooling(64),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.numeric_mlp = nn.Sequential(
            ResidualMLPBlock(numeric_dims, 256),
            ResidualMLPBlock(256, 256),
            ResidualMLPBlock(256, 512),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        with th.no_grad():
            map_sample = th.randn(2, *map_shape)  # [2, C, H, W]
            map_out_dim = self.map_cnn(map_sample).shape[1]
            
            local_map_sample = th.randn(2, *local_map_shape)  # [2, C, H, W]
            local_map_out_dim = self.local_map_cnn(local_map_sample).shape[1]
            
            numeric_sample = th.randn(2, numeric_dims)
            numeric_out_dim = self.numeric_mlp(numeric_sample).shape[1]
        
        self.combine = nn.Sequential(
            ResidualMLPBlock(map_out_dim + local_map_out_dim + numeric_out_dim, features_dim),
            ResidualMLPBlock(features_dim, features_dim),
            ResidualMLPBlock(features_dim, features_dim),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        map_input = observations['map']
        local_map_input = observations['local_map']
        numeric_input = observations['numeric']
        
        map_features = self.map_cnn(map_input)
        local_map_features = self.local_map_cnn(local_map_input)
        numeric_features = self.numeric_mlp(numeric_input)
        
        combined = th.cat([map_features, local_map_features, numeric_features], dim=1)
        return self.combine(combined)