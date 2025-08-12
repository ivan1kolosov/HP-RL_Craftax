from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

from gym import spaces

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # --------------------------------------------------
        # 1. Ветка для основной карты (map)
        # --------------------------------------------------
        self.map_cnn = nn.Sequential(
            nn.Conv2d(observation_space['map'].shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # --------------------------------------------------
        # 2. Ветка для локальной карты (local_map)
        # --------------------------------------------------
        self.local_map_cnn = nn.Sequential(
            nn.Conv2d(observation_space['local_map'].shape[2], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # --------------------------------------------------
        # 3. Ветка для числовых данных (numeric)
        # --------------------------------------------------
        self.numeric_mlp = nn.Sequential(
            nn.Linear(observation_space['numeric'].shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # --------------------------------------------------
        # Вычисление размеров выходов всех веток
        # --------------------------------------------------
        with th.no_grad():
            # Тестовый прогон для map
            map_sample = th.randn(1, *observation_space['map'].shape).permute(0, 3, 1, 2)
            map_out_dim = self.map_cnn(map_sample).shape[1]
            
            # Тестовый прогон для local_map
            local_map_sample = th.randn(1, *observation_space['local_map'].shape).permute(0, 3, 1, 2)
            local_map_out_dim = self.local_map_cnn(local_map_sample).shape[1]
            
            # Тестовый прогон для numeric
            numeric_sample = th.randn(1, *observation_space['numeric'].shape)
            numeric_out_dim = self.numeric_mlp(numeric_sample).shape[1]
        
        # --------------------------------------------------
        # Комбинированный выход
        # --------------------------------------------------
        self.combine = nn.Sequential(
            nn.Linear(map_out_dim + local_map_out_dim + numeric_out_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # Подготовка входов (BCHW для CNN)
        map_input = observations['map'].permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
        local_map_input = observations['local_map'].permute(0, 3, 1, 2)
        numeric_input = observations['numeric']
        
        # Обработка всех веток
        map_features = self.map_cnn(map_input)
        local_map_features = self.local_map_cnn(local_map_input)
        numeric_features = self.numeric_mlp(numeric_input)
        
        # Комбинация всех признаков
        combined = th.cat([map_features, local_map_features, numeric_features], dim=1)
        return self.combine(combined)