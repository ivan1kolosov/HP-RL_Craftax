import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from shimmy import GymV21CompatibilityV0

from MacroManagement.smartEnv import SmartEnv, default_actions_pool
from RlTraining.customFeatureExtractor import CustomFeatureExtractor

def train(path=None, name="ppo_smartenv_model"):
    # Конфигурация обучения
    SEED = 123
    ACTIONS_POOL = default_actions_pool
    TOTAL_TIMESTEPS = 1_000_000
    SAVE_FREQ = 100_000
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    env = make_vec_env(lambda: GymV21CompatibilityV0(env=SmartEnv(ACTIONS_POOL)),
                        n_envs=4,
                        seed=SEED)

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_smartenv"
    )

    policy_kwargs = {
            "features_extractor_class": CustomFeatureExtractor,
            "features_extractor_kwargs": {
            "features_dim": 512  # Размер выходного вектора признаков
            },
            "net_arch": {
                "pi": [256, 256],  # Архитектура для policy сети
                "vf": [256, 256]   # Архитектура для value функции
            }
        }

    if path is None:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            seed=SEED,
            tensorboard_log=LOG_DIR,
            device="auto",
            batch_size=256,
            n_steps=2048,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs = policy_kwargs
        )
    else:
        model = model = PPO.load(
            path,
            custom_objects={
                "policy_kwargs": policy_kwargs
            },
            device="auto"  # Автовыбор CPU/GPU
        )

    # model.learn(
    #     total_timesteps=TOTAL_TIMESTEPS,
    #     callback=checkpoint_callback,
    #     tb_log_name="ppo_run"
    # )

    Path("models").mkdir(parents=True, exist_ok=True)

    print("Saving model")
    model.save(f"models/{name}")