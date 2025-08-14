from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from MacroManagement.smartEnv import default_actions_pool
from RlTraining.customFeatureExtractor import CustomFeatureExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import gymnasium as gym

gym.register(
    id='SmartEnv-v0',
    entry_point='MacroManagement.smartEnv:SmartEnv',
    kwargs={'actions_pool': default_actions_pool}
)

policy_kwargs = {
    "features_extractor_class": CustomFeatureExtractor,
    "features_extractor_kwargs": {
        "features_dim": 512
    },
    "net_arch": {
        "pi": [256, 256],
        "vf": [256, 256]
    }
}

SEED = 123
ACTIONS_POOL = default_actions_pool

class MetricsLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards_buffer = []
        self.lengths_buffer = []
        self.episode_count = 0
        self.log_freq = 10

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards_buffer.append(info["episode"]["r"])
                self.lengths_buffer.append(info["episode"]["l"])
                self.episode_count += 1

        if self.n_calls % self.log_freq == 0 and self.episode_count > 0:
            mean_reward = sum(self.rewards_buffer) / self.episode_count
            mean_length = sum(self.lengths_buffer) / self.episode_count
            
            self.logger.record("train/mean_episode_reward", mean_reward)
            self.logger.record("train/mean_episode_length", mean_length)
            
            self.rewards_buffer = []
            self.lengths_buffer = []
            self.episode_count = 0
        
        return True

def make_env():
    env = gym.make('SmartEnv-v0')
    return env

def get_envs(n_envs=4):
    env = SubprocVecEnv([make_env for _ in range(n_envs)])
    env = VecMonitor(env)
    return env

def load_model(path, env=None):
    return PPO.load(
        path,
        custom_objects={"policy_kwargs": policy_kwargs},
        device="auto",
        env=env
    )

def train(path=None, name="ppo_smartenv_model"):
    TOTAL_TIMESTEPS = 200_000
    SAVE_FREQ = 100_000
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    env = get_envs()

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_smartenv"
    )

    if path is None:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            seed=SEED,
            tensorboard_log=LOG_DIR,
            device="auto",
            batch_size=64,
            n_steps=512,
            n_epochs=10,
            learning_rate=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs
        )
    else:
        model = load_model(path, env)
    metrics_callback = MetricsLoggerCallback()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, metrics_callback],
        tb_log_name="ppo_run"
    )

    Path("models").mkdir(parents=True, exist_ok=True)
    model.save(f"models/{name}")