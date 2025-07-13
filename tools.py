import bz2
import pickle
import time
import jax
from pathlib import Path
from craftax.craftax_env import make_craftax_env_from_name

from craftax.craftax.constants import Achievement

class CraftaxEnv:
    def __init__(self, seed):
        self.env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
        self.params = self.env.default_params
        self.rng = jax.random.PRNGKey(seed)
        self.state = None

    def __get_rng(self):
        self.rng, _rng = jax.random.split(self.rng)
        return _rng

    def reset(self):
        obs, state = self.env.reset(self.__get_rng(), self.params)
        self.state = state
        return state
    
    def step(self, action, print_achievements=False):
        old_achievements = self.state.achievements

        obs, next_state, reward, done, info = self.env.step(
            self.__get_rng(), self.state, action, self.params
        )
        self.state = next_state

        if print_achievements:
            new_achievements = next_state.achievements

            for i in range(len(old_achievements)):
                if old_achievements[i] == 0 and new_achievements[i] == 1:
                    print(
                        f"{Achievement(i).name} ({new_achievements.sum()}/{len(Achievement)})"
                    )

            if reward > 0.8:
                print(f"Reward: {reward}\n")

        return next_state, reward, done, info
    
class Trajectory:
    def __init__(self):
        self.data = {"state": [], "action": [], "reward": []}

    def add(self, state, action, reward):
        self.data["state"].append(state)
        self.data["action"].append(action)
        self.data["reward"].append(reward)

    def save(self, directory="play_data"):
        save_name = directory + f"/trajectory_{int(time.time())}"
        save_name += ".pkl"
        Path(directory).mkdir(parents=True, exist_ok=True)    
        with bz2.BZ2File(save_name + ".pbz2", "w") as f:
            pickle.dump(self.data, f)