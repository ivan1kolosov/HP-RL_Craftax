import bz2
import pickle
import time
import jax
from pathlib import Path
from craftax.craftax_env import make_craftax_env_from_name

from craftax.craftax.constants import Achievement

from Render.Renderer import CraftaxRenderer
from MacroManagement.scenario import Scenario

class CraftaxEnv:
    def __init__(self):
        self.env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
        self.params = self.env.default_params
        self.state = None
        self.rng = None

    def __get_rng(self):
        self.rng, _rng = jax.random.split(self.rng)
        return _rng

    def reset(self, seed):
        self.episode = SmartEpisode(seed)
        self.rng = jax.random.PRNGKey(seed)
        obs, state = self.env.reset(self.__get_rng(), self.params)
        self.state = state
        return state
    
    def step(self, action, print_achievements=False):
        self.episode.add_step(action)
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
    
    def save_last_episode(self):
        self.episode.save()
    
def save_in_folder(object, name, directory):
    save_name = directory + f"/{name}"
    Path(directory).mkdir(parents=True, exist_ok=True)    
    with bz2.BZ2File(save_name + ".pbz2", "w") as f:
        pickle.dump(object, f)

def load_object(name, directory):
    with bz2.open(directory + "/" + name, "rb") as f:
        res = pickle.load(f)
        return res

class SmartEpisode:
    def __init__(self, seed=None):
        self.seed = seed
        self.actions = []

    def add_step(self, action):
        self.actions.append(action)

    def generate_video(self, name="smart_episode", directory="videos"):
        episode = Episode()
        env = CraftaxEnv()
        state = env.reset(self.seed)
        for action in self.actions:
            next_state, reward, done, info = env.step(action)
            episode.add(state, action, reward)
            state = next_state
        episode.generate_video(name, directory)


    def save(self, directory="play_data"):
        save_in_folder(self, f"smart_episode_{int(time.time())}", directory=directory)

    def load(self, name, directory="play_data"):
        obj = load_object(name, directory)
        self.seed = obj.seed
        self.actions = obj.actions

class Episode:
    def __init__(self):
        self.data = {"state": [], "action": [], "reward": []}

    def add(self, state, action, reward):
        self.data["state"].append(state)
        self.data["action"].append(action)
        self.data["reward"].append(reward)

    def save(self, directory="play_data"):
        save_name = f"episode_{int(time.time())}"
        save_in_folder(self, save_name, directory=directory)
    
    def get_sum_reward(self):
        return sum(self.data["reward"])
    
    def load(self, name, directory="play_data"):
        self.data = load_object(name, directory).data

    def generate_video(self, name="episode", directory="videos"):
        Path(directory).mkdir(parents=True, exist_ok=True)
        renderer = CraftaxRenderer()
        renderer.start_video_recording()
        for state in self.data["state"]:
            renderer.render(state, Scenario(state))
        renderer.stop_video_recording(f"{directory}/{name}_{int(time.time())}.mp4", fps=10)