from tools import CraftaxEnv
from MacroManagement.scenario import Scenario
from craftax.craftax.constants import Action
from gym import spaces, Env
import numpy as np

move_actions_pool = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
place_block_actions_pool = [Action.PLACE_FURNACE, Action.PLACE_STONE, Action.PLACE_TABLE, Action.PLACE_TORCH]
fight_actions_pool = [Action.SHOOT_ARROW, Action.CAST_FIREBALL, Action.CAST_ICEBALL]
default_actions_pool = move_actions_pool + place_block_actions_pool + [Action.DO]

input_dims = {
    "map": (48, 48, 10),
    "local_map": (11, 11, 20),
    "numeric": (10,)
}

class SmartEnv(Env):
    def __init__(self, actions_pool: list):
        super().__init__()
        self.env = CraftaxEnv()
        self.actions_pool = actions_pool
        self.state = None
        self.scen = None

        self.observation_space = spaces.Dict({
            'map': spaces.Box(low=0.0, high=1.0, shape=input_dims["map"], dtype=np.float32),
            'local_map': spaces.Box(low=0.0, high=1.0, shape=input_dims["local_map"], dtype=np.float32),
            'numeric': spaces.Box(low=-1.0, high=1.0, shape=input_dims["numeric"], dtype=np.float32)
        })
        
        self.action_space = spaces.Discrete(len(actions_pool))

    def reset(self, seed, options=None):
        obs, self.state = self.env.reset(seed)
        self.scen = Scenario(self.state)
        return get_observation(self.state, self.scen), {}
    
    def step(self, action: int):
        next_state, reward, done, info = self.env.step(self.actions_pool[action])
        reward = self.scen.get_reward(self.state, next_state, done)

        if not done:
            self.state, reward1, done, info = self._execute_tricky_actions(next_state)
            reward += reward1

        return (get_observation(self.state, self.scen),
                reward if not done else -10.0,
                done,
                False,
                {})

    def _execute_tricky_actions(self, state):
        self.scen = Scenario(state)
        done = False
        cum_reward = 0.0
        while self.scen.is_action():
            state, reward, done, info = self.env.step(self.scen.get_action())
            cum_reward += reward
            self.scen = Scenario(state)
        return state, reward, done, {}

def get_observation(state: SmartEnv, scen: Scenario):
    return spaces.Dict({})
