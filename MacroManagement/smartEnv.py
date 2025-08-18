import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np

from craftax.craftax.constants import Action
from craftax.craftax.craftax_state import EnvState as CraftaxState

from tools import CraftaxEnv
from MacroManagement.scenario import Scenario
from MacroManagement.EnvStateWrapper import CraftaxSmartState

move_actions_pool = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
place_block_actions_pool = [Action.PLACE_FURNACE, Action.PLACE_STONE, Action.PLACE_TABLE, Action.PLACE_TORCH]
fight_actions_pool = [Action.SHOOT_ARROW, Action.CAST_FIREBALL, Action.CAST_ICEBALL]
default_actions_pool = move_actions_pool + place_block_actions_pool + [Action.DO]

class SmartEnv(Env):
    def __init__(self, actions_pool: list):
        super().__init__()
        self.env = CraftaxEnv()
        self.actions_pool = actions_pool
        self.state = None
        self.scen = None
        self._seed = None

        self.observation_space = spaces.Dict({
            'map': spaces.Box(low=-10.0, high=10.0, shape=input_dims["map"], dtype=np.float32),
            'local_map': spaces.Box(low=-10.0, high=10.0, shape=input_dims["local_map"], dtype=np.float32),
            'numeric': spaces.Box(low=-1.0, high=1.0, shape=input_dims["numeric"], dtype=np.float32)
        })
        
        self.action_space = spaces.Discrete(len(actions_pool))

    def seed(self, seed):
        self._seed = seed
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.state = self.env.reset(self._seed)
        self.state = CraftaxSmartState(None, self.state)
        self.scen = Scenario(self.state)
        return get_observation(self.state, self.scen), {}
    
    def step(self, action: int, eval_mode=False):
        prev_state = self.state
        self.state, rewardt1, done, info = self.env.step(self.actions_pool[action])
        reward = self.scen.get_reward(prev_state, self.state)

        if not done:
            self.state, rewardt2, done, info = self._execute_tricky_actions(self.state)
            if eval_mode:
                reward = rewardt1 + rewardt2

        self.state = CraftaxSmartState(prev_state, self.state)
        return (get_observation(self.state, self.scen),
                float(reward) if not done else -10.0,
                False,
                done,
                {})

    def _execute_tricky_actions(self, state):
        self.scen = Scenario(state)
        done = False
        cum_reward = 0.0
        while self.scen.is_action():
            state, reward, done, info = self.env.step(self.scen.get_action())
            cum_reward += reward
            self.scen = Scenario(state)
        return state, cum_reward, done, {}
    
gym.register(
    id='SmartEnv-v0',
    entry_point='MacroManagement.smartEnv:SmartEnv',
    kwargs={'actions_pool': default_actions_pool}
)

input_dims = {
    "map": (5, 48, 48),
    "local_map": (11, 9, 11),
    "numeric": (18,)
}
def get_observation(state: CraftaxSmartState, scen: Scenario):
    res = {}
    res["numeric"] = np.array([
        np.tanh(state.player_food),
        np.tanh(state.player_drink),
        np.tanh(state.player_health),
        np.tanh(state.player_mana),
        np.tanh(state.inventory.arrows),
        np.tanh(state.inventory.stone),
        np.tanh(state.inventory.torches),
        np.tanh(state.inventory.wood),
        *scen.get_task_mask(),
        *state.get_direction_mask()
        ], dtype=np.float32)
    
    res["map"] = np.array([
        state.get_explored_mask(),
        state.get_player_mask(),
        state.get_water_mask(),
        state.get_value_map(scen),
        state.get_path_map(scen)
    ], dtype=np.float32)

    res["local_map"] = np.array([
        state.get_local_walkable_mask(),
        state.get_local_breakable_mask(),
        state.get_local_buildable_mask(),
        state.get_local_value_map(scen),
        state.get_local_path_map(scen),
        state.get_local_entity_mask(),
        state.get_local_light_map(),
        *state.get_local_projectile_maps()
    ], dtype=np.float32)
    return res

def print_obs(obs):
    numeric = [
        (1, "player food"),
        (1, "player drink"),
        (1, "player health"),
        (1, "player mana"),
        (1, "arrows"),
        (1, "stone"),
        (1, "torches"),
        (1, "wood"),
        (5, "task"),
        (5, "direction")
    ]
    ptr = 0
    for len, name in numeric:
        print(f"{name}: {obs['numeric'][ptr : ptr+len]}")
        ptr += len
    local = [
        (1, "walkable"),
        (1, "breakable"),
        (1, "buildable"),
        (1, "value"),
        (1, "path"),
        (1, "entity"),
        (1, "light")
    ]
    ptr = 0
    for len, name in local:
        print(f"{name}:")
        if (name != "value"):
            for i in range(9):
                out = "    "
                for j in range(11):
                    out += str(int(obs["local_map"][ptr][i][j]))
                print(out)
        else:
            for i in range(9):
                out = "    "
                for j in range(11):
                    out += str(obs["local_map"][ptr][i][j])
                print(out)
        ptr += len
    