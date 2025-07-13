from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
from craftax.craftax.constants import BlockType, MobType

class Scenario:
    def __init__(self):
        self.values = {BlockType.TREE: 0.5, MobType.MELEE: -1.0}
        # self.task = map() #what is the best way to implement this?
        self.task = "Explore"

    def get_maps(self, state) -> tuple: #(tensorflow.Tensor, tensorflow.Tensor)
        return NotImplementedError
    
    def is_action(self):
        return NotImplementedError
    
    def get_action(self):
        return NotImplementedError
    
    def get_reward(self, state: CraftaxSymbolicEnv, next_state: CraftaxSymbolicEnv) -> float:
        return NotImplementedError

class TacticAgent:
    def __init__(self):
        pass

    def get_scen(self, state):
        return Scenario()