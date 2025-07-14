from craftax.craftax.envs.craftax_symbolic_env import *
from craftax.craftax.constants import BlockType, MobType

class Scenario:
    def __init__(self, state):
        self.values = {BlockType.TREE: 0.5, MobType.MELEE: -1.0}
        self.task = "Explore"

    def get_maps(self, state) -> tuple: #(tensorflow.Tensor, tensorflow.Tensor)
        return NotImplementedError
    
    def is_action(self):
        return NotImplementedError
    
    def get_action(self):
        return NotImplementedError
    
    def get_reward(self, state, next_state) -> float:
        return NotImplementedError