from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

class Scenario:
    def __init__(self):
        self.values = None
        # self.task = map() #what is the best way to implement this?

        self.task = "No task"

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