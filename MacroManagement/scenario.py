from craftax.craftax.craftax_state import EnvState as CraftaxState

from MacroManagement.resource import resources
from MacroManagement.task import tasks_pool

class Scenario:
    def __init__(self, state: CraftaxState):
        self.task = tasks_pool["Explore"]
        self.values = dict()
        self.strict_action = None
        for r in resources:
            self.values[r.name] = r.value(state)

    def get_maps(self, state):
        pass

    def is_action(self):
        return self.strict_action is not None
    
    def get_action(self):
        return self.strict_action
    
    def get_reward(self, state: CraftaxState, next_state: CraftaxState, print_reward=False) -> float:
        reward = 0.0
        for r in resources:
            reward += r.value(state) * (r.get_amount(next_state) - r.get_amount(state))
        #hp losing should be punished
        reward += self.task.reward(state, next_state)
        if print_reward:
            print("Smart reward: ", reward)
        return reward
    