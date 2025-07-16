from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import Action

from MacroManagement.resource_types_value import resources
from MacroManagement.vital_characteristics_value import hp_penalty, drink_value, hunger_value

from MacroManagement.task import tasks_pool

class Scenario:
    def __init__(self, state: CraftaxState):
        self.task = tasks_pool["Explore"]
        self.values = dict()
        self.strict_action = None
        for r in resources:
            self.values[r.name] = r.value(state)

    def get_maps(self, state) -> tuple:
        pass

    def is_action(self) -> bool:
        return self.strict_action is not None
    
    def get_action(self) -> Action:
        return self.strict_action
    
    def get_reward(self, state: CraftaxState, next_state: CraftaxState, print_reward=False) -> float:
        reward = 0.0

        #resources impact
        for r in resources:
            reward += r.value(state) * (r.get_amount(next_state) - r.get_amount(state))

        #vital characteristics impact
        reward += hp_penalty(state, next_state)
        if next_state.player_drink > state.player_drink:
            reward += drink_value(state)
        if next_state.player_hunger > state.player_hunger:
            reward += hunger_value(state)

        #task impact
        reward += self.task.reward(state, next_state)

        if print_reward:
            print("Smart reward: ", reward)

        return reward
    