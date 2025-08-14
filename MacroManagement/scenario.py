import numpy as np

from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import Action, BlockType

from MacroManagement.resource_types import resources
from MacroManagement.vital_characteristics_value import hp_penalty, drink_value, hunger_value

from MacroManagement.task import tasks_pool, TaskType

class Scenario:
    def __init__(self, state: CraftaxState):
        self.task = tasks_pool[TaskType.EXPLORE]
        self.strict_action = None
        self.init_values(state)

    def init_values(self, state: CraftaxState):
        self.values = dict()
        for r in resources:
            self.values[r.name] = r.value(state)
        for key, val in self.get_mobs_value(state).items():
            self.values[key] = val
        self.values["water"] = drink_value(state)

    def get_blocks_value(self, state) -> dict:
        blocks_value = dict()
        for r in resources:
            for block in r.blocks:
                blocks_value[block] = r.value(state)
        blocks_value[BlockType.WATER] = blocks_value[BlockType.FOUNTAIN] = drink_value(state)
        return blocks_value

    def get_mobs_value(self, state) -> dict:
        mobs_value = dict()
        mobs_value["enemy"] = 5.0 if self.task.type == TaskType.FIGHT else -2.0
        mobs_value["friend"] = hunger_value(state)
        return mobs_value

    def is_action(self) -> bool:
        return self.strict_action is not None
    
    def get_action(self) -> Action:
        return self.strict_action
    
    def get_reward(self, state: CraftaxState, next_state: CraftaxState, print_reward=False) -> float:
        reward = 0.0

        #vital characteristics impact
        reward += hp_penalty(state, next_state)
        if next_state.player_drink > state.player_drink:
            reward += drink_value(state)
        if next_state.player_food > state.player_food:
            reward += hunger_value(state)

        #task impact
        reward += self.task.reward(state, next_state)

        if print_reward:
            print("Smart reward: ", reward)

        return reward
    
    def get_task_mask(self):
        res = [0.0] * len(tasks_pool)
        res[self.task.number] = 1.0
        return res
        
    