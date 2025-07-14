from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import BlockType
from craftax.craftax.util.game_logic_utils import is_near_block

class Task:
    def __init__(self, name, reward_f):
        self.name = name
        self.reward_f = reward_f

    def reward(self, state: CraftaxState, next_state: CraftaxState):
        return self.reward_f(state, next_state)

def explore_reward(state: CraftaxState, next_state: CraftaxState):
    return -0.05

def place_crafting_table_reward(state: CraftaxState, next_state: CraftaxState):
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value)
    if is_at_crafting_table:
        return 10.0
    else:
        return -1.0
    
def place_furnace_reward(state: CraftaxState, next_state: CraftaxState):
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value)
    is_at_furnace = is_near_block(state, BlockType.FURNACE.value)
    if is_at_crafting_table and is_at_furnace:
        return 10.0
    else:
        return -1.0

def build_fortress_reward(state: CraftaxState, next_state: CraftaxState):
    in_safety = False
    if in_safety:
        return 21.0
    else:
        return -0.8

tasks_pool = {
    "Explore": Task("Explore", explore_reward),
    "Place crafting table": Task("Place crafting table", place_crafting_table_reward),
    "Place furnace": Task("Place furnace", place_furnace_reward),
    "Build fortress": Task("Build fortress", build_fortress_reward)
}
