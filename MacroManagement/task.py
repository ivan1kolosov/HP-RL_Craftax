from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import BlockType
from craftax.craftax.util.game_logic_utils import is_near_block

class Task:
    def __init__(self, name, reward_f):
        self.name = name
        self.reward_f = reward_f

    def reward(self, state: CraftaxState, next_state: CraftaxState) -> float:
        return self.reward_f(state, next_state)

def explore_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    return -0.05

def place_crafting_table_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value)
    return 5.0 if is_at_crafting_table else -0.15
    
def place_furnace_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value)
    is_at_furnace = is_near_block(state, BlockType.FURNACE.value)
    can_use_furnace = is_at_crafting_table and is_at_furnace
    return 5.0 if can_use_furnace else -0.15

def build_fortress_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    in_safety = False
    return 10.0 if in_safety else -0.15
    
def fight_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    defeated_monster = False
    return 5.0 if defeated_monster else -0.15

tasks_pool = {
    "Explore": Task("Explore", explore_reward),
    "Place crafting table": Task("Place crafting table", place_crafting_table_reward),
    "Place furnace": Task("Place furnace", place_furnace_reward),
    "Build fortress": Task("Build fortress", build_fortress_reward),
    "Fight": Task("Fight", fight_reward)
}
