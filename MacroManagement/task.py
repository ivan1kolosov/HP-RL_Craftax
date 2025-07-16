from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import BlockType
from craftax.craftax.util.game_logic_utils import is_near_block
from enum import Enum

class TaskType(Enum):
    FIGHT = "Fight"
    EXPLORE = "Explore"
    PLACE_CRAFTING_TABLE = "Place crafting table"
    PLACE_FURNACE = "Place furnace"
    BUILD_FORTRESS = "Build fortress"

class Task:
    def __init__(self, type, reward_f):
        self.type = type
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
    TaskType.EXPLORE: Task(TaskType.EXPLORE, explore_reward),
    TaskType.PLACE_CRAFTING_TABLE: Task(TaskType.PLACE_CRAFTING_TABLE, place_crafting_table_reward),
    TaskType.PLACE_FURNACE: Task(TaskType.PLACE_FURNACE, place_furnace_reward),
    TaskType.BUILD_FORTRESS: Task(TaskType.BUILD_FORTRESS, build_fortress_reward),
    TaskType.FIGHT: Task(TaskType.FIGHT, fight_reward)
}
