from enum import Enum

from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import BlockType
from craftax.craftax.util.game_logic_utils import is_near_block

from MacroManagement.game_logic_utils import player_in_safety, did_kill_enemy
from MacroManagement.resource_types import resources

class TaskType(Enum):
    FIGHT = "Fight"
    EXPLORE = "Explore"
    PLACE_CRAFTING_TABLE = "Place crafting table"
    PLACE_FURNACE = "Place furnace"
    BUILD_FORTRESS = "Build fortress"

class Task:
    def __init__(self, type, reward_f, number):
        self.type = type
        self.reward_f = reward_f
        self.number = number

    def reward(self, state: CraftaxState, next_state: CraftaxState) -> float:
        return self.reward_f(state, next_state)

def explore_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    reward = 0.0
    for r in resources:
        reward += r.value(state) * (r.get_amount(next_state) - r.get_amount(state))
    return -0.05 if reward == 0.0 else reward

def place_crafting_table_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    is_at_crafting_table = is_near_block(next_state, BlockType.CRAFTING_TABLE.value)
    return 5.0 if is_at_crafting_table else -0.15
    
def place_furnace_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    is_at_crafting_table = is_near_block(next_state, BlockType.CRAFTING_TABLE.value)
    is_at_furnace = is_near_block(next_state, BlockType.FURNACE.value)
    can_use_furnace = is_at_crafting_table and is_at_furnace
    return 5.0 if can_use_furnace else -0.15

def build_fortress_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    in_safety = player_in_safety(next_state)
    return 10.0 if in_safety else -0.15
    
def fight_reward(state: CraftaxState, next_state: CraftaxState) -> float:
    enemy_killed = did_kill_enemy(state, next_state)
    return 5.0 if enemy_killed else -0.15

tasks_pool = {
    TaskType.EXPLORE: Task(TaskType.EXPLORE, explore_reward, 0),
    TaskType.PLACE_CRAFTING_TABLE: Task(TaskType.PLACE_CRAFTING_TABLE, place_crafting_table_reward, 1),
    TaskType.PLACE_FURNACE: Task(TaskType.PLACE_FURNACE, place_furnace_reward, 2),
    TaskType.BUILD_FORTRESS: Task(TaskType.BUILD_FORTRESS, build_fortress_reward, 3),
    TaskType.FIGHT: Task(TaskType.FIGHT, fight_reward, 4)
}
