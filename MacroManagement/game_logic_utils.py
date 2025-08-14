import numpy as np

from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import SOLID_BLOCKS, ItemType, BlockType

def player_in_safety(state: CraftaxState) -> bool:
    return False #not implemented yet

def did_kill_enemy(state: CraftaxState, next_state: CraftaxState) -> bool:
    return False #Not Implemented yet

def is_in_solid_block_mask(state: CraftaxState):
    return np.isin(state.map[state.player_level], SOLID_BLOCKS)

def can_place_stone_mask(state: CraftaxState):
    is_placement_on_solid_block_or_item = (
        is_in_solid_block_mask(state) |
        state.item_map[state.player_level] != ItemType.NONE.value
    )
    return ~is_placement_on_solid_block_or_item & ~state.mob_map[state.player_level]

def can_break_mask(state: CraftaxState):
    break_levels = [
        [BlockType.CHEST.value, BlockType.TREE.value, 
        BlockType.ICE_SHRUB.value, BlockType.FIRE_TREE.value],
        [BlockType.STONE.value],
        [BlockType.IRON.value],
        [BlockType.DIAMOND.value],
        [BlockType.SAPPHIRE.value, BlockType.RUBY.value],
    ]
    
    breakable = []
    for i in range(state.inventory.pickaxe + 1):
        breakable += break_levels[i]
    return np.isin(state.map[state.player_level], breakable)

def can_walk_mask(state: CraftaxState):
    return (~is_in_solid_block_mask(state) 
            & ~np.isin(state.map[state.player_level], [BlockType.WATER, BlockType.LAVA])
            & ~state.mob_map[state.player_level])