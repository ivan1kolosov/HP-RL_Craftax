from craftax.craftax.envs.craftax_symbolic_env import *
from craftax.craftax.constants import BlockType, MobType, MAX_OBS_DIM
from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.util.game_logic_utils import get_max_drink
import math

"""
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    ruby: int
    sapphire: int
"""

class Resource:
    def __init__(self, name, value_f, blocks):
        self.name = name
        self.value_f = value_f
        self.blocks = blocks

    def value(self, state):
        return self.value_f(state)

def required_iron(state: CraftaxState):
    return max(0, 1 * (state.inventory.sword < 3)
               + 1 * (state.inventory.pickaxe < 3)
               + 3 * (state.inventory.armour < 1).sum()
               - state.inventory.iron)
def required_diamond(state: CraftaxState):
    return max(0, 2 * (state.inventory.sword < 4) 
               + 3 * (state.inventory.pickaxe < 4)
               + 3 * (state.inventory.armour < 2).sum()
               - state.inventory.diamond)

def value_of_wood(state: CraftaxState):
    return (1 - state.inventory.wood / 99) * 1.0
def value_of_stone(state: CraftaxState):
    return max(0.0, 1 - state.inventory.stone / 60) * 1.0 * (state.inventory.pickaxe >= 1)
def value_of_coal(state: CraftaxState):
    return max(0.0, 1 - state.inventory.coal / 99) * 1.1 * (state.inventory.pickaxe >= 1)
def value_of_iron(state: CraftaxState):
    return 2.0 * (state.inventory.pickaxe >= 2) * bool(required_iron(state))
def value_of_diamond(state: CraftaxState):
    return 5.0 * (state.inventory.pickaxe >= 3) * bool(required_diamond(state))
def value_of_ruby(state: CraftaxState):
    return 5.0 * (state.inventory.pickaxe >= 4)
def value_of_sapphire(state: CraftaxState):
    return 5.0 * (state.inventory.pickaxe >= 4)
def value_of_sapling(state: CraftaxState):
    return 0.0
def value_of_water(state: CraftaxState):
    max_drink = get_max_drink(state)
    return (1.0 - state.player_drink / max_drink) * 1.5

resources = [
    Resource("wood",
             value_of_wood,
             [BlockType.TREE, BlockType.FIRE_TREE, BlockType.ICE_SHRUB]),
    Resource("stone",
             value_of_stone,
             [BlockType.STONE]),
    Resource("coal",
             value_of_coal,
             [BlockType.COAL]),
    Resource("iron",
             value_of_iron,
             [BlockType.IRON]),
    Resource("diamond",
             value_of_diamond,
             [BlockType.DIAMOND]),
    Resource("ruby",
             value_of_ruby,
             [BlockType.RUBY]),
    Resource("sapphire",
             value_of_sapphire,
             [BlockType.SAPPHIRE]),
    Resource("sapling",
             value_of_sapling,
             [BlockType.GRASS]),
    Resource("water",
             value_of_water,
             [BlockType.WATER, BlockType.FOUNTAIN]
             )
]

class Scenario:
    def __init__(self, state: CraftaxState):
        self.task = "Explore"
        self.values = dict()
        for r in resources:
            self.values[r.name] = r.value(state)

    def get_maps(self, state):
        pass

    def is_action(self):
        return NotImplementedError
    
    def get_action(self):
        return NotImplementedError
    
    def get_reward(self, state, next_state) -> float:
        return NotImplementedError