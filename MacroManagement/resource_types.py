from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import BlockType, Achievement

class Resource:
    def __init__(self, name, value_f, blocks):
        self.name = name
        self.value_f = value_f
        self.blocks = blocks

    def value(self, state: CraftaxState) -> float:
        return self.value_f(state)
    
    def get_amount(self, state: CraftaxState) -> int:
        return state.inventory.__dict__[self.name]

def required_iron(state: CraftaxState) -> int:
    return max(0, 1 * (state.inventory.sword < 3)
               + 1 * (state.inventory.pickaxe < 3)
               + 3 * (state.inventory.armour < 1).sum()
               - state.inventory.iron)

def required_diamond(state: CraftaxState) -> int:
    return max(0, 2 * (state.inventory.sword < 4) 
               + 3 * (state.inventory.pickaxe < 4)
               + 3 * (state.inventory.armour < 2).sum()
               - state.inventory.diamond)

def value_of_wood(state: CraftaxState) -> float:
    return (1 - state.inventory.wood / 99) * 1.0

def value_of_stone(state: CraftaxState) -> float:
    return max(0.0 + 0.05 * (state.inventory.stone < 99), 1 - state.inventory.stone / 60) * 1.0 * (state.inventory.pickaxe >= 1)

def value_of_coal(state: CraftaxState) -> float:
    return max(0.0, 1 - state.inventory.coal / 99) * 1.1 * (state.inventory.pickaxe >= 1)

def value_of_iron(state: CraftaxState) -> float:
    return 2.0 * (state.inventory.pickaxe >= 2) * bool(required_iron(state))

def value_of_diamond(state: CraftaxState) -> float:
    return 5.0 * (state.inventory.pickaxe >= 3) * bool(required_diamond(state))

def value_of_ruby(state: CraftaxState) -> float:
    return 5.0 * (state.inventory.pickaxe >= 4)

def value_of_sapphire(state: CraftaxState) -> float:
    return 5.0 * (state.inventory.pickaxe >= 4)

def value_of_sapling(state: CraftaxState) -> float:
    return (not state.achievements[Achievement.COLLECT_SAPLING.value]) * 0.15

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
]
