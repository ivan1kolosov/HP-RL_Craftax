from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.util.game_logic_utils import get_max_drink, get_max_food
from math import log

def hp_penalty(state: CraftaxState, next_state: CraftaxState) -> float:
    penalty = (log(state.player_health) - log(next_state.player_health)) * 3.0 #int(1/x dx)
    return -penalty

def drink_value(state: CraftaxState) -> float:
    return (1 - state.player_drink / get_max_drink(state))

def hunger_value(state: CraftaxState) -> float:
    return (1 - state.player_hunger / get_max_food(state)) * 3