from craftax.craftax.craftax_state import EnvState as CraftaxState

def player_in_safety(state: CraftaxState) -> bool:
    return False

def did_kill_enemy(state: CraftaxState, next_state: CraftaxState) -> bool:
    return False