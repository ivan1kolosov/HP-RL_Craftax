import numpy as np
from dataclasses import fields

from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import BlockType, MobType

from MacroManagement.scenario import Scenario
from tools import get_local_view

from MacroManagement.game_logic_utils import can_place_stone_mask, can_break_mask, can_walk_mask

class CraftaxSmartState(CraftaxState):
    def __init__(self, prev_state, cur: CraftaxState):
        super().__init__(**cur.__dict__)
        self.explored = prev_state.explored if prev_state is not None else np.zeros((9, 48, 48), dtype=bool)
        self.explored[self.player_level] |= self.get_visible()

    def get_explored_mask(self):
        return self.explored[self.player_level]
    
    def get_player_mask(self):
        res = np.zeros((48, 48))
        res[self.player_position] = 1.0
        return res
    
    def get_water_mask(self):
        floor_map = self.map[self.player_level]
        return ((floor_map == BlockType.WATER.value) | (floor_map == BlockType.FOUNTAIN.value))
    
    def get_value_map(self, scen: Scenario):
        values = scen.get_blocks_value(self)
        value_map = np.zeros((48, 48), dtype=np.float32)
        level = self.player_level
        
        current_map = self.map[level]
        for block_type, value in values.items():
            value_map[current_map == block_type.value] = value
        
        mob_value = scen.get_mobs_value(self)
        
        def add_mob_value(positions, mask, value):
            active_mask = mask[level]
            if not np.any(active_mask):
                return
                
            active_positions = positions[level][active_mask]
            
            rows = active_positions[:, 0].astype(int)
            cols = active_positions[:, 1].astype(int)

            np.add.at(value_map, (rows, cols), value)
        
        add_mob_value(self.melee_mobs.position, self.melee_mobs.mask, mob_value["enemy"])
        add_mob_value(self.ranged_mobs.position, self.ranged_mobs.mask, mob_value["enemy"])
        add_mob_value(self.passive_mobs.position, self.passive_mobs.mask, mob_value["friend"])
        
        return value_map
    
    def get_path_map(self, scen: Scenario):
        return np.zeros((48, 48))
    
    def get_walkable_mask(self):
        return can_walk_mask(self)
    def get_local_walkable_mask(self):
        return self.local(self.get_walkable_mask(), False)

    def get_breakable_mask(self):
        return can_break_mask(self)
    def get_local_breakable_mask(self):
        return self.local(self.get_breakable_mask(), False)

    def get_buildable_mask(self):
        return can_place_stone_mask(self)
    def get_local_buildable_mask(self):
        return self.local(self.get_buildable_mask(), False)

    def get_local_value_map(self, scen):
        return self.local(self.get_value_map(scen), 0.0)

    def get_local_path_map(self, scen):
        return self.local(self.get_path_map(scen), 0.0)

    def get_entity_mask(self):
        return self.mob_map[self.player_level]
    def get_local_entity_mask(self):
        return self.local(self.get_entity_mask(), False)

    def get_light_map(self):
        return self.get_visible()
    def get_local_light_map(self):
        return self.local(self.get_light_map(), True)

    def get_projectile_maps(self):
        return [np.zeros((48, 48), dtype=bool)]*4 #not implemented yet
    def get_local_projectile_maps(self):
        res = self.get_projectile_maps()
        for i in range(len(res)):
            res[i] = self.local(res[i], False)
        return res
    
    def get_visible(self):
        return self.light_map[self.player_level] > 0

    def local(self, arr: np.ndarray, val):
        x, y = tuple(self.player_position)
        return get_local_view(arr, x, y, val).astype(np.float32)
    
    def get_direction_mask(self):
        res = [0.0] * 5
        res[self.player_direction] = 1.0
        return res
    
    def to_super(self):
        base_fields = {f.name: getattr(self, f.name) for f in fields(CraftaxState)}
        return CraftaxState(**base_fields)
    