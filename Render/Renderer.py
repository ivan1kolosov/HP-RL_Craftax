import pygame

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
)
from craftax.craftax.renderer import render_craftax_pixels

KEY_MAPPING = {
    pygame.K_q: Action.NOOP,
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
    pygame.K_SPACE: Action.DO,
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_DIAMOND_PICKAXE,
    pygame.K_5: Action.MAKE_WOOD_SWORD,
    pygame.K_6: Action.MAKE_STONE_SWORD,
    pygame.K_7: Action.MAKE_IRON_SWORD,
    pygame.K_8: Action.MAKE_DIAMOND_SWORD,
    pygame.K_t: Action.PLACE_TABLE,
    pygame.K_TAB: Action.SLEEP,
    pygame.K_r: Action.PLACE_STONE,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_e: Action.REST,
    pygame.K_COMMA: Action.ASCEND,
    pygame.K_PERIOD: Action.DESCEND,
    pygame.K_y: Action.MAKE_IRON_ARMOUR,
    pygame.K_u: Action.MAKE_DIAMOND_ARMOUR,
    pygame.K_i: Action.SHOOT_ARROW,
    pygame.K_o: Action.MAKE_ARROW,
    pygame.K_g: Action.CAST_FIREBALL,
    pygame.K_h: Action.CAST_ICEBALL,
    pygame.K_j: Action.PLACE_TORCH,
    pygame.K_z: Action.DRINK_POTION_RED,
    pygame.K_x: Action.DRINK_POTION_GREEN,
    pygame.K_c: Action.DRINK_POTION_BLUE,
    pygame.K_v: Action.DRINK_POTION_PINK,
    pygame.K_b: Action.DRINK_POTION_CYAN,
    pygame.K_n: Action.DRINK_POTION_YELLOW,
    pygame.K_m: Action.READ_BOOK,
    pygame.K_k: Action.ENCHANT_SWORD,
    pygame.K_l: Action.ENCHANT_ARMOUR,
    pygame.K_LEFTBRACKET: Action.MAKE_TORCH,
    pygame.K_RIGHTBRACKET: Action.LEVEL_UP_DEXTERITY,
    pygame.K_MINUS: Action.LEVEL_UP_STRENGTH,
    pygame.K_EQUALS: Action.LEVEL_UP_INTELLIGENCE,
    pygame.K_SEMICOLON: Action.ENCHANT_BOW,
}

size_tactic_agent = 500

class CraftaxRenderer:
    def __init__(self, pixel_render_size=4):
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size + size_tactic_agent,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = jax.jit(render_craftax_pixels, static_argnums=(1,))


    def font(self, task, values):
        font1 = pygame.font.Font(None, 45)
        font2 = pygame.font.Font(None, 30)
        text_color = (255, 255, 255)
        message = "Task: " + task.type.value
        for k, v in values.items():
            message += '\n' + str(k) + ': ' + str(v)
        lines = message.split('\n')
        rendered_lines = [font1.render(lines[0], True, (255, 70, 70))]
        rendered_lines += [font2.render(line, True, text_color) for line in lines[1:]]
        return rendered_lines
    

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()

    def render(self, env_state, scen):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (size_tactic_agent, 0))

        text_area_width = size_tactic_agent
        text_area_height = self.screen_surface.get_height()
        gray_color = (39, 39, 39)

        pygame.draw.rect(self.screen_surface, gray_color, (0, 0, text_area_width, text_area_height))
        rendered_lines = self.font(scen.task, scen.values)

        y_offset = 0
        for line in rendered_lines:
            self.screen_surface.blit(line, (0, y_offset))
            y_offset += line.get_height() + 5

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state):
        if state.is_sleeping or state.is_resting:
            return Action.NOOP.value

        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    return KEY_MAPPING[event.key].value

        return None
