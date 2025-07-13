from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN

from Render.Renderer import CraftaxRenderer

class TestAgent:
    def __init__(self):
            
        pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN
        self.renderer = CraftaxRenderer(pixel_render_size=pixel_render_size)

    def get_action(self, env_state, scen):

        self.renderer.render(env_state, scen)

        while not self.renderer.is_quit_requested():

            action = self.renderer.get_action_from_keypress(env_state)

            if action is not None:

                self.renderer.render(env_state, scen)
                self.renderer.update()

                return action
        
            self.renderer.update()
