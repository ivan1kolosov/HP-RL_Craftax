import pygame

import jax
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name

from Agents.rl_model import RlAgent
from Agents.tacticAgent import TacticAgent
from Agents.testAgent import TestAgent

from tools import save_traj_history, print_new_achievements

if __name__ == "__main__":

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    rng = jax.random.PRNGKey(np.random.randint(2**31))
    step_fn = jax.jit(env.step)

    rl_agent = RlAgent()
    tactic_agent = TacticAgent()
    test_agent = TestAgent(env)

    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)

    traj_history = {"state": [env_state], "action": [], "reward": [], "done": []}

    clock = pygame.time.Clock()

    done = False

    while not done:

        scen = tactic_agent.get_scen(env_state)
        
        action = test_agent.get_action(env_state, scen)

        """ or
        if scen.is_action():
            action = scen.get_action() 
        else:
            action = rl_agent.get_action(state, scen)

        """

        rng, _rng = jax.random.split(rng)
        old_achievements = env_state.achievements
        
        obs, next_env_state, reward, done, info = step_fn(
            _rng, env_state, action, env_params
        )
        
        new_achievements = next_env_state.achievements
        print_new_achievements(old_achievements, new_achievements)

        if reward > 0.8:
            print(f"Reward: {reward}\n")

        if (scen.is_action()):
            smart_reward = scen.get_reward(env_state, next_env_state)
            rl_agent.add_exp(env_state, action, (reward, smart_reward), next_env_state, done)

        traj_history["state"].append(next_env_state)
        traj_history["action"].append(action)
        traj_history["reward"].append(reward)
        traj_history["done"].append(done)

        env_state = next_env_state
        
    pygame.quit()
    save_traj_history(traj_history)