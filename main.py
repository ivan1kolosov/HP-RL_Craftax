import random
import pygame

from craftax.craftax.craftax_state import EnvState as CraftaxState
from craftax.craftax.constants import Action

from Agents.rl_model import RlAgent
from Agents.testAgent import TestAgent

from tools import CraftaxEnv, Episode, SmartEpisode
from MacroManagement.smartEnv import SmartEnv, default_actions_pool, print_obs
from MacroManagement.scenario import Scenario

if __name__ == "__main__":

    seed = random.randint(1, 10**9)
    env = SmartEnv(default_actions_pool)
    agent = RlAgent("models/ppo_smartenv_model400k.zip")

    state, info = env.reset(seed)
    traj = Episode()

    done = False

    while not done:
        if (isinstance(agent, TestAgent)):
            action = agent.get_action(env.state.to_super(), env.scen)
            next_state, reward, truncated, done, info = env.step(default_actions_pool.index(Action(action)))
        else:
            action = agent.get_action(state)
            next_state, reward, truncated, done, info = env.step(action)
        print(reward)
        traj.add(state, action, reward)

        state = next_state
    traj.done = True
    env.env.episode.generate_video()
    pygame.quit()

    print("Sum reward: " + str(traj.get_sum_reward()))