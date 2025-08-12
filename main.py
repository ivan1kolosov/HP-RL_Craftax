import random

import pygame

from Agents.rl_model import RlAgent
from MacroManagement.scenario import Scenario
from Agents.testAgent import TestAgent

from tools import CraftaxEnv, Episode

if __name__ == "__main__":
    
    ep = Episode()
    ep.load("episode_1755003450.pbz2")
    ep.generate_video()
    exit(0)

    seed = random.randint(1, 10**9)
    env = CraftaxEnv()
    agent = TestAgent()

    state = env.reset(seed)
    traj = Episode()

    done = False

    while not done:

        scen = Scenario(state)

        action = agent.get_action(state, scen)
        
        next_state, reward, done, info = env.step(action, print_achievements=True)

        if not scen.is_action():
            smart_reward = scen.get_reward(state, next_state)
            #agent.add_exp(state, action, (reward, smart_reward), next_state, done)

        traj.add(state, action, reward)

        state = next_state
    traj.done = True

    pygame.quit()
    traj.save()

    print("Sum reward: " + str(traj.get_sum_reward()))