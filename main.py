from craftax.craftax_env import make_craftax_env_from_name

from Agents.rl_model import RlAgent
from Agents.tacticAgent import TacticAgent

if __name__ == "__main__":
    rl_agent = RlAgent()
    tactic_agent = TacticAgent()
    done = False
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env.reset() #Set seed here
    while not done:
        scen =  tactic_agent.get_scen(state)
        action = scen.get_action() if scen.is_action() else rl_agent.get_action(state, scen)
        obs, next_state, basic_reward, done, info = env.step(action)
        if (scen.is_action()):
            smart_reward = scen.get_reward(state, next_state)
            rl_agent.add_exp(state, action, (basic_reward, smart_reward), next_state, done)
        state = next_state