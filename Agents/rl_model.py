from MacroManagement.smartEnv import SmartEnv 

class RlAgent:
    def __init__(self, path_to_model="NA"):
        pass

    def get_action(self, state, scen):
        return NotImplementedError
    
    def add_exp(self, state, action, reward, next_state, done):
        return NotImplementedError