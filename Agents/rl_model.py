class RlModel:
    def __init__(self, path):
        pass

class RlAgent:
    def __init__(self, path_to_model="NA"):
        self.model = RlModel(path_to_model)
        pass
    def get_action(self, state, scen):
        return NotImplementedError
    def add_exp(self, state, action, reward, next_state):
        return NotImplementedError