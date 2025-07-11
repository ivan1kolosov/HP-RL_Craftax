class Scenario:
    def __init__(self):
        self.values = None

    def get_maps(self, state) -> tuple: #(torch.Tensor, torsch.Tensor)
        return NotImplementedError
    
    def is_action(self):
        return NotImplementedError
    
    def get_action(self):
        return NotImplementedError

class TacticAgent:
    def __init__(self):
        pass

    def get_scen(self, state):
        return NotImplementedError
    
    def get_reward(self, state, next_state) -> float:
        return NotImplementedError