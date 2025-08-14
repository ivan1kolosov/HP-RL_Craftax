from RlTraining.model_training import load_model

class RlAgent:
    def __init__(self, path):
        self.model = load_model(path)

    def get_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True) 
        return int(action)