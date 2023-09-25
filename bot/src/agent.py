import os
from stable_baselines3 import PPO

models_dir = "../../models/PPO-rl"


class Agent:
    def __init__(self):
        # Get the directory of agent.py
        script_dir = os.path.dirname(os.path.realpath(__file__))

        model_path = os.path.join(script_dir, "../../models/PPO-rl/280000.zip")
        self.model = PPO.load(model_path)

    def act(self, obs):
        action, _ = self.model.predict(obs)
        return action.tolist()
