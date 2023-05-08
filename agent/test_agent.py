import torch
import torch.nn as nn

from config import args, device
from models.ppo_ac import Net
class TestAgent():
    """
    Agent for testing, trained on PPO (depending if complited or not, see thesis)
    """

    def __init__(self):
        self.net = Net().float().to(device)

    def select_action(self, state):
        """
        Selects an action for the given state using the policy learned by the PPO model

        Args:
            state (numpy.ndarray): The current state of the environment

        Returns:
            numpy.ndarray: The action to be taken in the current state
        """
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self):
        """
        Loads the trained parameters of the PPO model
        """
        self.net.load_state_dict(torch.load('saved_model_data/ppo_comp.pkl', map_location=torch.device('cpu')))