import numpy as np

from agents.agent import Agent
from neural_net import SARSA_NN, QLEARNING_NN


class ChessyAgent(Agent):
    def __init__(self, N_episodes):
        super().__init__()
        self.N_episodes = N_episodes

    def _get_layer_sizes(self):
        board_state, X, allowed_actions = self.env.initialise_game()
        N_a = np.shape(allowed_actions)[0]  # TOTAL NUMBER OF POSSIBLE ACTIONS
        N_in = np.shape(X)[0]  # INPUT SIZE
        N_h1 = 200  # NUMBER OF HIDDEN NODES
        # INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK AND...
        # PLEASE CONSIDER USING A MASK OF ONE FOR THE ACTION MADE
        # AND ZERO OTHERWISE IF YOU ARE NOT USING VANILLA GRADIENT DESCENT...
        # WE SUGGEST A NETWORK WITH ONE HIDDEN LAYER WITH SIZE 200.

        return [N_in, N_h1, N_a]

    def run(self, callback=lambda *args: None):
        pass


class SarsaChessyAgent(ChessyAgent):
    NAME = "SARSA ChessyAgent"

    def run(self, callback=lambda *args: None):
        nn = SARSA_NN(self.env, self._get_layer_sizes(), xavier=True)
        return nn.train(self.N_episodes, callback)


class QLearningChessyAgent(ChessyAgent):
    NAME = "Q-learning ChessyAgent"

    def run(self, callback=lambda *args: None):
        nn = QLEARNING_NN(self.env, self._get_layer_sizes(), xavier=True)
        return nn.train(self.N_episodes, callback)
