import numpy as np

from agents.agent import Agent
from chess_env import ChessEnv
from neuronal_engine.neural_net import SarsaNn, QlearningNn, DoubleQlearningNn, DoubleSarsaNn, NeuralNet


class ChessyAgent(Agent):
    NN_KLASS = NeuralNet

    def __init__(self, N_episodes):
        super().__init__()
        self.N_episodes = N_episodes

    def _get_layer_sizes(self):
        board_state, X, allowed_actions = self.env.initialise_game()
        N_a = np.shape(allowed_actions)[0]  # TOTAL NUMBER OF POSSIBLE ACTIONS
        N_in = np.shape(X)[0]  # INPUT SIZE
        N_h1 = 200  # NUMBER OF HIDDEN NODES
        N_h2 = 200
        # INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK AND...
        # PLEASE CONSIDER USING A MASK OF ONE FOR THE ACTION MADE
        # AND ZERO OTHERWISE IF YOU ARE NOT USING VANILLA GRADIENT DESCENT...
        # WE SUGGEST A NETWORK WITH ONE HIDDEN LAYER WITH SIZE 200.

        return [N_in, N_h1, N_h2, N_a]

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgent(ChessyAgent):
    NAME = "SARSA"
    NN_KLASS = SarsaNn


class SarsaChessyAgentCustomReward(SarsaChessyAgent):
    NAME = 'SARSA with negative reward (-0.1)'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_step=-0.1)


class SarsaChessyAgentCustomReward2(SarsaChessyAgent):
    NAME = 'SARSA with negative reward (-0.2)'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_step=-0.1)


class QLearningChessyAgent(ChessyAgent):
    NAME = "Q-learning"
    NN_KLASS = QlearningNn


class DoubleQLearningChessyAgent(ChessyAgent):
    NAME = "Double-Q-learning"
    NN_KLASS = DoubleQlearningNn


class DoubleSarsaChessyAgent(ChessyAgent):
    NAME = "Double-SARSA-learning"
    NN_KLASS = DoubleSarsaNn
