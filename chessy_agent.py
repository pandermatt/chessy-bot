import numpy as np

from chess_env import ChessEnv
from neural_net import NeuralNet


class ChessyAgent:
    def __init__(self, board_size=4):
        self.env = ChessEnv(board_size)

    def run(self, callback):
        # INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK AND...
        # PLEASE CONSIDER USING A MASK OF ONE FOR THE ACTION MADE
        # AND ZERO OTHERWISE IF YOU ARE NOT USING VANILLA GRADIENT DESCENT...
        # WE SUGGEST A NETWORK WITH ONE HIDDEN LAYER WITH SIZE 200.

        board_state, X, allowed_actions = self.env.initialise_game()
        N_a = np.shape(allowed_actions)[0]  # TOTAL NUMBER OF POSSIBLE ACTIONS
        N_in = np.shape(X)[0]  # INPUT SIZE
        N_h1 = 200  # NUMBER OF HIDDEN NODES

        # INITALISE YOUR NEURAL NETWORK...
        # HYPERPARAMETERS SUGGESTED (FOR A GRID SIZE OF 4)
        nn = NeuralNet(self.env, [N_in, N_h1, N_a], xavier=True)

        N_episodes = 300000  # THE NUMBER OF GAMES TO BE PLAYED

        nn.train(N_episodes, callback)
