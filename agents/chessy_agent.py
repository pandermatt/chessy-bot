import matplotlib.pyplot as plt
import numpy as np

from agents.agent import Agent
from neural_net import SARSA_NN, QLEARNING_NN


class ChessyAgent(Agent):
    NAME = "ChessyAgent"

    def run(self, callback=lambda *args: None):
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

        sarsa = SARSA_NN(self.env, [N_in, N_h1, N_a], xavier=True)
        qlearning = QLEARNING_NN(self.env, [N_in, N_h1, N_a], xavier=True)

        N_episodes = 1000  # THE NUMBER OF GAMES TO BE PLAYED

        name1, reward1, moves1 = sarsa.train(N_episodes, callback)
        name2, reward2, moves2 = qlearning.train(N_episodes, callback)

        self.print_stats(N_episodes, name1, reward1, moves1, name2, reward2, moves2)

    def print_stats(self, n_episodes, name1, r_save1, step_save1, name2, r_save2, step_save2):
        episodes = range(n_episodes)
        plt.subplots_adjust(wspace=1, hspace=0.3)

        plt.subplot(2, 1, 1)
        plt.plot(episodes, r_save1, label=name1)
        plt.plot(episodes, r_save2, label=name2)
        plt.title(f"Avg. Rewards")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(episodes, step_save1, label=name1)
        plt.plot(episodes, step_save2, label=name2)
        plt.title(f"Avg. Steps ")
        plt.legend()

        plt.show()
