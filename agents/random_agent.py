import numpy as np

from agents.agent import Agent


class RandomAgent(Agent):
    NAME = 'Random Agent'

    def run(self, callback=lambda *args: None):
        """
        Perform N Episodes Making Random Actions
        And Compute The Average Reward And Number Of Moves
        """
        N_episodes = 1000

        R_save_random = np.zeros([N_episodes, 1])
        N_moves_save_random = np.zeros([N_episodes, 1])
        checkmate_save = np.zeros(N_episodes)


        for n in range(N_episodes):
            board_state, X, allowed_actions = self.env.initialise_game()
            done = 0
            move_counter = 1

            callback(self, board_state, n, N_episodes, R_save_random, N_moves_save_random)
            while done == 0:
                a, _ = np.where(allowed_actions == 1)
                current_action = np.random.permutation(a)[0]

                board_state, X, allowed_actions, R, done = self.env.one_step(current_action)

                if done:
                    R_save_random[n] = np.copy(R)
                    N_moves_save_random[n] = np.copy(move_counter)
                    break

                move_counter += 1

        # AS YOU SEE, THE PERFORMANCE OF A RANDOM AGENT ARE NOT GREAT,
        # SINCE THE MAJORITY OF THE POSITIONS END WITH A DRAW
        # (THE ENEMY KING IS NOT IN CHECK AND CAN'T MOVE)

        print(
            "Random Agent, Average reward:",
            np.mean(R_save_random),
            "Number of steps: ",
            np.mean(N_moves_save_random),
            "Number of checkmates: ",
            np.count_nonzero(checkmate_save > 0)
        )
