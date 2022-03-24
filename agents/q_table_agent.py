import random

import numpy as np

from agents.agent import Agent
from neuronal_engine.helper import finished
from util.logger import log


class QTableAgent(Agent):
    NAME = 'Q Table Agent'

    def __init__(self, N_episodes):
        super().__init__()
        self.N_episodes = N_episodes
        self._name = self.NAME

    def generate_qvalues(self, q_table, a, X):
        key = tuple(X)

        if key not in q_table:
            a_transp = np.transpose(a)[0]
            values = np.random.rand(32)
            values = [values[i] if a_transp[i] == 1 else 0 for i in range(len(values))]
            q_table[key] = values

        return q_table, q_table[key]

    def epsilon_greedy_policy(self, Qvalues, a, epsilon):
        rand_value = np.random.uniform(0, 1)
        if rand_value < epsilon:
            a = random.choice(a)
        else:
            a = np.argmax(Qvalues)

        qvalue = Qvalues[a]
        return a, qvalue

    def run(self, callback=lambda *args: None):
        epsilon_0 = 0.2  # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
        beta = 0.00005  # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
        gamma = 0.85  # THE DISCOUNT FACTOR
        eta = 0.0035  # THE LEARNING RATE

        N_episodes = self.N_episodes

        N_moves_save = []
        R_save = []

        Qtable = {}

        for n in range(N_episodes):
            epsilon_f = epsilon_0 / (1 + beta * n)
            move_counter = 1

            S, X, allowed_a = self.env.initialise_game()
            callback(self, S, n, N_episodes, R_save, N_moves_save)
            if finished(R_save, n):
                log.info(f'Finished with {n}')
                callback(self, S, n, N_episodes, R_save, N_moves_save)
                break

            while True:
                a, _ = np.where(allowed_a == 1)
                Qtable, Qvalues = self.generate_qvalues(Qtable, allowed_a, X)
                a_agent, qvalue = self.epsilon_greedy_policy(Qvalues, a, epsilon_f)

                S_next, X_next, allowed_a_next, R, done = self.env.one_step(a_agent)

                if done == 1:
                    delta = R - Qtable[tuple(X)][a_agent]

                    Qtable[tuple(X)][a_agent] = Qtable[tuple(X)][a_agent] + eta * delta

                    R_save.append(R)
                    N_moves_save.append(move_counter)
                    break
                else:
                    _, qvalues_next = self.generate_qvalues(Qtable, allowed_a_next, X_next)
                    _, Qvalue_best_action_next = self.epsilon_greedy_policy(qvalues_next, allowed_a_next, 0)
                    delta = R + gamma * Qvalue_best_action_next - qvalue

                    Qtable[tuple(X)][a_agent] = Qtable[tuple(X)][a_agent] + eta * delta

                S = np.copy(S_next)
                X = np.copy(X_next)
                allowed_a = np.copy(allowed_a_next)

                move_counter += 1

        return self._name, R_save, N_moves_save
