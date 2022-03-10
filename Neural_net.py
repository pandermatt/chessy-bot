# Import
import math
import pickle
import random

import numpy.matlib
from Adam import Adam

from chess_env import ChessEnv
from degree_freedom import (degree_freedom_king1, degree_freedom_king2,
                            degree_freedom_queen)
from generate_game import *

# input_layer_size = 10
# first_hidden_layer_size = 15
# output_layer_size = x
# input: layer_sizes = [input_layer_size, first_hidden_layer_size, ..., output_layer_size]


def epsilongreedy_policy(Qvalues, a, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        a = random.choice(a)
    else:
        a = a[np.argmax(Qvalues[a])]

    qvalue = np.zeros(len(Qvalues))
    qvalue[a] = Qvalues[a]

    return a, qvalue


class NeuralNet:
    def __init__(self, env, layer_sizes, xavier=False):
        self._name = "chessy bot"
        self.env = env
        self.epsilon_0 = 0.25  # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
        # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
        self.beta = 0.00005
        self.gamma = 0.9  # THE DISCOUNT FACTOR
        self.eta = 0.005  # THE LEARNING RATE
        self.beta_adam = 0.9

        # initialize weights
        self.weights = []
        self.biases = []

        for idx in range(len(layer_sizes) - 1):
            if xavier:
                self.weights.append(
                    np.random.randn(layer_sizes[idx + 1], layer_sizes[idx])
                    * np.sqrt(1 / (layer_sizes[idx]))
                )
            else:
                self.weights.append(
                    np.random.uniform(0, 1, (layer_sizes[idx + 1], layer_sizes[idx]))
                )
                self.weights[idx] = np.divide(
                    self.weights[idx],
                    np.tile(
                        np.sum(self.weights[idx], 1)[:, None], (1, layer_sizes[idx])
                    ),
                )

            self.biases.append(np.zeros((layer_sizes[idx + 1])))

        # initialize Adam
        self.adam_w = []
        self.adam_b = []

        for idx in range(len(self.weights)):
            self.adam_w.append(Adam(self.weights[idx], self.beta_adam))
            self.adam_b.append(Adam(self.biases[idx], self.beta_adam))

    def _forward_pass(self, X):
        x = [X]
        h = []
        for idx in range(len(self.weights)):
            h.append(np.dot(self.weights[idx], x[-1]) + self.biases[idx])
            x.append(1 / (1 + np.exp(-h[-1])))

        return x

    def _backprop(self, x, a, R, qvalue, Done, qvalue_next=0):
        dweights = []
        dbiases = []

        action_taken = np.zeros(len(x[-1]))
        action_taken[a] = 1

        x[-1] = action_taken * qvalue

        for idx in range(len(self.weights)):
            dweights.append(np.zeros(self.weights[idx].shape))
            dbiases.append(np.zeros(self.biases[idx].shape))

        for idx in range(len(self.weights)):
            if idx == 0:
                if Done == 1:
                    e_n = self._error_func_done(R, qvalue, action_taken)
                else:
                    e_n = self._error_func_not_done(
                        R, qvalue, qvalue_next, action_taken
                    )
                delta = x[-1] * (1 - x[-1]) * e_n
            else:
                delta = (
                    x[-(idx + 1)]
                    * (1 - x[-(idx + 1)])
                    * np.dot(np.transpose(self.weights[-idx]), delta)
                )
            dweights[-(idx + 1)] += np.outer(delta, x[-(idx + 2)])
            dbiases[-(idx + 1)] += delta

        for idx in range(len(self.weights)):
            self.weights[idx] += (
                self.eta * self.adam_w[idx].Compute(dweights[idx]) * x[idx]
            )
            self.biases[idx] += self.eta * self.adam_b[idx].Compute(dbiases[idx])

    def _error_func_done(self, R, qvalue, action_taken):
        return (R - qvalue) * action_taken

    def _error_func_not_done(self, R, qvalue, qvalue_next, action_taken):
        return (R + self.gamma * qvalue_next - qvalue) * action_taken

    def _call_epsilongreedy(self, param, a_next, epsilon_f):
        return epsilongreedy_policy(param, a_next, epsilon_f)

    def train(self, N_episodes, callback):
        R_save = np.zeros([N_episodes, 1])
        avg_reward = np.zeros(N_episodes)
        checkmate_save = np.zeros(N_episodes)
        N_moves_save = np.zeros([N_episodes, 1])
        avg_moves = np.zeros(N_episodes)

        intern_output_nr = 1000
        web_output_nr = 10

        for n in range(N_episodes):
            epsilon_f = self.epsilon_0 / (1 + self.beta * n)  # DECAYING EPSILON
            Done = 0  # SET DONE TO ZERO (BEGINNING OF THE EPISODE)
            move_counter = 1  # COUNTER FOR NUMBER OF ACTIONS

            S, X, allowed_a = self.env.initialise_game()

            if n % intern_output_nr == 0 and n > 0:
                print(f"Epoche ({n}/{N_episodes})")

                print(
                    "Chessy Agent, Average reward:",
                    np.mean(R_save[(n - intern_output_nr) : n]),
                    "Number of steps: ",
                    np.mean(N_moves_save[(n - intern_output_nr) : n]),
                )

            while Done == 0:
                a, _ = np.where(allowed_a == 1)
                x = self._forward_pass(X)
                a_agent, qvalue = epsilongreedy_policy(x[-1], a, epsilon_f)

                S_next, X_next, allowed_a_next, R, Done = self.env.one_step(a_agent)

                # THE EPISODE HAS ENDED, UPDATE... BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
                if Done == 1:

                    R_save[n] = np.copy(R)
                    N_moves_save[n] = np.copy(move_counter)
                    checkmate_save[n] = np.copy(R)

                    if n > 0:
                        avg_reward[n] = np.mean(R_save[0:n])
                        avg_moves[n] = np.mean(N_moves_save[0:n])
                    else:
                        avg_reward[n] = R_save[n]
                        avg_moves[n] = N_moves_save[n]

                    self._backprop(x, a, R, qvalue, Done)

                    break

                # IF THE EPISODE IS NOT OVER...
                else:
                    # Compute the delta
                    a_next, _ = np.where(allowed_a_next == 1)
                    x_next = self._forward_pass(X_next)
                    a_agent_next, qvalue_next = self._call_epsilongreedy(
                        x_next[-1], a_next, epsilon_f
                    )

                    self._backprop(x, a, R, qvalue, Done, qvalue_next)

                # NEXT STATE AND CO. BECOME ACTUAL STATE...
                S = np.copy(S_next)
                X = np.copy(X_next)
                allowed_a = np.copy(allowed_a_next)

                move_counter += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS

            if n % web_output_nr == 0:
                callback(
                    {
                        "board": self.calculate_location(S),
                        "epoche_string": f"{n}/{N_episodes}",
                        "average_reward": np.mean(R_save[(n - intern_output_nr) : n]),
                        "num_of_steps": np.mean(
                            N_moves_save[(n - intern_output_nr) : n]
                        ),
                        "percentage": f"{n / N_episodes * 100}%",
                        "percentage_label": f"{math.ceil(n / N_episodes * 100)}%",
                    }
                )

        print(
            f"{self._name}, Average reward: {np.mean(R_save)}\n"
            f"Number of steps: {np.mean(N_moves_save)}\n"
            f"Checkmates: {np.count_nonzero(checkmate_save > 0)}"
        )
        return self._name, avg_reward, avg_moves


class SARSA_NN(Neural_net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "SARSA BOT"

    def _call_epsilongreedy(self, param, a_next, epsilon_f):
        return epsilongreedy_policy(param, a_next, epsilon_f)


class QLEARNING_NN(Neural_net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "QLEARNING BOT"

    def _call_epsilongreedy(self, param, a_next, epsilon_f):
        return epsilongreedy_policy(param, a_next, 0)


def calculate_location(self, S):
    board = np.array(S)
    board_location = {
        # 1 = location of the King bK
        self.convert_location_to_letters(board, 1): "wK",
        # 2 = location of the Queen wQ
        self.convert_location_to_letters(board, 2): "wQ",
        # 3 = location fo the Enemy King wK
        self.convert_location_to_letters(board, 3): "bK",
    }
    return board_location


@staticmethod
def convert_location_to_letters(board, figure_id):
    match = np.where(board == figure_id)
    return f"{chr(97 + match[0][0])}{match[1][0] + 1}"
