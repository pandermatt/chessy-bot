# Import

import numpy as np
import numpy.matlib
import random
import matplotlib.pyplot as plt
from degree_freedom import degree_freedom_queen
from degree_freedom import degree_freedom_king1
from degree_freedom import degree_freedom_king2
from generate_game import *
from chess_env import ChessEnv
import pickle
from Adam import Adam


# input_layer_size = 10
# first_hidden_layer_size = 15
# output_layer_size = x
# input: layer_sizes = [input_layer_size, first_hidden_layer_size, ..., output_layer_size]

class Neural_net:
    def __init__(self, env, layer_sizes, xavier=False):
        self.env = env
        self.epsilon_0 = 0.25  # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
        # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
        self.beta = 0.00005
        self.gamma = 0.8  # THE DISCOUNT FACTOR
        self.eta = 0.004  # THE LEARNING RATE
        self.beta_adam = 0.9

        self.adam_w = []
        self.adam_b = []

        # initialize weights
        self.weights = []
        self.biases = []

        for idx in range(len(layer_sizes) - 1):
            if xavier:
                self.weights.append(
                    np.random.randn(layer_sizes[idx + 1], layer_sizes[idx]) * np.sqrt(1 / (layer_sizes[idx])))
            else:
                self.weights.append(np.random.uniform(
                    0, 1, (layer_sizes[idx + 1], layer_sizes[idx])))
                self.weights[idx] = np.divide(self.weights[idx],
                                              np.matlib.repmat(np.sum(self.weights[idx], 1)[:, None], 1,
                                                               layer_sizes[idx]))

            self.biases.append(np.zeros((layer_sizes[idx + 1])))

        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)

        # initialize Adam
        for idx in range(len(self.weights)):
            self.adam_w.append(Adam(self.weights[idx], self.beta_adam))
            self.adam_b.append(Adam(self.biases[idx], self.beta_adam))

    def _epsilongreedy_policy(self, Qvalues, a, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            choice = random.choice(a)
            a = choice
        else:
            choice = np.argmax(Qvalues[a])
            a = a[choice]

        qvalue = np.array(
            [Qvalues[a] if i == a else 0 for i in range(len(Qvalues))])
        return a, qvalue

    def _forward_pass(self, X):
        x = [X]
        h = []
        for idx in range(len(self.weights)):
            h.append(np.dot(self.weights[idx], x[-1]) + self.biases[idx])
            x.append(1 / (1 + np.exp(-h[-1])))

        return x

    def _backprop(self, x, R, qvalue, Done, qvalue_next=0):
        dweights = []
        dbiases = []
        R_vec = qvalue.copy()
        idx = np.argmax(qvalue)
        R_vec[idx] = R
        x[-1] = qvalue
        for idx in range(len(self.weights)):
            dweights.append(np.zeros(self.weights[idx].shape))
            dbiases.append(np.zeros(self.biases[idx].shape))

        for idx in range(len(self.weights)):
            if idx == 0:
                if Done == 1:
                    e_n = R - qvalue
                else:
                    e_n = R + self.gamma * qvalue_next - qvalue
                delta = x[-1] * (1 - x[-1]) * e_n
            else:
                delta = x[-(idx + 1)] * (1 - x[-(idx + 1)]) * \
                    np.dot(np.transpose(self.weights[-idx]), delta)
            dweights[-(idx + 1)] += np.outer(delta, x[-(idx + 2)])
            dbiases[-(idx + 1)] += delta

        for idx in range(len(self.weights)):
            self.weights[idx] += self.eta * \
                self.adam_w[idx].Compute(dweights[idx]) * x[idx]
            self.biases[idx] += self.eta * \
                self.adam_b[idx].Compute(dbiases[idx])

    def train(self, N_episodes):

        R_save = np.zeros([N_episodes, 1])
        N_moves_save = np.zeros([N_episodes, 1])

        interm_output_nr = 1000

        for n in range(N_episodes):
            epsilon_f = self.epsilon_0 / \
                (1 + self.beta * n)  # DECAYING EPSILON
            Done = 0  # SET DONE TO ZERO (BEGINNING OF THE EPISODE)
            move_counter = 1  # COUNTER FOR NUMBER OF ACTIONS

            S, X, allowed_a = self.env.initialise_game()

            if n % interm_output_nr == 0 and n > 0:
                print(f"Epoche ({n}/{N_episodes})")
                print('Chessy Agent, Average reward:', np.mean(R_save[(n-interm_output_nr):n]),
                      'Number of steps: ', np.mean(N_moves_save[(n-interm_output_nr):n]))

            while Done == 0:
                a, _ = np.where(allowed_a == 1)
                x = self._forward_pass(X)
                a_agent, qvalue = self._epsilongreedy_policy(
                    x[-1], a, epsilon_f)

                S_next, X_next, allowed_a_next, R, Done = self.env.one_step(
                    a_agent)

                # THE EPISODE HAS ENDED, UPDATE... BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
                if Done == 1:

                    R_save[n] = np.copy(R)
                    N_moves_save[n] = np.copy(move_counter)
                    self._backprop(x, R, qvalue, Done)

                    break

                # IF THE EPISODE IS NOT OVER...
                else:
                    # Compute the delta
                    a_next, _ = np.where(allowed_a_next == 1)
                    x_next = self._forward_pass(X_next)
                    a_agent_next, qvalue_next = self._epsilongreedy_policy(
                        x_next[-1], a_next, 0)

                    self._backprop(x, R, qvalue, Done, qvalue_next)

                # NEXT STATE AND CO. BECOME ACTUAL STATE...
                S = np.copy(S_next)
                X = np.copy(X_next)
                allowed_a = np.copy(allowed_a_next)

                move_counter += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS

        print('Chessy Agent, Average reward:', np.mean(R_save),
              'Number of steps: ', np.mean(N_moves_save))
