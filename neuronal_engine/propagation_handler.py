import random

import numpy as np


class PropagationHandler:
    def __init__(self, nn):
        self.nn = nn

    def forward_pass(self, X):
        return self._execute_forwardpass(X, self.nn.weights, self.nn.biases)

    def _execute_forwardpass(self, X, weights, biases):
        x = [X]
        h = []
        for idx in range(len(weights)):
            h.append(np.dot(weights[idx], x[-1]) + biases[idx])

            if self.nn.activation == 'relu':
                x.append(np.maximum(0, h[-1]))

            elif self.nn.activation == 'sigmoid':
                x.append(1/(1+np.exp(-h[-1])))

            elif self.nn.activation == 'leakyrelu':
                x.append(np.maximum(self.nn.leaky_constant*h[-1], h[-1]))

        return x, h

    def backprop(self, x, h, a, R, qvalue, Done, qvalue_next=0):
        self._execute_backprop(x, h, a, R, qvalue, Done, qvalue_next, self.nn.weights, self.nn.biases)

    def _execute_backprop(self, x, h, a, R, qvalue, Done, qvalue_next, weights, biases):
        dweights = []
        dbiases = []

        for idx in range(len(weights)):
            dweights.append(np.zeros(weights[idx].shape))
            dbiases.append(np.zeros(biases[idx].shape))

        action_taken = np.zeros(len(x[-1]))
        action_taken[a] = 1

        if self.nn.activation == 'relu':
            dweights, dbiases = self.backprop_relu(x, h, R, qvalue, Done, qvalue_next, weights, dweights, dbiases, action_taken)

        elif self.nn.activation == 'sigmoid':
            dweights, dbiases = self.backprop_sigmoid(x, R, qvalue, Done, qvalue_next, weights, dweights, dbiases, action_taken)

        elif self.nn.activation == 'leakyrelu':
            dweights, dbiases = self.backprop_leakyrelu(x, h, R, qvalue, Done, qvalue_next, weights, dweights, dbiases, action_taken)

        for idx in range(len(weights)):
            if self.nn.optimizer is None:
                weights[idx] += self.nn.eta * dweights[idx] * x[idx]
                biases[idx] += self.nn.eta * dbiases[idx]

            elif self.nn.optimizer == 'adam':
                weights[idx] += self.nn.adam_w[idx].Compute(dweights[idx], self.nn.adam_eta)
                biases[idx] += self.nn.adam_b[idx].Compute(dbiases[idx], self.nn.adam_eta)

            elif self.nn.optimizer == 'rmsprop':
                weights[idx] += self.nn.rms_eta * self.nn.rms_w[idx].Compute(dweights[idx])
                biases[idx] += self.nn.rms_eta * self.nn.rms_b[idx].Compute(dbiases[idx])

    def backprop_relu(self, x, h, R, qvalue, Done, qvalue_next, weights, dweights, dbiases, action_taken):
        for idx in range(len(weights)):
            if idx == 0:
                if Done == 1:
                    e_n = self._error_func_done(R, qvalue) * action_taken
                else:
                    e_n = self._error_func_not_done(R, qvalue, qvalue_next) * action_taken

                delta = np.heaviside(h[-1], 0) * e_n
            else:
                delta = np.heaviside(h[-(idx+1)], 0) * np.dot(np.transpose(weights[-idx]), delta)
            dweights[-(idx + 1)] = np.outer(delta, x[-(idx + 2)])
            dbiases[-(idx + 1)] = delta

        return dweights, dbiases

    def backprop_sigmoid(self, x, R, qvalue, Done, qvalue_next, weights, dweights, dbiases, action_taken):
        for idx in range(len(weights)):
            if idx == 0:
                if Done == 1:
                    e_n = self._error_func_done(R, qvalue) * action_taken
                else:
                    e_n = self._error_func_not_done(R, qvalue, qvalue_next) * action_taken

                delta = x[-1] * (1 - x[-1]) * e_n
            else:
                delta = (x[-(idx + 1)] * (1 - x[-(idx + 1)]) * np.dot(np.transpose(weights[-idx]), delta))
            dweights[-(idx + 1)] += np.outer(delta, x[-(idx + 2)])
            dbiases[-(idx + 1)] += delta

        return dweights, dbiases

    def backprop_leakyrelu(self, x, h, R, qvalue, Done, qvalue_next, weights, dweights, dbiases, action_taken):
        dh = []
        for _h in h:
            dh_next = np.ones_like(_h)
            dh_next[_h < 0] = self.nn.leaky_constant
            dh.append(dh_next)

        for idx in range(len(weights)):
            if idx == 0:
                if Done == 1:
                    e_n = self._error_func_done(R, qvalue) * action_taken
                else:
                    e_n = self._error_func_not_done(R, qvalue, qvalue_next) * action_taken

                delta = dh[-1] * e_n
            else:
                delta = dh[-(idx+1)] * np.dot(np.transpose(weights[-idx]), delta)
            dweights[-(idx + 1)] = np.outer(delta, x[-(idx + 2)])
            dbiases[-(idx + 1)] = delta

        return dweights, dbiases

    @staticmethod
    def _error_func_done(R, qvalue):
        return R - qvalue

    def _error_func_not_done(self, R, qvalue, qvalue_next):
        return R + self.nn.gamma * qvalue_next - qvalue


class DoublePropagationHandler(PropagationHandler):
    def forward_pass(self, X):
        if self.nn.counter == 0:
            self.nn.counter += 1
            self.nn.choice = random.randint(0, 1)
            if self.nn.choice:
                return self._execute_forwardpass(X, self.nn.weights2, self.nn.biases2)
            else:
                return self._execute_forwardpass(X, self.nn.weights, self.nn.biases)
        else:
            self.nn.counter = 0
            if self.nn.choice:
                return self._execute_forwardpass(X, self.nn.weights, self.nn.biases)
            else:
                return self._execute_forwardpass(X, self.nn.weights2, self.nn.biases2)

    def backprop(self, x, h, a, R, qvalue, Done, qvalue_next=0):
        if self.nn.choice:
            self._execute_backprop(x, h, a, R, qvalue, Done, qvalue_next, self.nn.weights2, self.nn.biases2)
        else:
            self._execute_backprop(x, h, a, R, qvalue, Done, qvalue_next, self.nn.weights, self.nn.biases)
