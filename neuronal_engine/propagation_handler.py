import random

import numpy as np


class PropagationHandler:
    def __init__(self, nn):
        self.nn = nn

    def forward_pass(self, X):
        return self._execute_forwardpass(X, self.nn.weights, self.nn.biases)

    @staticmethod
    def _execute_forwardpass(X, weights, biases):
        x = [X]
        h = []
        for idx in range(len(weights)):
            h.append(np.dot(weights[idx], x[-1]) + biases[idx])
            x.append(np.maximum(0.1*h[-1], h[-1]))

        return x, h

    def backprop(self, x, h, a, R, qvalue, Done, qvalue_next=0):
        self._execute_backprop(x, h, a, R, qvalue, Done, qvalue_next)

    def _execute_backprop(self, x, h, a, R, qvalue, Done, qvalue_next):
        weights = self.nn.weights
        biases = self.nn.biases

        adam_w = self.nn.adam_w
        adam_b = self.nn.adam_b

        rms_w = self.nn.rms_w
        rms_b = self.nn.rms_b

        dweights = []
        dbiases = []

        for idx in range(len(weights)):
            dweights.append(np.zeros(weights[idx].shape))
            dbiases.append(np.zeros(biases[idx].shape))

        for idx in range(len(weights)):
            if idx == 0:
                if Done == 1:
                    e_n = self._error_func_done(R, qvalue)
                else:
                    e_n = self._error_func_not_done(R, qvalue, qvalue_next)
                dweights[-1][a, :] = e_n * x[-2]
                dbiases[-1][a] = self.nn.eta * e_n
            else:
                dweights[-(idx + 1)] = np.outer(e_n * weights[-1][a, :] * np.heaviside(h[-(idx + 1)], 0), x[-(idx + 2)])
                dbiases[-(idx + 1)] = self.nn.eta * e_n * weights[-1][a, :] * np.heaviside(h[-(idx + 1)], 0)

        for idx in range(len(weights)):
            #self.nn.weights[idx] += self.nn.eta * dweights[idx]
            #self.nn.biases[idx] += self.nn.eta * dbiases[idx]

            #self.nn.weights[idx] += adam_w[idx].Compute(dweights[idx], self.nn.adam_eta)
            #self.nn.biases[idx] += adam_b[idx].Compute(dbiases[idx], self.nn.adam_eta)

            self.nn.weights[idx] += self.nn.rms_eta * rms_w[idx].Compute(dweights[idx])
            self.nn.biases[idx] += self.nn.rms_eta * rms_b[idx].Compute(dbiases[idx])


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
            self._execute_backprop(x, h, a, R, qvalue, Done, qvalue_next, self.nn.weights2, self.nn.biases2,
                                   self.nn.adam_w2, self.nn.adam_b2)
        else:
            self._execute_backprop(x, h, a, R, qvalue, Done, qvalue_next, self.nn.weights, self.nn.biases,
                                   self.nn.adam_w, self.nn.adam_b)
