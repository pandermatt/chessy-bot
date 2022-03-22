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
            x.append(1 / (1 + np.exp(-h[-1])))

        return x

    def backprop(self, x, a, R, qvalue, Done, qvalue_next=0):
        self._execute_backprop(
            x,
            a,
            R,
            qvalue,
            Done,
            qvalue_next,
            self.nn.weights,
            self.nn.biases,
            self.nn.adam_w,
            self.nn.adam_b,
        )

    def _execute_backprop(
        self, x, a, R, qvalue, Done, qvalue_next, weights, biases, adam_w, adam_b
    ):
        dweights = []
        dbiases = []

        action_taken = np.zeros(len(x[-1]))
        action_taken[a] = 1

        x[-1] = action_taken * qvalue

        for idx in range(len(weights)):
            dweights.append(np.zeros(weights[idx].shape))
            dbiases.append(np.zeros(biases[idx].shape))

        for idx in range(len(weights)):
            if idx == 0:
                if Done == 1:
                    e_n = self._error_func_done(R, qvalue, action_taken)
                else:
                    e_n = self._error_func_not_done(
                        R, qvalue, qvalue_next, action_taken
                    )
                delta = x[-1] * (1 - x[-1]) * e_n
            else:
                # TODO: delta falsch?
                delta = (
                    x[-(idx + 1)]
                    * (1 - x[-(idx + 1)])
                    * np.dot(np.transpose(weights[-idx]), delta)
                )
            dweights[-(idx + 1)] += np.outer(delta, x[-(idx + 2)])
            dbiases[-(idx + 1)] += delta

        for idx in range(len(weights)):
            weights[idx] += self.nn.eta * adam_w[idx].Compute(dweights[idx]) * x[idx]
            biases[idx] += self.nn.eta * adam_b[idx].Compute(dbiases[idx])

    @staticmethod
    def _error_func_done(R, qvalue, action_taken):
        return (R - qvalue) * action_taken

    def _error_func_not_done(self, R, qvalue, qvalue_next, action_taken):
        return (R + self.nn.gamma * qvalue_next - qvalue) * action_taken


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

    def backprop(self, x, a, R, qvalue, Done, qvalue_next=0):
        if self.nn.choice:
            self._execute_backprop(
                x,
                a,
                R,
                qvalue,
                Done,
                qvalue_next,
                self.nn.weights2,
                self.nn.biases2,
                self.nn.adam_w2,
                self.nn.adam_b2,
            )
        else:
            self._execute_backprop(
                x,
                a,
                R,
                qvalue,
                Done,
                qvalue_next,
                self.nn.weights,
                self.nn.biases,
                self.nn.adam_w,
                self.nn.adam_b,
            )
