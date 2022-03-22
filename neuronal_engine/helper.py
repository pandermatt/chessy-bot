import random

import numpy as np

from neuronal_engine.adam import Adam


def epsilon_greedy_policy(Qvalues, a, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        a = random.choice(a)
    else:
        a = a[np.argmax(Qvalues[a])]

    qvalue = np.zeros(len(Qvalues))
    qvalue[a] = Qvalues[a]

    return a, qvalue


def initialize_weights(layer_sizes, weights, biases, adam_w, adam_b, beta_adam,
                       xavier):
    for idx in range(len(layer_sizes) - 1):
        if xavier:
            weights.append(
                np.random.randn(layer_sizes[idx + 1], layer_sizes[idx]) *
                np.sqrt(1 / (layer_sizes[idx])))
        else:
            weights.append(
                np.random.uniform(0, 1,
                                  (layer_sizes[idx + 1], layer_sizes[idx])))
            weights[idx] = np.divide(
                weights[idx],
                np.tile(
                    np.sum(weights[idx], 1)[:, None], (1, layer_sizes[idx])),
            )

        biases.append(np.zeros((layer_sizes[idx + 1])))

    for idx in range(len(weights)):
        adam_w.append(Adam(weights[idx], beta_adam))
        adam_b.append(Adam(biases[idx], beta_adam))
