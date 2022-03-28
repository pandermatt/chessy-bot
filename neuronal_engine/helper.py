import random

import numpy as np

from neuronal_engine.optimizer import Adam, RMSProp

AIM = 0.99


def epsilon_greedy_policy(Qvalues, a, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        a = random.choice(a)
    else:
        a = a[np.argmax(Qvalues[a])]

    qvalue = Qvalues[a]

    return a, qvalue


def finished(reward, n, last=500):
    if len(reward) <= last:
        return False
    return np.mean(reward[(n - last):n]) >= AIM


def initialize_weights(layer_sizes, xavier):
    weights = []
    biases = []
    for idx in range(len(layer_sizes) - 1):
        if xavier:
            weights.append(np.random.randn(layer_sizes[idx + 1], layer_sizes[idx]) * np.sqrt(1 / (layer_sizes[idx])))
        else:
            weights.append(np.random.uniform(0, 1, (layer_sizes[idx + 1], layer_sizes[idx])))
            weights[idx] = np.divide(
                weights[idx],
                np.tile(np.sum(weights[idx], 1)[:, None], (1, layer_sizes[idx])), )

        biases.append(np.zeros((layer_sizes[idx + 1])))

    return weights, biases


def initialize_adam(weights, biases, beta_adam):
    adam_w = []
    adam_b = []
    for idx in range(len(weights)):
        adam_w.append(Adam(weights[idx], beta_adam))
        adam_b.append(Adam(biases[idx], beta_adam))

    return adam_w, adam_b


def initialize_rmsprop(weights, biases, gamma_rmsprop):
    rms_w = []
    rms_b = []
    for idx in range(len(weights)):
        rms_w.append(RMSProp(weights[idx], gamma_rmsprop))
        rms_b.append(RMSProp(biases[idx], gamma_rmsprop))

    return rms_w, rms_b
