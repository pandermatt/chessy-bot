# Import

from generate_game import *
from neuronal_engine.helper import initialize_weights, epsilon_greedy_policy
from neuronal_engine.propagation_handler import PropagationHandler, DoublePropagationHandler
from util.logger import log

SARSA = 'sarsa'
QLEARNING = 'q-learning'
MAX_STEPS_ALLOWED = 1000


class NeuralNet:
    def __init__(self, env, layer_sizes, xavier=False):
        self._name = "chessy bot"
        self.env = env
        self.layer_sizes = layer_sizes
        self.xavier = xavier
        self.epsilon_0 = 0.25  # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
        # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
        self.beta = 0.00005
        self.gamma = 0.9  # THE DISCOUNT FACTOR
        self.eta = 0.005  # THE LEARNING RATE
        self.beta_adam = 0.9

        # initialize weights
        self.weights = []
        self.biases = []

        # initialize Adam
        self.adam_w = []
        self.adam_b = []

        self.prop = PropagationHandler(self)

        initialize_weights(self.layer_sizes, self.weights, self.biases, self.adam_w, self.adam_b, self.beta_adam,
                           self.xavier)

    def train(self, N_episodes, callback):
        R_save = np.zeros([N_episodes, 1])
        avg_reward = np.zeros(N_episodes)
        checkmate_save = np.zeros(N_episodes)
        N_moves_save = np.zeros([N_episodes, 1])
        avg_moves = np.zeros(N_episodes)

        for n in range(N_episodes):
            epsilon_f = self.epsilon_0 / (1 + self.beta * n)  # DECAYING EPSILON
            move_counter = 1

            S, X, allowed_a = self.env.initialise_game()

            a_agent_next, qvalue_next = None, None

            for i in range(MAX_STEPS_ALLOWED):
                a, _ = np.where(allowed_a == 1)
                x = self.prop.forward_pass(X)

                if self.type == SARSA and a_agent_next is not None and qvalue_next is not None:
                    a_agent, qvalue = a_agent_next, qvalue_next
                else:
                    a_agent, qvalue = epsilon_greedy_policy(x[-1], a, epsilon_f)

                S_next, X_next, allowed_a_next, R, Done = self.env.one_step(a_agent, i)

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

                    self.prop.backprop(x, a, R, qvalue, Done)

                    break
                else:
                    a_next, _ = np.where(allowed_a_next == 1)
                    x_next = self.prop.forward_pass(X_next)
                    a_agent_next, qvalue_next = epsilon_greedy_policy(
                        x_next[-1], a_next,
                        epsilon_f if self.type == SARSA else 0)
                    self.prop.backprop(x, a_agent, R, qvalue, Done, qvalue_next)

                S = np.copy(S_next)
                X = np.copy(X_next)
                allowed_a = np.copy(allowed_a_next)

                move_counter += 1
            else:
                log.error(f"Invalid Epoche. Epoche was longer than {MAX_STEPS_ALLOWED}")

            callback(self, S, n, N_episodes, R_save, N_moves_save)

        log.info(f"{self._name}, Average reward: {np.mean(R_save)}")
        log.info(f"Number of steps: {np.mean(N_moves_save)}")
        log.info(f"Checkmates: {np.count_nonzero(checkmate_save > 0)}")
        return self._name, avg_reward, avg_moves


class SarsaNn(NeuralNet):
    type = SARSA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "SARSA BOT"


class QlearningNn(NeuralNet):
    type = QLEARNING

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "QLEARNING BOT"


class DoubleQlearningNn(NeuralNet):
    type = QLEARNING

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "DOUBLE-QLEARNING BOT"
        self.weights2 = []
        self.biases2 = []
        self.adam_w2 = []
        self.adam_b2 = []

        self.choice = 0
        self.counter = 0

        self.prop = DoublePropagationHandler(self)

        initialize_weights(self.layer_sizes, self.weights2, self.biases2, self.adam_w2, self.adam_b2, self.beta_adam,
                           self.xavier)


class DoubleSarsaNn(DoubleQlearningNn):
    type = SARSA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "DOUBLE-SARSA BOT"
