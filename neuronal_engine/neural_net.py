# Import

from generate_game import *
from neuronal_engine.helper import initialize_weights, epsilon_greedy_policy, finished, initialize_adam, \
    initialize_rmsprop
from neuronal_engine.propagation_handler import PropagationHandler, DoublePropagationHandler
from util.logger import log

SARSA = 'sarsa'
QLEARNING = 'q-learning'
MAX_STEPS_ALLOWED = 1000


class NeuralNet:
    def __init__(self, agent, xavier=False):
        self.agent = agent
        self.env = agent.env
        self.layer_sizes = agent._get_layer_sizes()
        self.xavier = xavier
        self.optimizer = None
        self.activation = 'relu'
        self.epsilon_0 = 0.25  # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
        # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
        self.beta = 0.00005
        self.gamma = 0.9  # THE DISCOUNT FACTOR
        self.eta = 0.005  # THE LEARNING RATE

        # initialize weights
        self.weights = []
        self.biases = []

        self.prop = PropagationHandler(self)

        self.weights, self.biases = initialize_weights(self.layer_sizes, self.xavier)


    def train(self, N_episodes, callback):
        checkmate_save = np.zeros(N_episodes)
        N_moves_save = []
        R_save = []

        for n in range(N_episodes):
            epsilon_f = self.epsilon_0 / (1 + self.beta * n)  # DECAYING EPSILON
            move_counter = 1

            S, X, allowed_a = self.env.initialise_game()

            a_agent_next, qvalue_next = None, None

            callback(self.agent, S, n, N_episodes, R_save, N_moves_save)
            if finished(R_save, n, self.env.reward_checkmate):
                log.info(f'Finished with {n}')
                callback(self.agent, S, n, N_episodes, R_save, N_moves_save)
                break

            for i in range(MAX_STEPS_ALLOWED):
                a, _ = np.where(allowed_a == 1)
                x, h = self.prop.forward_pass(X)

                if self.type == SARSA and a_agent_next is not None and qvalue_next is not None:
                    a_agent, qvalue = a_agent_next, qvalue_next
                else:
                    a_agent, qvalue = epsilon_greedy_policy(x[-1], a, epsilon_f)

                S_next, X_next, allowed_a_next, R, Done = self.env.one_step(a_agent)

                if Done == 1:
                    self.prop.backprop(x, h, a_agent, R, qvalue, Done)

                    checkmate_save[n] = np.copy(R)
                    R_save.append(R)
                    N_moves_save.append(move_counter)
                    break
                else:
                    a_next, _ = np.where(allowed_a_next == 1)
                    x_next, _ = self.prop.forward_pass(X_next)
                    a_agent_next, qvalue_next = epsilon_greedy_policy(
                        x_next[-1], a_next,
                        epsilon_f if self.type == SARSA else 0)
                    self.prop.backprop(x, h, a_agent, R, qvalue, Done, qvalue_next)

                S = np.copy(S_next)
                X = np.copy(X_next)
                allowed_a = np.copy(allowed_a_next)

                move_counter += 1
            else:
                log.error(f"Invalid Epoche. Epoche was longer than {MAX_STEPS_ALLOWED}")

        log.info(f"{self.agent.NAME}, Average reward: {np.mean(R_save)}")
        log.info(f"Number of steps: {np.mean(N_moves_save)}")
        log.info(f"Checkmates: {np.count_nonzero(checkmate_save > 0)}")
        return self.agent.NAME, R_save, N_moves_save


class SarsaNn(NeuralNet):
    type = SARSA


class QlearningNn(NeuralNet):
    type = QLEARNING


class DoubleQlearningNn(NeuralNet):
    type = QLEARNING

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights2 = []
        self.biases2 = []

        self.choice = 0
        self.counter = 0

        self.prop = DoublePropagationHandler(self)

        self.weights2, self.biases2 = initialize_weights(self.layer_sizes, self.xavier)


class DoubleSarsaNn(DoubleQlearningNn):
    type = SARSA


class SarsaNnAdam(SarsaNn):
    type = SARSA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = 'adam'
        # params for adam
        self.adam_eta = 0.01
        self.beta_adam = 0.9
        # initialize Adam
        self.adam_w, self.adam_b = initialize_adam(self.weights, self.biases, self.beta_adam)


class SarsaNnRMSProp(SarsaNn):
    type = SARSA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = 'rmsprop'
        # params for rmsprop
        self.rms_eta = 0.1
        self.gamma_rmsprop = 0.9
        # initialize RMSProp
        self.rms_w, self.rms_b = initialize_rmsprop(self.weights, self.biases, self.gamma_rmsprop)


class SarsaNnSoftmax(SarsaNn):
    type = SARSA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = 'softmax'

