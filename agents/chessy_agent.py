import numpy as np

from agents.agent import Agent
from chess_env import ChessEnv
from neuronal_engine.neural_net import SarsaNn, QlearningNn, DoubleQlearningNn, DoubleSarsaNn, NeuralNet, SarsaNnAdam, \
    SarsaNnRMSProp, SarsaNnSigmoid, SarsaNnLeakyReLU, QLearningNnLeakyReLU, QlearningNnRMSProp, \
    QlearningNnRMSPropLeakyReLU


class ChessyAgent(Agent):
    """
    Main Class for all ChessyAgents.
    Each Class has a name, and an own custom neural network.
    """
    NN_KLASS = NeuralNet

    def __init__(self, N_episodes):
        """
        @param N_episodes: Nr of episodes
        """
        super().__init__()
        self.N_episodes = N_episodes

    def _get_layer_sizes(self):
        """
        @return: layer sizes with hidden nodes
        """
        board_state, X, allowed_actions = self.env.initialise_game()
        # AMOUNT POSSIBLE ACTIONS
        N_a = np.shape(allowed_actions)[0]
        # INPUT SIZE
        N_in = np.shape(X)[0]

        # NUMBER OF HIDDEN NODES
        N_h1 = 200
        N_h2 = 200

        return [N_in, N_h1, N_h2, N_a]

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        return nn.train(self.N_episodes, callback)


# PLEASE: CHECK NAME OF AGENT FOR THE DESCRIPTION
# THE CLEAN CLASS ALWAYS RUNS WITH THE DEFAULT CHESSY_AGENT AND ITS CONFIG

class SarsaChessyAgent(ChessyAgent):
    NAME = "SARSA"
    NN_KLASS = SarsaNn


class SarsaChessyAgentOneHidden(SarsaChessyAgent):
    NAME = "SARSA with one hidden layer"

    def _get_layer_sizes(self):
        board_state, X, allowed_actions = self.env.initialise_game()
        N_a = np.shape(allowed_actions)[0]
        N_in = np.shape(X)[0]
        N_h1 = 200

        return [N_in, N_h1, N_a]


class SarsaChessyAgentThreeHidden(SarsaChessyAgent):
    NAME = "SARSA with three hidden layer (200 - 100 - 200)"

    def _get_layer_sizes(self):
        board_state, X, allowed_actions = self.env.initialise_game()
        N_a = np.shape(allowed_actions)[0]
        N_in = np.shape(X)[0]
        return [N_in, 200, 100, 200, N_a]


class SarsaChessyAgentCustomReward(SarsaChessyAgent):
    NAME = 'SARSA with negative reward (-0.1)'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_step=-0.1)


class SarsaChessyAgentCustomReward2(SarsaChessyAgent):
    NAME = 'SARSA with negative reward (-0.2)'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_step=-0.1)


class QLearningChessyAgent(ChessyAgent):
    NAME = "Q-learning"
    NN_KLASS = QlearningNn


class QLearningChessyAgentLeakyReLU(ChessyAgent):
    NAME = "Q-learning Leaky ReLU"
    NN_KLASS = QLearningNnLeakyReLU


class QLearningChessyAgentRMSProp(ChessyAgent):
    NAME = "Q-learning RMSProp"
    NN_KLASS = QlearningNnRMSProp


class QLearningChessyAgentRMSPropLeakyReLU(ChessyAgent):
    NAME = "Q-learning RMSProp & Leaky ReLU"
    NN_KLASS = QlearningNnRMSPropLeakyReLU


class QLearningChessyAgentCustomReward(SarsaChessyAgent):
    NAME = 'Q-learning with negative reward (-0.1)'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_step=-0.1)


class QLearningChessyAgentCustomReward2(SarsaChessyAgent):
    NAME = 'Q-learning with negative reward (-0.2)'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_step=-0.1)


class DoubleQLearningChessyAgent(ChessyAgent):
    NAME = "Double-Q-learning"
    NN_KLASS = DoubleQlearningNn


class DoubleSarsaChessyAgent(ChessyAgent):
    NAME = "Double-SARSA-learning"
    NN_KLASS = DoubleSarsaNn


class SarsaChessyAgentAdam(ChessyAgent):
    NAME = "SARSA Adam"
    NN_KLASS = SarsaNnAdam


class SarsaChessyAgentRMSProp(ChessyAgent):
    NAME = "SARSA RMSProp"
    NN_KLASS = SarsaNnRMSProp


class SarsaChessyAgentSigmoid(ChessyAgent):
    NAME = "SARSA Sigmoid"
    NN_KLASS = SarsaNnSigmoid


class SarsaChessyAgentHighReward(SarsaChessyAgent):
    NAME = 'SARSA with checkmate reward of 5'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_checkmate=5)


class QLearningChessyAgentHighReward(QLearningChessyAgent):
    NAME = 'Q-learning with checkmate reward of 5'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_checkmate=5)


class SarsaChessyAgentLeakyReLU(ChessyAgent):
    NAME = "SARSA Leaky ReLU"
    NN_KLASS = SarsaNnLeakyReLU


class SarsaChessyAgentStepReward1(SarsaChessyAgent):
    NAME = 'SARSA'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_step=0, reward_checkmate=1, reward_draw=0)


class SarsaChessyAgentStepReward2(SarsaChessyAgent):
    NAME = 'SARSA step -0.01'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_step=-0.01, reward_checkmate=1, reward_draw=0)


class SarsaChessyAgentStepReward3(SarsaChessyAgent):
    NAME = 'SARSA step -0.1'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_step=-0.1, reward_checkmate=1, reward_draw=0)


class SarsaChessyAgentMateReward1(SarsaChessyAgent):
    NAME = 'SARSA'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_checkmate=1, reward_step=0, reward_draw=0)


class SarsaChessyAgentMateReward2(SarsaChessyAgent):
    NAME = 'SARSA checkmate 5'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_checkmate=5, reward_step=0, reward_draw=0)


class SarsaChessyAgentMateReward3(SarsaChessyAgent):
    NAME = 'SARSA checkmate 10'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_checkmate=10, reward_step=0, reward_draw=0)


class SarsaChessyAgentDrawReward1(SarsaChessyAgent):
    NAME = 'SARSA draw -1'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_draw=-1, reward_step=0, reward_checkmate=1)


class SarsaChessyAgentDrawReward2(SarsaChessyAgent):
    NAME = 'SARSA draw -5'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_draw=-5, reward_step=0, reward_checkmate=1)


class SarsaChessyAgentDrawReward3(SarsaChessyAgent):
    NAME = 'SARSA'

    def __init__(self, N_episodes):
        super().__init__(N_episodes)
        self.env = ChessEnv(4, reward_draw=0, reward_step=0, reward_checkmate=1)

