from agents.chessy_agent import ChessyAgent
from neuronal_engine.neural_net import SarsaNnCustomValues


class SarsaChessyAgentCustomValues(ChessyAgent):
    NAME = "SARSA $\gamma = 0$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(gamma=0.0)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues1(ChessyAgent):
    NAME = "SARSA $\gamma = 1$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(gamma=1.0)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues2(ChessyAgent):
    NAME = "SARSA $\\beta = 1$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=1.0)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues3(ChessyAgent):
    NAME = "SARSA $\eta = 1$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(eta=1.0)
        return nn.train(self.N_episodes, callback)

class SarsaChessyAgentCustomValues4(ChessyAgent):
    NAME = "SARSA $\\beta = 0.5$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.5)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues5(ChessyAgent):
    NAME = "SARSA $\eta = 0.5$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(eta=0.5)
        return nn.train(self.N_episodes, callback)
