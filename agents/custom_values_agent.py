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
    NAME = "SARSA $\\beta = 0.001$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.001)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues3(ChessyAgent):
    NAME = "SARSA $\eta = 0.01$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(eta=0.01)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues4(ChessyAgent):
    NAME = "SARSA $\\beta = 0.0$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.0)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues5(ChessyAgent):
    NAME = "SARSA $\eta = 0.0$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(eta=0.0)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues6(ChessyAgent):
    NAME = "SARSA $\\beta = 0.01$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.01)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues7(ChessyAgent):
    NAME = "SARSA $\\beta = 0.1$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.1)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues8(ChessyAgent):
    NAME = "SARSA $\\beta = 0.001$ & $\gamma = 0.95$ & $\eta = 0.005$ "
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.001, gamma=0.95, eta=0.005)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValues9(ChessyAgent):
    NAME = "SARSA $\\beta = 0.001$ & $\gamma = 0.85$ "
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.001, gamma=0.85)
        return nn.train(self.N_episodes, callback)
