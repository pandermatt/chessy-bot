from agents.chessy_agent import ChessyAgent
from neuronal_engine.neural_net import SarsaNnCustomValues


class SarsaChessyAgentCustomValuesGamma1(ChessyAgent):
    NAME = "SARSA $\gamma = 0.95$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(gamma=0.95)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesGamma2(ChessyAgent):
    NAME = "SARSA $\gamma = 0.8$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(gamma=0.8)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesGamma3(ChessyAgent):
    NAME = "SARSA $\gamma = 0.6$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(gamma=0.6)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesGamma4(ChessyAgent):
    NAME = "SARSA $\gamma = 0.5$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(gamma=0.5)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesGamma5(ChessyAgent):
    NAME = "SARSA $\gamma = 0.9$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(gamma=0.9)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesBeta1(ChessyAgent):
    NAME = "SARSA $\\beta = 0.0005$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.0005)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesBeta2(ChessyAgent):
    NAME = "SARSA $\\beta = 0.005$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.005)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesBeta3(ChessyAgent):
    NAME = "SARSA $\\beta = 0.05$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.05)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesBeta4(ChessyAgent):
    NAME = "SARSA $\\beta = 0.000005$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.000005)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesBeta5(ChessyAgent):
    NAME = "SARSA $\\beta = 0.00005$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(beta=0.00005)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesEta1(ChessyAgent):
    NAME = "SARSA $\eta = 0.05$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(eta=0.01)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesEta2(ChessyAgent):
    NAME = "SARSA $\\eta = 0.5$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(eta=0.5)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesEta3(ChessyAgent):
    NAME = "SARSA $\eta = 0.0005$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(eta=0.0005)
        return nn.train(self.N_episodes, callback)


class SarsaChessyAgentCustomValuesEta4(ChessyAgent):
    NAME = "SARSA $\eta = 0.005$"
    NN_KLASS = SarsaNnCustomValues

    def run(self, callback=lambda *args: None):
        nn = self.NN_KLASS(self, xavier=True)
        nn.set_custom_values(eta=0.005)
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
