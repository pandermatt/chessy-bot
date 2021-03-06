import numpy as np

class Adam:
    """
    Handles the computation of the Adam optimizer.
    """
    def __init__(self, Params, beta1):
        # It finds out if the parameters given are in a vector (N_dim=1) or a matrix (N_dim=2)
        N_dim = np.shape(np.shape(Params))[0]

        # INITIALISATION OF THE MOMENTUMS
        if N_dim == 1:
            self.N1 = np.shape(Params)[0]

            self.mt = np.zeros([self.N1])
            self.vt = np.zeros([self.N1])

        if N_dim == 2:
            self.N1 = np.shape(Params)[0]
            self.N2 = np.shape(Params)[1]

            self.mt = np.zeros([self.N1, self.N2])
            self.vt = np.zeros([self.N1, self.N2])

        # HYPERPARAMETERS OF ADAM
        self.beta1 = beta1
        self.beta2 = 0.999

        self.epsilon = 1e-7

        # COUNTER OF THE TRAINING PROCESS
        self.counter = 0

    def Compute(self, Grads, lr):

        self.counter += 1

        eta_new = lr * np.sqrt(1 - self.beta2 ** self.counter) / (1 - self.beta1 ** self.counter)

        self.mt = self.beta1 * self.mt + (1 - self.beta1) * Grads

        self.vt = self.beta2 * self.vt + (1 - self.beta2) * Grads * Grads

        New_grads = eta_new * self.mt / (np.sqrt(self.vt) + self.epsilon)

        return New_grads


class RMSProp:
    """
    Handles the computation of the RMSProp optimizer.
    """

    def __init__(self, Params, gamma):
        # It finds out if the parameters given are in a vector (N_dim=1) or a matrix (N_dim=2)
        N_dim = np.shape(np.shape(Params))[0]

        if N_dim == 1:
            self.N1 = np.shape(Params)[0]

            self.st = np.zeros([self.N1])

        if N_dim == 2:
            self.N1 = np.shape(Params)[0]
            self.N2 = np.shape(Params)[1]

            self.st = np.zeros([self.N1, self.N2])

        self.gamma = gamma
        self.epsilon = 1e-6

    def Compute(self, Grads):

        self.st = self.gamma * self.st + (1 - self.gamma) * Grads * Grads
        new_grads = np.sqrt(self.st + self.epsilon) * Grads

        return new_grads
