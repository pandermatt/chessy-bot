import numpy as np

class Adam:

    def __init__(self, Params, beta1):

        N_dim = np.shape(np.shape(Params))[
            0]  # It finds out if the parameters given are in a vector (N_dim=1) or a matrix (N_dim=2)

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

        self.epsilon = 10 ** (-8)

        # COUNTER OF THE TRAINING PROCESS
        self.counter = 0

    def Compute(self, Grads):

        self.counter = self.counter + 1

        self.mt = self.beta1 * self.mt + (1 - self.beta1) * Grads

        self.vt = self.beta2 * self.vt + (1 - self.beta2) * Grads ** 2

        mt_n = self.mt / (1 - self.beta1 ** self.counter)
        vt_n = self.vt / (1 - self.beta2 ** self.counter)

        New_grads = mt_n / (np.sqrt(vt_n) + self.epsilon)

        return New_grads
