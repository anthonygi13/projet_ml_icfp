import numpy as np

class HMM:  # Hidden markov chain with unidimensional symbol with discrete distribution

    def __init__(self, M, N):

        ##The fixed things
        self.M = M  # Number of hidden states
        self.N = N  # Number of symbols

        ##The parameters to optimize
        self.pi = np.zeros(N)  # Initial state distribution
        self.A = np.zeros((N, N))  # Transition matrix
        self.B = np.zeros((M, N))  # Symbol generation

    def init_parameter(self):  # Take uniform everything as initial parameters

        N = self.N  # Oh c'est tellement plus simple que de remettre self.N a chaque fois ...
        M = self.M

        self.pi = 1 / N * np.ones(N)  # Take the uniform distribution
        self.A = 1 / N * np.ones((N, N))
        self.B = 1 / M * np.ones((M, N))

    def forward(self, Y):
        """
        :param Y: sequence (y1, ..., yT)
        :return: alpha, shape (M, T), alpha_i_t being the proba of seeing (y1, ..., yt)
         and being at state i at time t
        """
        pass

    def backward(self, Y):
        """
        :param Y: sequence (y1, ..., yT)
        :return: beta, shape (M, T), beta_i_t being the proba of seeing (y(t+1), ..., yT)
         knowing that we are in state i at time t
        """
        pass

    def Baum_welch(self, Y):  # Given a sample Y = (y1,....yL), what are the new parameters ?

        N = self.N  # Oh c'est tellement plus simple que de remettre self.N a chaque fois ...
        M = self.M

        T = Y.shape  # Data length

        alpha, beta = self.FW()

        ##UPDATE

        # compute gamma
        gamma = np.zeros((N, T))

        for t in range(T - 1):
            inter = 0

            # compute the intermediate component
            for j in range(N - 1):
                inter += alpha[j, t] * beta[j, t]

            for i in range(N - 1):
                gamma[i, t] = alpha[i, t] * beta[i, t] / inter

        # Compute xsi

        xsi = np.zeros((N, N, T))

        for t in range(T - 1):

            inter = 0

            for k in range(N - 1):
                for w in range(N - 1):
                    inter += alpha[k, t] * self.A[k, w] * beta[w, t + 1] * B[w, Y(t + 1)]

            for i in range(N - 1):
                for j in range(N - 1):
                    xsi[i, j, t] = alpha[i, t] * self.A[i, j] * beta[j, t + 1] * B[j, Y(t + 1)] / inter

                    # UPDATE THE PARAMETER

        self.pi[:] = gamma[:, 1]

        for i in range(N - 1):
            for j in range(N - 1):
                self.A[i, j] = np.sum(xsi, axis=-1)[i, j] / np.sum(gamma[i], -1)
