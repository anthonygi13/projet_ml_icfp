import numpy as np
from generate_coin_toss import CoinToss

class HMM:  # Hidden markov chain with unidimensional symbol with discrete distribution

    def __init__(self, M, N, pi=None, A=None, B=None):

        ##The fixed things
        self.M = M  # Number of symbols
        self.N = N  # Number of hidden states

        ##The parameters to optimize
        if pi is None:
            self.pi = np.zeros(N)  # Initial state distribution
        if A is None:
            self.A = np.zeros((N, N))  # Transition matrix
        if B is None:
            self.B = np.zeros((N, M))  # Symbol generation

    def init_parameter(self):  # Take uniform everything as initial parameters

        N = self.N  # Oh c'est tellement plus simple que de remettre self.N a chaque fois ...
        M = self.M

        self.pi = 1 / N * np.ones(N)  # Take the uniform distribution
        self.A = 1 / N * np.ones((N, N))
        self.B = 1 / M * np.ones((N, M))

    def forward(self, Y):
        """
        :param Y: sequence (y1, ..., yT)
        :return: alpha, shape (N, T), alpha_i_t being the proba of seeing (y1, ..., yt)
         and being at state i at time t
        """
        alpha = np.zeros((self.N, Y.shape[0]))
        alpha[:, 0] = np.einsum('i,i->i', self.pi, self.B[:, Y[0]])
        for t in range(1, Y.shape[0]):
            alpha[:, t] = np.einsum('i,i->i', self.B[:, Y[t]], np.einsum('j,ji->i', alpha[:, t-1], self.A))
        return alpha

    def backward(self, Y):
        """
        :param Y: sequence (y1, ..., yT)
        :return: beta, shape (N, T), beta_i_t being the proba of seeing (y(t+1), ..., yT)
         knowing that we are in state i at time t
        """
        T = Y.shape[0]
        beta = np.zeros((self.N, Y.shape[0]))
        beta[:, T-1] = 1
        for t in range(T-2, -1, -1):
            beta[:, t] = np.einsum('j,ij,j->i', beta[:, t+1], self.A, self.B[:, Y[t+1]])
        return beta

    def Baum_welch(self, Y):  # Given a sample Y = (y1,....yL), what are the new parameters ?

        N = self.N  # Oh c'est tellement plus simple que de remettre self.N a chaque fois ...
        M = self.M

        T = Y.shape[0]  # Data length

        alpha, beta = self.forward(Y), self.backward(Y)

        ##UPDATE

        # compute gamma
        gamma = np.zeros((N, T))

        for t in range(T):
            inter = 0

            # compute the intermediate component
            for j in range(N):
                inter += alpha[j, t] * beta[j, t]

            for i in range(N):
                gamma[i, t] = alpha[i, t] * beta[i, t] / inter

        print("gamma", gamma)
        # Compute xsi

        xsi = np.zeros((N, N, T-1))

        for t in range(T-1):

            inter = 0

            for k in range(N):
                for w in range(N):
                    inter += alpha[k, t] * self.A[k, w] * beta[w, t + 1] * self.B[w, Y[t + 1]]

            for i in range(N):
                for j in range(N):

                    xsi[i, j, t] = alpha[i, t] * self.A[i, j] * beta[j, t + 1] * self.B[j, Y[t + 1]] / inter

                    # UPDATE THE PARAMETER
        print("xsi", xsi)
        self.pi[:] = gamma[:, 0] #update pi

        for i in range(N): #Update transition matrix
            for j in range(N):
                self.A[i, j] = np.sum(xsi, axis=-1)[i, j] / np.sum(gamma[i,:-1], -1)
                
        
        for i in range(N): #Update symbol generation
            for j in range(M):
                inter1 = 0
                inter2 = 0
                for t in range(T):
                    if(Y[t]==j):
                        inter1+=gamma[i,t]
                    inter2+=gamma[i,t]
                
                self.B[i,j]= inter1 / inter2

    def BW_bis(self, Y):
        alpha, beta = self.forward(Y), self.backward(Y)
        gamma = alpha*beta
        gamma /= np.sum(gamma, axis=0)
        print("gamma", gamma)
        xsi = np.einsum("it,ij,jt,jt->ijt", alpha[:, :-1], self.A, beta[:, 1:], self.B[:, Y[1:]])
        xsi /= np.sum(xsi, axis=(0, 1))
        print("xsi", xsi)
        self.pi = np.array(gamma[:, 0], copy=True)
        self.A = (np.sum(xsi, axis=-1).T/np.sum(gamma[:, :-1], axis=-1)).T
        mask = np.tile(np.arange(self.M), (Y.shape[0], 1)).T == np.tile(Y, (self.M, 1))
        self.B = (np.einsum("jt,it->ij", mask, gamma).T/np.sum(gamma, axis=-1)).T

X, Y = CoinToss(10)
print("bw corentin")
hmm = HMM(2, 2)
hmm.pi = np.array([0.2, 0.8])
hmm.A = np.array([[0.2, 0.9], [0.8, 0.1]])
hmm.init_parameter()
hmm.B = np.array([[0.2, 0.9], [0.8, 0.1]])
hmm.Baum_welch(Y)
print("pi", hmm.pi)
print("A", hmm.A)
print("B", hmm.B)
print(np.sum(hmm.B, axis=1))

print("bw bis")
hmm = HMM(2, 2)
hmm.pi = np.array([0.2, 0.8])
hmm.A = np.array([[0.2, 0.9], [0.8, 0.1]])
hmm.init_parameter()
hmm.B = np.array([[0.2, 0.9], [0.8, 0.1]])
hmm.BW_bis(Y)
print("pi", hmm.pi)
print("A", hmm.A)
print("B", hmm.B)
print(np.sum(hmm.B, axis=1))