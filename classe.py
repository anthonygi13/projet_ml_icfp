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
        :return: alpha, shape (M, T), alpha_i_t being the proba of seeing (y1, ..., yt)
         and being at state i at time t
        """
        alpha = np.zeros((self.M, Y.shape[0]))
        alpha[:, 0] = np.einsum('i,i->i', self.pi, self.B[:, Y[0]])
        for t in range(1, Y.shape[0]):
            alpha[:, t] = np.einsum('i,i->i', self.B[:, Y[t]], np.einsum('j,ji->i', alpha[:, t-1], self.A))
        return alpha

    def backward(self, Y):
        """
        :param Y: sequence (y1, ..., yT)
        :return: beta, shape (M, T), beta_i_t being the proba of seeing (y(t+1), ..., yT)
         knowing that we are in state i at time t
        """
        T = Y.shape[0]
        beta = np.zeros((self.M, Y.shape[0]))
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

        for t in range(T - 1):
            inter = 0

            # compute the intermediate component
            for j in range(N):
                inter += alpha[j, t] * beta[j, t]

            for i in range(N):
                gamma[i, t] = alpha[i, t] * beta[i, t] / inter
                

        # Compute xsi

        xsi = np.zeros((N, N, T))

        for t in range(T - 1):

            inter = 0

            for k in range(N):
                for w in range(N):
                    inter += alpha[k, t] * self.A[k, w] * beta[w, t + 1] * self.B[w, Y[t + 1]]

            for i in range(N):
                for j in range(N):
<<<<<<< Updated upstream
=======
                    
>>>>>>> Stashed changes
                    xsi[i, j, t] = alpha[i, t] * self.A[i, j] * beta[j, t + 1] * self.B[j, Y[t + 1]] / inter

                    # UPDATE THE PARAMETER

        self.pi[:] = gamma[:, 0] #update pi

        for i in range(N): #Update transition matrix
            for j in range(N):
                self.A[i, j] = np.sum(xsi[:,:,:-1], axis=-1)[i, j] / np.sum(gamma[i,:-1], -1)
                
        
        for i in range(N): #Update symbol generation
            for j in range(M):
                inter1 = 0
                inter2 = 0
                for t in range(T-1):
                    if(Y[t]==j):
                        inter1+=gamma[i,t]
                    inter2+=gamma[i,t]
                
                self.B[i,j]= inter1 / inter2


hmm = HMM(2, 2)
hmm.pi = np.array([0.2, 0.8])

hmm.A = np.array([[0.2, 0.9], [0.8, 0.1]])
hmm.init_parameter()
print(hmm.A)
hmm.B = np.array([[0.2, 0.9], [0.8, 0.1]])
print

<<<<<<< Updated upstream
X, Y = CoinToss(20)
print(X)
hmm.Baum_welch(Y)

print("pi", hmm.pi)
print("A", hmm.A)
print("B", hmm.B)
=======
X, Y = CoinToss(200)
print(X[0])
hmm.Baum_welch(Y)


print(hmm.pi)
print("________________________________")
print(hmm.A.transpose())
print("________________________________")
print(hmm.B.transpose())
>>>>>>> Stashed changes
