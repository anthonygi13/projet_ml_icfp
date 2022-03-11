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

    def backward_bis(self,Y):
        
        N = self.N  # Oh c'est tellement plus simple que de remettre self.N a chaque fois ...
        M = self.M
        
        T = Y.shape[0]
        beta = np.zeros((self.N, Y.shape[0]))
        
        beta[:,T-1] = 1
        
        for t in range(T-2,-1,-1):
            for i in range(N):
                inter = 0
                for j in range(N):
                    inter+=beta[j,t+1]*self.A[i,j]*self.B[j,Y[t+1]]
                beta[i,t]=inter
            
        return beta            

    def forward_bis(self,Y):
        
        N = self.N  # Oh c'est tellement plus simple que de remettre self.N a chaque fois ...
        M = self.M
        
        T= Y.shape[0]
        alpha = np.zeros((self.N, Y.shape[0]))
        
        alpha[:,0] = self.pi[:] * self.B[:,Y[0]]
        
        
        for t in range(0,T-1):
            for i in range(N):
                inter = 0
                for j in range(N):
                    inter+=alpha[j,t]*self.A[j,i]
                alpha[i,t+1]=self.B[i,Y[t+1]]*inter 
                
        return alpha

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

    def probability(self, Y):
        """
        :param Y: sequence (y1, ..., yT)
        :return: probability of Y being generated by current hmm
        """
        alpha = self.forward(Y)
        return np.sum(alpha[:, -1])

    def Baum_welch_sequence(self, Y):  # Given a sample Y = (y1,....yL), what are the new parameters ?

        N = self.N  # Oh c'est tellement plus simple que de remettre self.N a chaque fois ...
        M = self.M
        
        T=Y.shape[0]
        R=Y.shape[1]
    
        ##UPDATE

        # compute gamma
        gamma = np.zeros((N, T,R))
        xsi = np.zeros((N, N, T-1,R))
        
        for r in range(R):
        
            alpha, beta = self.forward(Y[:,r]), self.backward(Y[:,r])
            
            for t in range(T):
                inter = 0
    
                # compute the intermediate component
                for j in range(N):
                    inter += alpha[j, t] * beta[j, t]
    
                for i in range(N):
                    gamma[i, t,r] = alpha[i, t] * beta[i, t] / inter
                    
    
            # Compute xsi
    
            for t in range(T - 1):
    
                inter = 0
    
                for k in range(N):
                    for w in range(N):
                        inter += alpha[k, t] * self.A[k, w] * beta[w, t + 1] * self.B[w, Y[t + 1,r]]
    
                for i in range(N):
                    for j in range(N):
    
                        xsi[i, j, t,r] = alpha[i, t] * self.A[i, j] * beta[j, t + 1] * self.B[j, Y[t + 1,r]] / inter

                    # UPDATE THE PARAMETER

        self.pi[:] = np.sum(gamma[:, 0,:],axis=-1) / R #update pi

        for i in range(N): #Update transition matrix
            for j in range(N):
                self.A[i, j] = np.sum(xsi, axis=(-1,-2))[i, j] / np.sum(gamma[i,:-1], axis=(-1,-2))

        
        for i in range(N): #Update symbol generation
            for j in range(M):
                inter1 = 0
                inter2 = 0
                for r in range(R):
                    for t in range(T-1):
                        if(Y[t,r]==j):
                            inter1+=gamma[i,t,r]
                        inter2+=gamma[i,t,r]
                
                self.B[i,j]= inter1 / inter2           


hmm = HMM(2, 2)
hmm.pi = np.array([0.4, 0.6])

hmm.A = np.array([[0.8, 0.2], [0.2, 0.8]])
hmm.B = np.array([[0.9, 0.1], [0.6, 0.4]])
hmm.init_parameter()

T_0 = 100 #number of coin toss
R_0 = 300 #number of trials

T_0 = 300  # number of coin toss
R_0 = 10000  # number of trials

Y_seq = np.zeros((T_0, R_0), dtype=int)

for r in range(R_0):
    X, Y = CoinToss(T_0)
    Y_seq[:, r] = Y


for i in range(100):
    hmm.Baum_welch_sequence(Y_seq)

print("pi", hmm.pi)
print("A", hmm.A)
print("B", hmm.B)
