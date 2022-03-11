import numpy as np
from generate_coin_toss import CoinToss
import matplotlib.pyplot as plt

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
        
        p = np.random.rand()
        self.pi = np.array([p,1-p])  # Take the uniform distribution
        p1 = np.random.rand()
        p2 =  np.random.rand()
        p3 =  np.random.rand()
        p4 =  np.random.rand()
        self.A = np.array([[p1, 1-p1], [p2, 1-p2]])
        self.B = np.array([[p3, 1-p3], [p4, 1-p4]])
        

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

    def BW_bis(self, Y_seq):
        R = Y_seq.shape[0]
        T = Y_seq.shape[1]
        gamma_tot = np.zeros((R, self.N, T))
        xsi_tot = np.zeros((R, self.N, self.N, T-1))
        for r, Y in enumerate(Y_seq):  # TODO: virer cette boucle
            alpha, beta = self.forward(Y), self.backward(Y)
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=0)
            xsi = np.einsum("it,ij,jt,jt->ijt", alpha[:, :-1], self.A, beta[:, 1:], self.B[:, Y[1:]])
            xsi /= np.sum(xsi, axis=(0, 1))
            gamma_tot[r] = gamma
            xsi_tot[r] = xsi
        self.pi = np.sum(gamma_tot[:, :, 0], axis=0) / R
        self.A = (np.sum(xsi_tot, axis=(0, -1)).T / np.sum(gamma_tot[:, :, :-1], axis=(0, -1))).T
        #print(np.tile(np.arange(self.M), (Y_seq.shape[0], Y_seq.shape[1], 1)).shape)
        #print(np.einsum("ijk->jki", np.tile(Y_seq, (self.M, 1, 1))).shape)
        mask = (np.tile(np.arange(self.M), (Y_seq.shape[0], Y_seq.shape[1], 1)) == np.einsum("ijk->jki", np.tile(Y_seq, (self.M, 1, 1))))
        #print(mask.shape)
        self.B = (np.einsum("rtj,rit->ij", mask, gamma_tot).T / np.sum(gamma_tot, axis=(0, -1))).T


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
                    for t in range(T):
                        if(Y[t, r] == j):
                            inter1 += gamma[i, t, r]
                        inter2 += gamma[i, t, r]
                
                self.B[i, j] = inter1 / inter2


hmm = HMM(2, 2)
hmm.pi = np.array([0.4, 0.6])

hmm.A = np.array([[0.3, 0.7], [0.7, 0.3]])
hmm.B = np.array([[0.7, 0.3], [0.3, 0.7]])

T_0 = 100  # number of coin toss
R_0 = 300  # number of trials


Y_seq = np.zeros((R_0, T_0), dtype=int)

for r in range(R_0):
    X, Y = CoinToss(T_0)
    Y_seq[r, :] = Y

logvs = []
logv = 0
for Y in Y_seq:
    logv += np.log(hmm.probability(Y))
logvs += [logv]

for i in range(20):
    hmm.BW_bis(Y_seq)
    logv = 0
    for Y in Y_seq:
        logv += np.log(hmm.probability(Y))
    logvs += [logv]

plt.scatter(np.arange(len(logvs)), logvs)
plt.show()

print("pi", hmm.pi)
print("A", hmm.A)
print("B", hmm.B)
