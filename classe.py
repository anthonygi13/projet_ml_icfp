import numpy as np
from generate_coin_toss import CoinToss
import matplotlib.pyplot as plt


class HMM:  # Hidden markov chain with unidimensional symbol with discrete distribution

    def __init__(self, M, N, pi=None, A=None, B=None):

        ##The fixed things
        self.M = M  # Number of symbols
        self.N = N  # Number of hidden states

        ##The parameters to optimize
        if pi is None and A is None and B is None:
            self.init_parameter()
        else:
            self.pi = pi
            self.A = A
            self.B = B

    def init_parameter(self):  # Take uniform everything as initial parameters

        p = np.random.rand()
        self.pi = np.array([p, 1-p])  # Take the uniform distribution
        p1 = np.random.rand()
        p2 = np.random.rand()
        p3 = np.random.rand()
        p4 = np.random.rand()
        self.A = np.array([[p1, 1-p1], [p2, 1-p2]])
        self.B = np.array([[p3, 1-p3], [p4, 1-p4]])
     
    def viterbi(self,y): #from https://www.delftstack.com/fr/howto/python/viterbi-algorithm-python/
        K = self.N 
        A=self.A 
        B=self.B 
        initial_probs = self.pi 
        T = len(y)
        T1 = np.empty((K, T), 'd') 
        T2 = np.empty((K, T), 'B') 
        T1[:, 0] = initial_probs * B[:, y[0]] 
        T2[:, 0] = 0 
         
        for i in range(1, T): 
            T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1) 
            T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1) 
     
        x = np.empty(T, 'B') 
        x[-1] = np.argmax(T1[:, T - 1]) 
         
        for i in reversed(range(1, T)): 
            x[i - 1] = T2[x[i], i] 
     
        return x, T1, T2

    def forward_seq(self, Y_seq):
        """
        :param Y_seq: sequences, shape (R, T)
        :return: alpha, shape (R, N, T), alpha_rit being the proba of seeing (y_r1, ..., y_rt)
         and being at state i at time t
        """
        alpha = np.zeros((Y_seq.shape[0], self.N, Y_seq.shape[1]))
        alpha[:, :, 0] = np.einsum('i,ir->ri', self.pi, self.B[:, Y_seq[:, 0]])
        for t in range(1, Y_seq.shape[1]):
            alpha[:, :, t] = np.einsum('ir,ri->ri', self.B[:, Y_seq[:, t]], np.einsum('rj,ji->ri', alpha[:, :, t-1], self.A))
        return alpha

    def backward_seq(self, Y_seq):
        """
        :param Y_seq: sequences, shape (R, T)
        :return: beta, shape (R, N, T), beta_rit being the proba of seeing (y_r(t+1), ..., y_rT)
         knowing that we are in state i at time t
        """
        T = Y_seq.shape[1]
        beta = np.zeros((Y_seq.shape[0], self.N, T))
        beta[:, :, T-1] = 1
        for t in range(T-2, -1, -1):
            beta[:, :, t] = np.einsum('rj,ij,jr->ri', beta[:, :, t+1], self.A, self.B[:, Y_seq[:, t+1]])
        return beta

    def logv(self, Y_seq):
        """
        :param Y_seq: sequences, shape (R, T)
        :return: log-vraisemblance de Y_seq
        """
        alpha = self.forward_seq(Y_seq)
        probas = np.sum(alpha[:, :, -1], axis=1)
        return np.sum(np.log(probas))

    def BW(self, Y_seq):
        R = Y_seq.shape[0]
        alpha, beta = self.forward_seq(Y_seq), self.backward_seq(Y_seq)

        gamma = alpha * beta
        gamma = np.einsum("rit,rt->rit", gamma, 1/np.sum(gamma, axis=1))
        xsi = np.einsum("rit,ij,rjt,jrt->rijt", alpha[:, :, :-1], self.A, beta[:, :, 1:], self.B[:, Y_seq[:, 1:]])
        xsi = np.einsum("rijt,rt->rijt", xsi, 1/np.sum(xsi, axis=(1, 2)))

        self.pi = np.sum(gamma[:, :, 0], axis=0) / R
        self.A = (np.sum(xsi, axis=(0, -1)).T / np.sum(gamma[:, :, :-1], axis=(0, -1))).T
        mask = (np.tile(np.arange(self.M), (Y_seq.shape[0], Y_seq.shape[1], 1)) == np.einsum("ijk->jki", np.tile(Y_seq, (self.M, 1, 1))))
        self.B = (np.einsum("rtj,rit->ij", mask, gamma).T / np.sum(gamma, axis=(0, -1))).T

    def train(self, niter, Y_train, Y_test=None):
        print("Training...")
        logvs_train = [self.logv(Y_train)]
        max_reached = False
        if Y_test is not None:
            logvs_test = [self.logv(Y_test)]
            optimal_pi = None
            optimal_A = None
            optimal_B = None
        for i in range(niter):
            if i % 200 == 0 and i != 0:
                print("Iteration {}/{}".format(i, niter))
            self.BW(Y_train)
            logvs_train += [self.logv(Y_train)]
            if Y_test is not None:
                logvs_test += [self.logv(Y_test)]
                if logvs_test[-1] >= logvs_test[-2] and not max_reached:
                    optimal_pi = np.array(self.pi, copy=True)
                    optimal_A = np.array(self.A, copy=True)
                    optimal_B = np.array(self.B, copy=True)
                else:
                    max_reached = True
        print("Training finished")
        if Y_test is None:
            return logvs_train
        else:
            return logvs_train, logvs_test, optimal_pi, optimal_A, optimal_B


if __name__ == '__main__':

    hmm = HMM(2, 2)
    hmm.pi = np.array([0.4, 0.6])
    hmm.A = np.array([[0.3, 0.7], [0.7, 0.3]])
    hmm.B = np.array([[0.7, 0.3], [0.3, 0.7]])

    print("pi", hmm.pi)
    print("A", hmm.A)
    print("B", hmm.B)
    print("____________")
    T_0 = 100  # number of coin toss
    R_0 = 300  # number of trials

    Y_seq = np.zeros((R_0, T_0), dtype=int)

    for r in range(R_0):
        X, Y = CoinToss(T_0)
        Y_seq[r, :] = Y

    logvs = [hmm.logv(Y_seq)]

    for i in range(200):
        hmm.BW(Y_seq)
        logvs += [hmm.logv(Y_seq)]

    print("pi", hmm.pi)
    print("A", hmm.A)
    print("B", hmm.B)

    plt.scatter(np.arange(len(logvs)), logvs)
    plt.show()
