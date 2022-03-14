import numpy as np
import matplotlib.pyplot as plt
from classe import *
from generate_coin_toss import *

data_X = np.loadtxt('dataset1/dataset1_X_p0pi=0.90_p0biased=0.25_pchange=0.10_R=1000_T=200.txt', dtype=int)
data_Y = np.loadtxt('dataset1/dataset1_Y_p0pi=0.90_p0biased=0.25_pchange=0.10_R=1000_T=200.txt', dtype=int)
test_frac = 0.3
niter = 500
nRandomInit = 10

R = data_Y.shape[0]
Y_test = data_Y[:int(R * 0.3)]
X_test = data_X[:int(R * 0.3)]
Y_train = data_Y[int(R * 0.3):]
X_train = data_X[int(R * 0.3):]

list_logv_max = []
list_optimal_niter = []
best_hmm = None
best_i = None
logv_max = -np.inf

#plt.subplot(221)
for i in range(nRandomInit):
    print("Random initialisation number {}/{}".format(i+1, nRandomInit))
    hmm = HMM(2, 2)
    logvs_train, logvs_test, optimal_pi, optimal_A, optimal_B = hmm.train(niter, Y_train, Y_test)
    list_logv_max += [np.amax(np.array(logvs_test))]
    list_optimal_niter += [np.argmax(np.array(logvs_test))]
    #plt.plot(np.arange(len(logvs_train)), logvs_train, label='Training set, hmm #{}'.format(i+1))
    plt.plot(np.arange(len(logvs_test)), logvs_test, label='hmm #{}'.format(i+1))
    if list_logv_max[-1] > logv_max:
        logv_max = list_logv_max[-1]
        best_hmm = HMM(2, 2, optimal_pi, optimal_A, optimal_B)
        best_i = i
plt.xlabel("Iteration number")
plt.ylabel("log-likelihood")
plt.legend()

print("pi", best_hmm.pi)
print("A", best_hmm.A)
print("B", best_hmm.B)
print("Logv of test set", best_hmm.logv(Y_test))
print("best_hmm", i+1)

"""
print("list logv max", list_logv_max)
print("list optimal niter", list_optimal_niter)
"""
"""
plt.subplot(222)
plt.scatter(np.arange(nRandomInit)+1, list_logv_max)
plt.xlabel("Random HMM")
plt.ylabel("Best log-likelihood")
plt.legend()

plt.subplot(223)
plt.scatter(np.arange(nRandomInit)+1, list_optimal_niter)
plt.xlabel("Random HMM")
plt.ylabel("Optimal iteration number")
plt.legend()
"""
plt.show()

"""
print("pi", hmm.pi)
print("A", hmm.A)
print("B", hmm.B)
"""
"""
plt.plot(np.arange(len(logvs_train)), logvs_train, label='Training set')
plt.plot(np.arange(len(logvs_test)), logvs_test, label='Test set')
plt.xlabel("Iteration number")
plt.ylabel("log-likelihood")
plt.legend()
plt.show()
"""
