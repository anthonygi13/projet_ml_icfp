import numpy as np

def CoinToss(T):
    Y = np.zeros(T, dtype=int)
    p = 0.9  # ☻proba of biased coin
    pi = np.array([0.8, 0.2])  # proba of initial state
    A = np.array([[0.8, 0.2], [0.1, 0.9]])
    B = np.array([[p, 1-p], [0.5, 0.5]])


    X = np.zeros(T,dtype=int)

    if (np.random.rand() < pi[0]):  # Init first coin toss
        X[0] = 0
    else:
        X[0] = 1

    if (np.random.rand() < B[X[0], 0]):
        Y[0] = 0
    else:
        Y[0] = 1

    for t in range(1, T):
        # Transition
        if (np.random.rand() < A[X[t - 1], 1 - X[t - 1]]):
            X[t] = 1 - X[t - 1]
            
        #Symbol emission
        if (np.random.rand() < B[X[t],0]):
            Y[t] = 0
        else:
            Y[t] = 1

    return X, Y

"""
CT = CoinToss(10)
print(CT[0])
print(CT[1])
"""