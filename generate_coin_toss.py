import numpy as np

def CoinToss(T):
    Y = np.zeros(T, dtype=int)
    X = np.zeros(T, dtype=int)
    p = 0.3  # ☻proba of biased coin
    pi = np.array([0.3, 0.7])  # proba of initial state
    A = np.array([[0.6, 0.4], [0.1, 0.9]])
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
        if np.random.rand() < A[X[t - 1], 1 - X[t - 1]]:
            X[t] = 1 - X[t - 1]
        else:
            X[t] = X[t - 1]
        # Emission
        if np.random.rand() < B[X[t], 0]:
            Y[t] = 0
        else:
            Y[t] = 1

    return X, Y

"""
CT = CoinToss(10)
print(CT[0])
print(CT[1])
"""