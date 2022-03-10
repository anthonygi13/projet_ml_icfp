import numpy as np

def CoinToss(T):
    Y = np.zeros(T, dtype=int)
    p = 0.2  # ☻proba of biased coin
    pi = np.array([0.8, 0.2])  # proba of initial state
    A = np.array([[0.5, 0.5], [0.5, 0.5]])
    B = np.array([[p, 0.5], [1 - p, 0.5]])

    X = np.zeros(T)

    if (np.random.rand() < pi[0]):  # Init first coin toss
        X[0] = 0
    else:
        X[0] = 1

    if (np.random.rand() < B[0, int(X[0])]):
        Y[0] = 0
    else:
        Y[0] = 1

    for t in range(1, T):
        # Transition
        if (np.random.rand() < A[int(X[t - 1]), 1 - int(X[t - 1])]):
            X[t] = 1 - X[t - 1]
        if (np.random.rand() < B[0, int(X[t])]):
            Y[t] = 0
        else:
            Y[t] = 1

    return X, Y

"""
CT = CoinToss(10)
print(CT[0])
print(CT[1])
"""