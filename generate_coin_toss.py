import numpy as np

def CoinToss(T, pi=np.array([0.3, 0.7]), A=np.array([[0.6, 0.4], [0.1, 0.9]]), B=np.array([[0.3, 0.7], [0.5, 0.5]])):
    Y = np.zeros(T, dtype=int)
    X = np.zeros(T, dtype=int)

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


def generate_dataset(R, T, p0_pi, p0_biased, p_change, add2fname=''):
    pi = np.array([p0_pi, 1-p0_pi])
    A = np.array([[1-p_change, p_change], [p_change, 1-p_change]])
    B = np.array([[0.5, 0.5], [p0_biased, 1-p0_biased]])

    X_seq = np.zeros((R, T), dtype=int)
    Y_seq = np.zeros((R, T), dtype=int)
    for r in range(R):
        X, Y = CoinToss(T, pi=pi, A=A, B=B)
        X_seq[r, :] = X
        Y_seq[r, :] = Y

    np.savetxt('{}_X_p0pi={:.2f}_p0biased={:.2f}_pchange={:.2f}_R={}_T={}.txt'.format(add2fname, p0_pi, p0_biased, p_change, R, T), X_seq)
    np.savetxt('{}_Y_p0pi={:.2f}_p0biased={:.2f}_pchange={:.2f}_R={}_T={}.txt'.format(add2fname, p0_pi, p0_biased, p_change, R, T), Y_seq)


#generate_dataset(1000, 200, 0.9, 0.25, 0.1, 'dataset1')