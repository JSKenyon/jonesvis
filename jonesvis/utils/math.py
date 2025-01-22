import numpy as np

def kron_matvec(A, b):
    D = len(A)
    N = b.size
    x = b

    for d in range(D):
        Gd = A[d].shape[0]
        NGd = N // Gd
        X = np.reshape(x, (Gd, NGd))
        Z = A[d].dot(X).T
        x = Z.ravel()
    return x.reshape(b.shape)