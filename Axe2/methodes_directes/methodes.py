import numpy as np


def _residu(A, x, b):
    return float(np.linalg.norm(np.array(A) @ np.array(x) - np.array(b)))


# ══════════════════════════════════════════════════════════
#  GAUSS (élimination avec pivot partiel)
# ══════════════════════════════════════════════════════════

def gauss(A, b):
    n  = len(b)
    Ab = np.hstack([A.copy(), b.reshape(-1, 1)])
    for k in range(n - 1):
        p = np.argmax(np.abs(Ab[k:, k])) + k
        Ab[[k, p]] = Ab[[p, k]]
        if abs(Ab[k, k]) < 1e-14:
            raise ValueError("Pivot nul")
        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, n] - Ab[i, i+1:n] @ x[i+1:]) / Ab[i, i]
    return x


# ══════════════════════════════════════════════════════════
#  LU (décomposition LU sans pivot)
# ══════════════════════════════════════════════════════════

def lu(A, b):
    n = len(b)
    L = np.eye(n)
    U = A.copy()
    for k in range(n - 1):
        if abs(U[k, k]) < 1e-14:
            raise ValueError("Pivot nul")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    y = np.linalg.solve(L, b)
    return np.linalg.solve(U, y)


# ══════════════════════════════════════════════════════════
#  CHOLESKY (matrice symétrique définie positive)
# ══════════════════════════════════════════════════════════

def cholesky(A, b):
    n = len(b)
    if not np.allclose(A, A.T):
        raise ValueError("Matrice non symétrique")
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("Matrice non définie positive")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)
