import numpy as np


def _residu(A, x, b):
    return float(np.linalg.norm(np.array(A) @ np.array(x) - np.array(b)))


# ══════════════════════════════════════════════════════════
#  JACOBI
# ══════════════════════════════════════════════════════════

def run_jacobi(A, b, tol=1e-4, max_iter=200):
    A, b = np.array(A, dtype=float), np.array(b, dtype=float)
    n    = len(b)
    D    = np.diag(np.diag(A))
    R    = A - D
    D_inv= np.linalg.inv(D)
    x    = np.zeros(n)
    history, errors = [x.tolist()], []
    for k in range(max_iter):
        xnew = D_inv @ (b - R @ x)
        err  = float(np.linalg.norm(xnew - x))
        history.append(xnew.tolist())
        errors.append(err)
        if err < tol:
            return xnew, history, errors, k + 1
        x = xnew
    return x, history, errors, max_iter


# ══════════════════════════════════════════════════════════
#  GAUSS-SEIDEL
# ══════════════════════════════════════════════════════════

def run_gauss_seidel(A, b, tol=1e-4, max_iter=200):
    A, b = np.array(A, dtype=float), np.array(b, dtype=float)
    n    = len(b)
    x    = np.zeros(n)
    history, errors = [x.tolist()], []
    for k in range(max_iter):
        xnew = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * xnew[j] for j in range(i))
            s2 = sum(A[i, j] * x[j]    for j in range(i+1, n))
            xnew[i] = (b[i] - s1 - s2) / A[i, i]
        err = float(np.linalg.norm(xnew - x))
        history.append(xnew.tolist())
        errors.append(err)
        if err < tol:
            return xnew, history, errors, k + 1
        x = xnew
    return x, history, errors, max_iter
