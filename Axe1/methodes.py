import numpy as np
import sympy as sp
import math
from utils import safe_eval


# ══════════════════════════════════════════════════════════
#  BISECTION
# ══════════════════════════════════════════════════════════

def run_bisection(f, a, b, eps, max_iter=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a)·f(b) > 0 : pas de changement de signe sur [a, b]")
    rows, c = [], None
    for i in range(1, max_iter + 1):
        c   = (a + b) / 2
        fc  = f(c)
        err = (b - a) / 2
        rows.append({ "iter": i, "a": a, "b": b, "c": c, "fc": fc, "err": err })
        if err < eps:
            break
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, rows


# ══════════════════════════════════════════════════════════
#  NEWTON-RAPHSON
# ══════════════════════════════════════════════════════════

def run_newton(f, df, ddf, a, b, x0, eps, max_iter=200):
    xs       = np.linspace(a, b, 100)
    df_vals  = np.array([abs(float(df(xi))) for xi in xs if math.isfinite(float(df(xi)))])
    ddf_vals = np.array([abs(float(ddf(xi))) for xi in xs if math.isfinite(float(ddf(xi)))])
    m = float(np.min(df_vals))  if len(df_vals)  else 1.0
    M = float(np.max(ddf_vals)) if len(ddf_vals) else 1.0

    x    = x0
    rows = []
    for i in range(1, max_iter + 1):
        fx  = float(f(x))
        dfx = float(df(x))
        if abs(dfx) < 1e-14:
            raise ValueError(f"Dérivée nulle en x = {x:.6f}")
        xnew = x - fx / dfx
        err  = (M / (2 * m)) * (xnew - x) ** 2 if m > 0 else abs(xnew - x)
        rows.append({ "iter": i, "xn": x, "fx": fx, "dfx": dfx,
                      "xnew": xnew, "err": abs(xnew - x), "err_estimee": err })
        if abs(xnew - x) < eps:
            return xnew, rows
        x = xnew
    return x, rows


# ══════════════════════════════════════════════════════════
#  POINT FIXE
# ══════════════════════════════════════════════════════════

def run_point_fixe(g, dg, x0, eps, max_iter=200):
    x    = x0
    rows = []
    for i in range(1, max_iter + 1):
        xnew = float(g(x))
        err  = abs(xnew - x)
        rows.append({ "iter": i, "x": x, "xnew": xnew, "err": err,
                      "dg": safe_eval(dg, x) })
        if err < eps:
            return xnew, rows
        x = xnew
        if not math.isfinite(x):
            raise ValueError("Divergence détectée")
    return x, rows
