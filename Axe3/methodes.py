import numpy as np
import sympy as sp
import math


_ALLOWED_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "pi": np.pi, "e": np.e, "abs": np.abs,
}


# ══════════════════════════════════════════════════════════
#  UTILITAIRES AXE 3
# ══════════════════════════════════════════════════════════

def parse_values(text):
    values = text if isinstance(text, list) else str(text).replace(";", ",").split(",")
    result = []
    for value in values:
        if str(value).strip() != "":
            result.append(float(value))
    if not result:
        raise ValueError("Liste vide.")
    return np.array(result, dtype=float)


def poly_value(coefficients, x):
    result = 0.0
    for power, coefficient in enumerate(coefficients):
        result += coefficient * (x ** power)
    return float(result)


def format_poly(coefficients):
    terms = []
    for power, coefficient in enumerate(coefficients):
        if abs(coefficient) < 1e-12:
            continue
        if power == 0:
            terms.append(f"{coefficient:.6g}")
        elif power == 1:
            terms.append(f"{coefficient:+.6g}x")
        else:
            terms.append(f"{coefficient:+.6g}x^{power}")
    return "P(x) = " + (" ".join(terms) if terms else "0")


def interpolation_coefficients(X, Y):
    degree = len(X) - 1
    coeff_desc = np.polyfit(X, Y, degree)
    return coeff_desc[::-1]


def lagrange_value(x, X, Y):
    result = 0.0
    n = len(X) - 1
    for i in range(n + 1):
        Li = 1.0
        for j in range(n + 1):
            if j != i:
                Li *= (x - X[j]) / (X[i] - X[j])
        result += Y[i] * Li
    return float(result)


def divided_differences(X, Y):
    n = len(X) - 1
    table = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        table[i, 0] = Y[i]
    for j in range(1, n + 1):
        for i in range(n + 1 - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (X[i + j] - X[i])
    return table


def newton_value(x, X, table):
    result = table[0, 0]
    product = 1.0
    for j in range(1, len(X)):
        product *= x - X[j - 1]
        result += table[0, j] * product
    return float(result)


def normal_system(X, Y, degree):
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)
    for row in range(degree + 1):
        for col in range(degree + 1):
            A[row, col] = np.sum(X ** (row + col))
        B[row] = np.sum(Y * (X ** row))
    return A, B


def least_squares(X, Y, degree):
    A, B = normal_system(X, Y, degree)
    coefficients = np.linalg.solve(A, B)
    return coefficients, A, B


def continuous_least_squares(f_txt, a, b, degree):
    x = sp.Symbol("x")
    expr = sp.sympify(str(f_txt).replace("^", "**"))
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)
    for row in range(degree + 1):
        for col in range(degree + 1):
            A[row, col] = float(sp.integrate(x ** (row + col), (x, a, b)))
        B[row] = float(sp.integrate(expr * x ** row, (x, a, b)))
    coefficients = np.linalg.solve(A, B)
    return coefficients, A, B, str(expr)


def continuous_error(f_txt, a, b, coefficients):
    x = sp.Symbol("x")
    expr = sp.sympify(str(f_txt).replace("^", "**"))
    poly = sum(float(c) * x ** i for i, c in enumerate(coefficients))
    return float(sp.integrate((expr - poly) ** 2, (x, a, b)))


def sse(X, Y, coefficients):
    total = 0.0
    for x, y in zip(X, Y):
        total += (y - poly_value(coefficients, x)) ** 2
    return float(total)


def curve(X, func, count=180):
    margin = max(1.0, float(np.max(X) - np.min(X)) * 0.2)
    xs = np.linspace(float(np.min(X) - margin), float(np.max(X) + margin), count)
    ys = [float(func(x)) for x in xs]
    return xs.tolist(), ys


def eval_expr(expr, **values):
    env = dict(_ALLOWED_FUNCS)
    env.update(values)
    return eval(str(expr).replace("^", "**"), {"__builtins__": None}, env)


# ══════════════════════════════════════════════════════════
#  DESCENTE DE GRADIENT
# ══════════════════════════════════════════════════════════

def gradient_1d(f_txt, df_txt, x0, alpha, eps, max_iter):
    x = float(x0)
    rows = []
    for k in range(max_iter + 1):
        fx   = float(eval_expr(f_txt,  x=x))
        grad = float(eval_expr(df_txt, x=x))
        x_new = x - alpha * grad
        err   = abs(x_new - x)
        rows.append({"iter": k, "x": x, "fx": fx, "grad": grad, "err": err})
        if abs(grad) < eps or err < eps:
            x = x_new
            break
        if not math.isfinite(x_new):
            raise ValueError("Divergence détectée : valeur non finie.")
        x = x_new
    return x, float(eval_expr(f_txt, x=x)), rows


def gradient_2d(f_txt, dfdx_txt, dfdy_txt, x0, y0, alpha, eps, max_iter):
    x = float(x0)
    y = float(y0)
    rows = []
    for k in range(max_iter + 1):
        fx   = float(eval_expr(f_txt,    x=x, y=y))
        gx   = float(eval_expr(dfdx_txt, x=x, y=y))
        gy   = float(eval_expr(dfdy_txt, x=x, y=y))
        grad_norm = float(np.sqrt(gx ** 2 + gy ** 2))
        x_new = x - alpha * gx
        y_new = y - alpha * gy
        err   = float(np.sqrt((x_new - x) ** 2 + (y_new - y) ** 2))
        rows.append({"iter": k, "x": x, "y": y, "fx": fx,
                     "gx": gx, "gy": gy, "grad_norm": grad_norm, "err": err})
        if grad_norm < eps or err < eps:
            x, y = x_new, y_new
            break
        if not math.isfinite(x_new) or not math.isfinite(y_new):
            raise ValueError("Divergence détectée : valeur non finie.")
        x, y = x_new, y_new
    return x, y, float(eval_expr(f_txt, x=x, y=y)), rows
