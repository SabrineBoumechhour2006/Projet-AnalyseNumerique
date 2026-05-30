import sympy as sp
import math


def parse_expr(fx_str):
    """Parse une expression sympy depuis une chaîne."""
    x = sp.Symbol('x')
    expr = sp.sympify(fx_str.replace('^', '**'))
    f    = sp.lambdify(x, expr, modules=['numpy'])
    df_expr  = sp.diff(expr, x)
    ddf_expr = sp.diff(df_expr, x)
    df  = sp.lambdify(x, df_expr,  modules=['numpy'])
    ddf = sp.lambdify(x, ddf_expr, modules=['numpy'])
    return f, df, ddf, str(df_expr), str(ddf_expr)


def safe_eval(f, x):
    try:
        v = float(f(x))
        return v if math.isfinite(v) else None
    except Exception:
        return None
