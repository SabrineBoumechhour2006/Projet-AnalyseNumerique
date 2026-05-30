from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import sympy as sp
import math

# ── Utilitaires communs ────────────────────────────────────
from utils import parse_expr, safe_eval

# ── Axe 1 : méthodes non-linéaires ────────────────────────
from Axe1.methodes import run_bisection, run_newton, run_point_fixe

# ── Axe 2 : méthodes indirectes (Jacobi, Gauss-Seidel) ────
from Axe2.methodes_indirectes.methodes import (
    run_jacobi, run_gauss_seidel, _residu as _residu_indirect
)

# ── Axe 2 : méthodes directes (Gauss, LU, Cholesky) ───────
from Axe2.methodes_directes.methodes import (
    gauss, lu, cholesky, _residu as _residu_direct
)

# ── Axe 3 : interpolation, approximation, gradient ────────
from Axe3.methodes import (
    parse_values       as _a3_parse_values,
    poly_value         as _a3_poly_value,
    format_poly        as _a3_format_poly,
    interpolation_coefficients as _a3_interpolation_coefficients,
    lagrange_value     as _a3_lagrange_value,
    divided_differences as _a3_divided_differences,
    newton_value       as _a3_newton_value,
    least_squares      as _a3_least_squares,
    continuous_least_squares as _a3_continuous_least_squares,
    continuous_error   as _a3_continuous_error,
    sse                as _a3_sse,
    curve              as _a3_curve,
    eval_expr          as _a3_eval_expr,
    gradient_1d        as _a3_gradient_1d,
    gradient_2d        as _a3_gradient_2d,
)

app = Flask(__name__, static_folder=".", static_url_path="")


#  AXE 1 — Dichotomie
@app.route("/api/bisection", methods=["POST"])
def api_bisection():
    try:
        d   = request.json
        f, *_ = parse_expr(d["fx"])
        a, b  = float(d["a"]), float(d["b"])
        eps   = float(d.get("eps", 1e-4))
        racine, rows = run_bisection(f, a, b, eps)
        return jsonify({
            "racine":     racine,
            "iterations": len(rows),
            "rows":       rows,
            "f_racine":   safe_eval(f, racine),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 1 — NEWTON-RAPHSON
@app.route("/api/newton", methods=["POST"])
def api_newton():
    try:
        d   = request.json
        f, df, ddf, df_str, ddf_str = parse_expr(d["fx"])
        a, b = float(d.get("a", 0)), float(d.get("b", 10))
        x0   = float(d.get("x0", (a + b) / 2))
        eps  = float(d.get("eps", 1e-4))
        racine, rows = run_newton(f, df, ddf, a, b, x0, eps)
        return jsonify({
            "racine":     racine,
            "iterations": len(rows),
            "rows":       rows,
            "df_str":     df_str,
            "ddf_str":    ddf_str,
            "f_racine":   safe_eval(f, racine),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 1 — POINT FIXE
@app.route("/api/point_fixe", methods=["POST"])
def api_point_fixe():
    try:
        d   = request.json
        x   = sp.Symbol('x')
        g_expr  = sp.sympify(d["gx"].replace('^', '**'))
        dg_expr = sp.diff(g_expr, x)
        g   = sp.lambdify(x, g_expr,  modules=['numpy'])
        dg  = sp.lambdify(x, dg_expr, modules=['numpy'])
        x0  = float(d.get("x0", 1.0))
        eps = float(d.get("eps", 1e-4))
        racine, rows = run_point_fixe(g, dg, x0, eps)
        return jsonify({
            "racine":     racine,
            "iterations": len(rows),
            "rows":       rows,
            "g_str":      str(g_expr),
            "dg_str":     str(dg_expr),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 1 — COMPARAISON DES 3 MÉTHODES NL
@app.route("/api/comparaison_nl", methods=["POST"])
def api_comparaison_nl():
    try:
        d    = request.json
        f, df, ddf, df_str, _ = parse_expr(d["fx"])
        a, b = float(d.get("a", 0)), float(d.get("b", 10))
        x0   = float(d.get("x0", (a + b) / 2))
        eps  = float(d.get("eps", 1e-4))
        result = {}

        try:
            r, rows = run_bisection(f, a, b, eps)
            result["bisection"] = { "racine": r, "iterations": len(rows),
                                    "err_final": rows[-1]["err"] if rows else None }
        except Exception as e:
            result["bisection"] = { "error": str(e) }

        try:
            r, rows = run_newton(f, df, ddf, a, b, x0, eps)
            result["newton"] = { "racine": r, "iterations": len(rows),
                                 "err_final": rows[-1]["err"] if rows else None }
        except Exception as e:
            result["newton"] = { "error": str(e) }

        if d.get("gx"):
            try:
                x_sym = sp.Symbol('x')
                g_expr  = sp.sympify(d["gx"].replace('^', '**'))
                dg_expr = sp.diff(g_expr, x_sym)
                g  = sp.lambdify(x_sym, g_expr,  modules=['numpy'])
                dg = sp.lambdify(x_sym, dg_expr, modules=['numpy'])
                r, rows = run_point_fixe(g, dg, x0, eps)
                result["point_fixe"] = { "racine": r, "iterations": len(rows),
                                         "err_final": rows[-1]["err"] if rows else None }
            except Exception as e:
                result["point_fixe"] = { "error": str(e) }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#  AXE 1 — RECOMMANDATION
@app.route("/api/recommandation", methods=["POST"])
def api_recommandation():
    d = request.json
    derivable  = d.get("derivable",  "oui")
    intervalle = d.get("intervalle", "oui")
    precision  = d.get("precision",  "moyenne")
    gx         = d.get("gx",        "non")

    if derivable == "oui" and precision == "haute":
        methode = "Newton-Raphson"
        raison  = "Dérivée disponible + haute précision → convergence quadratique optimale."
        alts    = ["Bisection (si x₀ difficile)"]
    elif intervalle == "oui" and derivable != "oui":
        methode = "Bisection"
        raison  = "Intervalle de signe disponible, sans dérivée → garantie de convergence."
        alts    = ["Newton (si dérivée calculable)"]
    elif gx == "oui":
        methode = "Point fixe"
        raison  = "Réécriture x = g(x) possible → convergence si g contractante."
        alts    = ["Bisection (si intervalle trouvé)"]
    else:
        methode = "Newton-Raphson"
        raison  = "Cas général avec f dérivable → meilleur compromis vitesse/précision."
        alts    = ["Bisection (plus robuste)", "Point fixe (si g constructible)"]

    return jsonify({ "methode": methode, "raison": raison, "alternatives": alts })


#  AXE 1 — ANALYSE DE LA FONCTION
@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    try:
        d   = request.json
        f, df, ddf, df_str, ddf_str = parse_expr(d["fx"])
        a, b = float(d["a"]), float(d["b"])
        N    = 500
        xs   = np.linspace(a, b, N + 1)
        vals    = [safe_eval(f, xi) for xi in xs]
        discos  = [round(float(xs[i]), 4) for i, v in enumerate(vals) if v is None]
        continue_ = len(discos) == 0
        fa = safe_eval(f, a)
        fb = safe_eval(f, b)
        tvi = (fa is not None and fb is not None and fa * fb < 0)
        racines = []
        for i in range(N):
            v1, v2 = vals[i], vals[i+1]
            if v1 is not None and v2 is not None and v1 * v2 < 0:
                racines.append([round(float(xs[i]), 4), round(float(xs[i+1]), 4)])
        xm = (a + b) / 2
        df_mid  = safe_eval(df, xm)
        ddf_mid = safe_eval(ddf, xm)
        return jsonify({
            "continue":  continue_,
            "discos":    discos[:5],
            "tvi":       tvi,
            "fa": fa, "fb": fb,
            "racines":   racines,
            "df_str":    df_str,
            "ddf_str":   ddf_str,
            "df_mid":    df_mid,
            "ddf_mid":   ddf_mid,
            "xm":        xm,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 2 -INDIRECT- JACOBI
@app.route("/api/jacobi", methods=["POST"])
def api_jacobi():
    try:
        d    = request.json
        A, b = d["A"], d["b"]
        tol  = float(d.get("tol", 1e-4))
        sol, hist, errs, it = run_jacobi(A, b, tol)
        return jsonify({ "solution": sol.tolist(), "iterations": it,
                         "history": hist, "errors": errs,
                         "residue": _residu_indirect(A, sol, b), "tol": tol })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 2 -INDIRECT- GAUSS-SEIDEL
@app.route("/api/gauss_seidel", methods=["POST"])
def api_gauss_seidel():
    try:
        d    = request.json
        A, b = d["A"], d["b"]
        tol  = float(d.get("tol", 1e-4))
        sol, hist, errs, it = run_gauss_seidel(A, b, tol)
        return jsonify({ "solution": sol.tolist(), "iterations": it,
                         "history": hist, "errors": errs,
                         "residue": _residu_indirect(A, sol, b), "tol": tol })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 2 — COMPARAISON Jacobi vs Gauss-Seidel
@app.route("/api/comparaison", methods=["POST"])
def api_comparaison():
    try:
        d    = request.json
        A, b = d["A"], d["b"]
        tol  = float(d.get("tol", 1e-4))
        sol_j,  _, err_j,  it_j  = run_jacobi(A, b, tol)
        sol_gs, _, err_gs, it_gs = run_gauss_seidel(A, b, tol)
        winner = ("Gauss-Seidel" if it_gs < it_j
                  else ("Jacobi" if it_j < it_gs else "Égalité"))
        return jsonify({
            "jacobi":       { "solution": sol_j.tolist(),  "iterations": it_j,
                              "errors": err_j,  "residue": _residu_indirect(A, sol_j, b)  },
            "gauss_seidel": { "solution": sol_gs.tolist(), "iterations": it_gs,
                              "errors": err_gs, "residue": _residu_indirect(A, sol_gs, b) },
            "winner": winner, "tol": tol,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 2 — NORMES & DÉTERMINANT
@app.route("/api/autre_option", methods=["POST"])
def api_autre_option():
    try:
        A   = np.array(request.json["A"], dtype=float)
        det = float(np.linalg.det(A))
        return jsonify({
            "det":       det,
            "singular":  abs(det) < 1e-12,
            "norme_1":   float(np.max(np.sum(np.abs(A), axis=0))),
            "norme_inf": float(np.max(np.sum(np.abs(A), axis=1))),
            "norme_F":   float(np.sqrt(np.sum(A**2))),
            "norme_2":   float(np.linalg.norm(A, ord=2)),
            "cond_1":    float(np.linalg.cond(A, p=1)),
            "cond_2":    float(np.linalg.cond(A, p=2)),
            "cond_inf":  float(np.linalg.cond(A, p=np.inf)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 2 — MÉTHODES DIRECTES (Gauss, LU, Cholesky)
@app.route("/api/direct", methods=["POST"])
def api_direct():
    try:
        d      = request.json
        A      = np.array(d["A"], dtype=float)
        b      = np.array(d["b"], dtype=float)
        method = d.get("method", "gauss")
        fns    = {"gauss": gauss, "lu": lu, "cholesky": cholesky}
        if method not in fns:
            return jsonify({"error": "Méthode inconnue"}), 400
        sol = fns[method](A, b)
        return jsonify({ "solution": sol.tolist(), "residue": _residu_direct(A, sol, b) })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 3 — INTERPOLATION
@app.route("/api/axe3_interpolation", methods=["POST"])
def api_axe3_interpolation():
    try:
        d = request.json
        X = _a3_parse_values(d.get("x", "0,1,2"))
        Y = _a3_parse_values(d.get("y", "1,-1,-3.333333"))
        if len(X) != len(Y):
            raise ValueError("X et Y doivent avoir la meme taille.")
        if len(X) < 2:
            raise ValueError("Il faut au moins deux points.")
        if len(np.unique(X)) != len(X):
            raise ValueError("Les valeurs de X doivent etre distinctes.")
        table = _a3_divided_differences(X, Y)
        coefficients = _a3_interpolation_coefficients(X, Y)
        xs, y_lagrange = _a3_curve(X, lambda t: _a3_lagrange_value(t, X, Y))
        _, y_newton = _a3_curve(X, lambda t: _a3_newton_value(t, X, table))
        ecarts = [abs(a - b) for a, b in zip(y_lagrange, y_newton)]
        return jsonify({
            "degree": len(X) - 1,
            "points": [{"x": float(x), "y": float(y)} for x, y in zip(X, Y)],
            "polynomial": _a3_format_poly(coefficients),
            "table": table.tolist(),
            "curve": {"x": xs, "lagrange": y_lagrange, "newton": y_newton},
            "ecart_max": float(max(ecarts) if ecarts else 0.0),
            "recommendation": "Lagrange et Newton donnent le meme polynome interpolateur. Lagrange est plus direct ; Newton est plus avantageux si on ajoute des points grace aux differences divisees.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 3 — APPROXIMATION (moindres carrés)
@app.route("/api/axe3_approximation", methods=["POST"])
def api_axe3_approximation():
    try:
        d    = request.json
        mode = d.get("mode_approx", "discrete")

        if mode == "comparison":
            X     = _a3_parse_values(d.get("x", "0,1,2,3,4"))
            Y     = _a3_parse_values(d.get("y", "1,2,2.8,3.6,5.1"))
            f_txt = d.get("f_approx", "x^2+sin(x)")
            a     = float(d.get("a", float(np.min(X))))
            b     = float(d.get("b", float(np.max(X))))
            degree = int(d.get("degree", 1))
            n = len(X) - 1
            if len(X) != len(Y): raise ValueError("X et Y doivent avoir la meme taille.")
            if len(X) < 3: raise ValueError("Il faut au moins trois points.")
            if degree < 0 or degree >= n: raise ValueError("Choisir s < n.")
            if b <= a: raise ValueError("Il faut choisir a < b.")
            coeff_discrete, A_discrete, B_discrete = _a3_least_squares(X, Y, degree)
            err_discrete_points = _a3_sse(X, Y, coeff_discrete)
            coeff_continuous, A_continuous, B_continuous, expr = _a3_continuous_least_squares(f_txt, a, b, degree)
            err_continuous_integral = _a3_continuous_error(f_txt, a, b, coeff_continuous)
            err_discrete_integral   = _a3_continuous_error(f_txt, a, b, coeff_discrete)
            err_continuous_points   = _a3_sse(X, Y, coeff_continuous)
            xs = np.linspace(min(float(np.min(X)), a), max(float(np.max(X)), b), 200)
            f  = sp.lambdify(sp.Symbol("x"), sp.sympify(str(f_txt).replace("^", "**")), modules=["numpy"])
            ys_f          = [float(f(float(x))) for x in xs]
            ys_discrete   = [_a3_poly_value(coeff_discrete,   float(x)) for x in xs]
            ys_continuous = [_a3_poly_value(coeff_continuous, float(x)) for x in xs]
            return jsonify({
                "mode": "comparison", "degree": degree, "function": expr,
                "interval": [a, b],
                "points": [{"x": float(x), "y": float(y)} for x, y in zip(X, Y)],
                "discrete": {
                    "polynomial": _a3_format_poly(coeff_discrete),
                    "coefficients": coeff_discrete.tolist(),
                    "A": A_discrete.tolist(), "B": B_discrete.tolist(),
                    "error_points": err_discrete_points, "error_integral": err_discrete_integral,
                },
                "continuous": {
                    "polynomial": _a3_format_poly(coeff_continuous),
                    "coefficients": coeff_continuous.tolist(),
                    "A": A_continuous.tolist(), "B": B_continuous.tolist(),
                    "error_integral": err_continuous_integral, "error_points": err_continuous_points,
                },
                "curve": {"x": xs.tolist(), "f": ys_f, "discrete": ys_discrete, "continuous": ys_continuous},
                "recommendation": "Le cas discret optimise l'erreur sur les points donnes, le cas continu optimise l'erreur integrale sur tout l'intervalle.",
            })

        if mode == "continuous":
            f_txt  = d.get("f_approx", "x^2+sin(x)")
            a      = float(d.get("a", 0))
            b      = float(d.get("b", 2))
            degree = int(d.get("degree", 1))
            if b <= a: raise ValueError("Il faut choisir a < b.")
            if degree < 0: raise ValueError("Le degre doit etre positif ou nul.")
            coefficients, A, B, expr = _a3_continuous_least_squares(f_txt, a, b, degree)
            error = _a3_continuous_error(f_txt, a, b, coefficients)
            xs  = np.linspace(a, b, 180)
            f   = sp.lambdify(sp.Symbol("x"), sp.sympify(str(f_txt).replace("^", "**")), modules=["numpy"])
            ys_f = [float(f(x)) for x in xs]
            ys_p = [_a3_poly_value(coefficients, x) for x in xs]
            return jsonify({
                "mode": "continuous", "degree": degree, "function": expr,
                "interval": [a, b],
                "coefficients": coefficients.tolist(),
                "polynomial": _a3_format_poly(coefficients),
                "A": A.tolist(), "B": B.tolist(), "error": error,
                "curve": {"x": xs.tolist(), "f": ys_f, "y": ys_p},
                "recommendation": "Les moindres carres continus approximent une fonction sur [a,b] en minimisant l'integrale de (f(x)-P(x))^2.",
            })

        # mode == "discrete" (défaut)
        X      = _a3_parse_values(d.get("x", "0,1,2,3,4"))
        Y      = _a3_parse_values(d.get("y", "1,2,2.8,3.6,5.1"))
        degree = int(d.get("degree", 1))
        n = len(X) - 1
        if len(X) != len(Y): raise ValueError("X et Y doivent avoir la meme taille.")
        if len(X) < 3: raise ValueError("Il faut au moins trois points.")
        if degree < 0 or degree >= n: raise ValueError("Choisir s < n.")
        coefficients, A, B = _a3_least_squares(X, Y, degree)
        error = _a3_sse(X, Y, coefficients)
        xs, ys = _a3_curve(X, lambda t: _a3_poly_value(coefficients, t))
        comparisons = []
        for deg in range(1, n):
            coeff_test, _, _ = _a3_least_squares(X, Y, deg)
            comparisons.append({"degree": deg, "error": _a3_sse(X, Y, coeff_test), "poly": _a3_format_poly(coeff_test)})
        best = min(comparisons, key=lambda r: r["error"]) if comparisons else None
        return jsonify({
            "mode": "discrete", "degree": degree,
            "coefficients": coefficients.tolist(),
            "polynomial": _a3_format_poly(coefficients),
            "A": A.tolist(), "B": B.tolist(), "error": error,
            "comparisons": comparisons, "best": best,
            "curve": {"x": xs, "y": ys},
            "recommendation": "Les moindres carres minimisent la somme des carres des ecarts : le polynome ne passe pas forcement par tous les points.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 3 — DESCENTE DE GRADIENT
@app.route("/api/axe3_gradient", methods=["POST"])
def api_axe3_gradient():
    try:
        d        = request.json
        mode     = d.get("mode", "1d")
        alpha    = float(d.get("alpha", 0.1))
        eps      = float(d.get("eps", 1e-5))
        max_iter = int(d.get("max_iter", 300))

        if mode == "2d":
            f_txt    = d.get("f",    "x^2+2*y^2+x*y+x-y+30")
            dfdx_txt = d.get("dfdx", "2*x+y+1")
            dfdy_txt = d.get("dfdy", "4*y+x-1")
            x0 = float(d.get("x0", 3))
            y0 = float(d.get("y0", 3))
            x_star, y_star, f_star, rows = _a3_gradient_2d(f_txt, dfdx_txt, dfdy_txt, x0, y0, alpha, eps, max_iter)
            return jsonify({
                "mode": "2d", "x_star": x_star, "y_star": y_star, "f_star": f_star,
                "iterations": len(rows) - 1, "rows": rows,
                "trajectory": {"x": [r["x"] for r in rows], "y": [r["y"] for r in rows], "z": [r["fx"] for r in rows]},
                "recommendation": "En deux variables, le minimum est approche quand la norme du gradient devient proche de zero.",
            })

        f_txt  = d.get("f",  "x^2+2*x+1")
        df_txt = d.get("df", "2*x+2")
        x0     = float(d.get("x0", 5))
        x_star, f_star, rows = _a3_gradient_1d(f_txt, df_txt, x0, alpha, eps, max_iter)
        hx = [r["x"] for r in rows]
        hy = [r["fx"] for r in rows]
        xs = np.linspace(min(hx) - 2, max(hx) + 2, 220)
        ys = [float(_a3_eval_expr(f_txt, x=float(x))) for x in xs]
        return jsonify({
            "mode": "1d", "x_star": x_star, "f_star": f_star,
            "iterations": len(rows) - 1, "rows": rows,
            "curve": {"x": xs.tolist(), "y": ys, "hx": hx, "hy": hy},
            "recommendation": "x(k+1) = x(k) - alpha*f'(x(k)). Le minimum est approche quand f'(x) devient proche de zero.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 3 — INTERP vs APPROX
@app.route("/api/axe3_interp_approx", methods=["POST"])
def api_axe3_interp_approx():
    try:
        d      = request.json
        X      = _a3_parse_values(d.get("x", "0,1,2,3,4"))
        Y      = _a3_parse_values(d.get("y", "1,2,2.8,3.6,5.1"))
        degree = int(d.get("degree", 1))
        n = len(X) - 1
        if len(X) != len(Y): raise ValueError("X et Y doivent avoir la meme taille.")
        if len(np.unique(X)) != len(X): raise ValueError("Les valeurs de X doivent etre distinctes.")
        if degree < 0 or degree >= n: raise ValueError("Choisir s < n.")
        table = _a3_divided_differences(X, Y)
        interp_coeffs = _a3_interpolation_coefficients(X, Y)
        coeffs, _, _  = _a3_least_squares(X, Y, degree)
        xs, y_interp  = _a3_curve(X, lambda t: _a3_newton_value(t, X, table))
        _, y_approx   = _a3_curve(X, lambda t: _a3_poly_value(coeffs, t))
        approximation_error = _a3_sse(X, Y, coeffs)
        rows = []
        for x, y in zip(X, Y):
            pi = _a3_newton_value(x, X, table)
            pa = _a3_poly_value(coeffs, x)
            rows.append({"x": float(x), "y": float(y),
                         "interpolation": pi, "approximation": pa,
                         "err_interp": abs(y - pi), "err_approx": abs(y - pa)})
        return jsonify({
            "degree_interpolation": n, "degree_approximation": degree,
            "interp_polynomial": _a3_format_poly(interp_coeffs),
            "approx_polynomial": _a3_format_poly(coeffs),
            "interpolation_error": 0.0, "approximation_error": approximation_error,
            "rows": rows,
            "points": [{"x": float(x), "y": float(y)} for x, y in zip(X, Y)],
            "curve": {"x": xs, "interpolation": y_interp, "approximation": y_approx},
            "recommendation": "Interpolation : utile si les points sont exacts. Approximation : utile si les points sont experimentaux.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  AXE 3 — COMPARAISON GÉNÉRALE
@app.route("/api/axe3_comparaison", methods=["POST"])
def api_axe3_comparaison():
    try:
        d      = request.json
        X      = _a3_parse_values(d.get("x", "0,1,2,3,4"))
        Y      = _a3_parse_values(d.get("y", "1,2,2.8,3.6,5.1"))
        degree = min(int(d.get("degree", 1)), len(X) - 2)
        table  = _a3_divided_differences(X, Y)
        interp_err = max(abs(_a3_lagrange_value(x, X, Y) - _a3_newton_value(x, X, table)) for x in X)
        coeffs, _, _ = _a3_least_squares(X, Y, degree)
        approx_err   = _a3_sse(X, Y, coeffs)
        return jsonify({
            "items": [
                {"name": "Interpolation",       "criterion": "Reproduire exactement un nuage de points", "value": float(interp_err), "advice": "Lagrange est direct ; Newton est pratique si on ajoute des points."},
                {"name": "Moindres carres",     "criterion": "Approximer une tendance",                  "value": float(approx_err), "advice": "A choisir pour un nuage bruite ou nombreux."},
                {"name": "Descente de gradient","criterion": "Minimiser une fonction",                   "value": None,              "advice": "A choisir pour une optimisation iterative ; depend du pas alpha."},
            ],
            "recommendation": "Interpolation pour respecter les points, moindres carres pour approximer une tendance, descente de gradient pour minimiser une fonction.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  PAGE PRINCIPALE
@app.route("/")
def index():
    return send_from_directory(".", "numanalyse.html")


#  LANCEMENT
if __name__ == "__main__":
    print("=" * 55)
    print("  NumAnalyse — Flask backend")
    print("  Installer : pip install flask numpy sympy")
    print("  Lancer    : python app.py")
    print("  URL       : http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)
