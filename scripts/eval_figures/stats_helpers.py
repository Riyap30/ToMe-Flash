import math

import numpy as np

Z95 = 1.96


def ci95(values):
    """Return (mean, half_width) for 95% CI; uses scipy t if available."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n))
    try:
        from scipy.stats import t as _t

        half = float(_t.ppf(0.975, df=n - 1) * se)
    except ImportError:
        half = Z95 * se  # z-approx; fine for n > 30
    return mean, half


def prop_diff_ci95_half(p_base, p_cond, n):
    """95% CI half-width for difference of two independent proportions (pp)."""
    se = math.sqrt(p_base * (1 - p_base) / n + p_cond * (1 - p_cond) / n)
    return Z95 * se * 100


def prop_ci95_half(p, n):
    """95% CI half-width for a single proportion (in pp)."""
    return Z95 * math.sqrt(p * (1 - p) / n) * 100


def hedges_g(arr_a, arr_b):
    """Hedges' g: (mean_a - mean_b) / pooled SD, bias-corrected."""
    a = np.asarray(arr_a, float)
    b = np.asarray(arr_b, float)
    na, nb = len(a), len(b)
    s_p = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if s_p == 0:
        return 0.0
    g_raw = (a.mean() - b.mean()) / s_p
    correction = 1 - 3 / (4 * (na + nb - 2) - 1)
    return g_raw * correction


def cohen_h(p1, p2):
    """Cohen's h effect size for two proportions (arcsine transformation)."""
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return phi1 - phi2


def bh_correct(p_values):
    """Benjamini-Hochberg FDR correction; returns array of adjusted p-values."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, float)
    for i, idx in enumerate(order):
        adj[idx] = min(p[idx] * m / (i + 1), 1.0)
    for i in range(m - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return adj


def stars(p):
    if p < 1e-10:
        return "★★★"
    if p < 0.001:
        return "★★"
    if p < 0.05:
        return "★"
    return "n.s."


def fmt_p(p):
    """Consistent p-value formatting for tables and annotations."""
    p = float(p)
    if p < 1e-300:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"


def pareto_frontier(xs, ys):
    """Return indices of Pareto-optimal points (maximize both x and y)."""
    pts = sorted(zip(xs, ys, range(len(xs))), key=lambda val: -val[0])
    pareto = []
    max_y = -float("inf")
    for _, y, idx in pts:
        if y > max_y:
            pareto.append(idx)
            max_y = y
    return pareto
