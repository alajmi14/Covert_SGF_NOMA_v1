"""Residual-jamming analysis (BS unaware of the residual) -- closed forms (Theorem 5, Proposition 3, Corollary 5).

W = zeta P_J |h_J|^2 / sigma^2 = c U E,  U ~ U[0,1], E ~ Exp(1),  c = zeta rho_J Om_JB (residual in noise units),
ccdf  G(w) = P(W > w) = e^{-w/c} - (w/c) E1(w/c) = int_0^1 exp(-w/(u c)) du,   E[W] = c/2.

GB user:  outage  <=>  W > gap_b,  gap_b = tau - rx*      (tau = rho_b|h_b|^2/delta_b - 1; rx* = 0 if nobody admitted)
GF user:  success <=>  admitted and W <= min(gap_b, gap_g),  gap_g = rx*/delta_g - 1.
E[G(gap)] = int_0^1 Lambda(1/(u c)) du with Lambda(s) = E[e^{-s gap}] a finite sum of rational (GF) or
rational x e^{-s delta_g} (GB) functions of s; the u-integral is then elementary (logarithms, E1).

Notation as in Theorem 3:  theta = delta_g/rho_g,  x0 = delta_b(1+delta_g)/rho_b,  a = rho_b/(delta_b rho_g),
c_k = 1/Om_b + k a/Om_g,  phi = 1 - e^{-theta/Om_g};  b1 = delta_g a/(1+delta_g),  b2 = a/(1+delta_g).
"""
import numpy as np
from scipy.special import exp1, comb
from model import gf_outage, gb_outage


def G(w, c):
    w = np.asarray(w, float)
    a = w / c
    out = np.ones_like(a)
    pos = a > 0
    out[pos] = np.exp(-a[pos]) - a[pos] * exp1(a[pos])
    return out


def _consts(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g):
    db, dg = 2.0 ** R_b - 1.0, 2.0 ** R_g - 1.0
    theta = dg / rho_g
    xa = db / rho_b
    x0 = db * (1.0 + dg) / rho_b
    phi = 1.0 - np.exp(-theta / Om_g)
    a = rho_b / (db * rho_g)
    ck = lambda k: 1.0 / Om_b + k * a / Om_g
    return db, dg, theta, xa, x0, phi, a, ck


def _terms(K):
    """f_M(m|x) = (K/Om_g) sum coef e^{-p psi/Om_g} e^{-q m/Om_g},  coef = C(K-1,j) C(j,l) (-1)^l, p = j-l, q = l+1."""
    return [(comb(K - 1, j) * comb(j, l) * (-1) ** l, j - l, l + 1, j) for j in range(K) for l in range(j + 1)]


def _vartheta(D, E, c):
    """int_0^1 du / (D + E/(u c)) = (1/D) [1 - (E/(D c)) ln(1 + D c/E)],  D, E > 0."""
    z = D * c / E
    return (1.0 / D) * (1.0 - np.log1p(z) / z)


def _xi(ck, dg, rg_a, c):
    """int_0^1 e^{-dg/(u c)} / (ck + rg_a/(u c)) du
       = (e^{-al} - al E1(al))/ck - (rg_a/(ck^2 c)) [E1(al) - e^{be} E1(al + be)],  al = dg/c, be = dg ck/rg_a."""
    al = dg / c
    be = dg * ck / rg_a
    return (np.exp(-al) - al * exp1(al)) / ck - (rg_a / (ck ** 2 * c)) * (exp1(al) - np.exp(be) * exp1(al + be))


# ---------------------------------------------------------------- Laplace transforms (for checks)
def lambda_g(s, rho_b, rho_g, Om_b, Om_g, K, R_b, R_g):
    """Lambda_g(s) = E[e^{-s min(gap_b,gap_g)} 1{admitted}] in closed form."""
    s = np.asarray(s, float)
    db, dg, theta, xa, x0, phi, a, ck = _consts(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g)
    b1, b2 = dg * a / (1.0 + dg), a / (1.0 + dg)
    tot = 0.0
    for coef, p, q, j in _terms(K):
        gam = s * rho_g / dg + q / Om_g
        beta = q / Om_g - s * rho_g
        tot = tot + coef * np.exp(-(j + 1) * theta / Om_g) * (
            b1 / (ck(p) * (ck(p) + gam * b1)) + b2 / (ck(j + 1) * (ck(j + 1) - beta * b2)))
    return (K * np.exp(-x0 / Om_b) / (Om_g * Om_b)) * tot


def lambda_b(s, rho_b, rho_g, Om_b, Om_g, K, R_b, R_g):
    """Lambda_b(s) = E[e^{-s gap_b} 1{rho_b|h_b|^2 >= delta_b}] in closed form."""
    s = np.asarray(s, float)
    db, dg, theta, xa, x0, phi, a, ck = _consts(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g)
    lam1 = (np.exp(-xa / Om_b) - np.exp(-x0 / Om_b - s * dg)) / (Om_b * (1.0 / Om_b + s * rho_g * a))
    lam2 = sum(comb(K, j) * phi ** (K - j) * np.exp(-x0 / Om_b - j * theta / Om_g - s * dg)
               / (Om_b * (ck(j) + s * rho_g * a)) for j in range(K + 1))
    tot = 0.0
    for coef, p, q, j in _terms(K):
        beta = q / Om_g - s * rho_g
        tot = tot + coef * np.exp(-(j + 1) * theta / Om_g) * a / (ck(j + 1) * (ck(j + 1) - beta * a))
    lam3 = (K * np.exp(-x0 / Om_b) / (Om_g * Om_b)) * tot
    return lam1 + lam2 + lam3


# ---------------------------------------------------------------- outage probabilities (closed form)
def gf_outage_residual(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g, c):
    """Theorem: GF outage with residual jamming, BS unaware.  c = zeta rho_J Om_JB."""
    db, dg, theta, xa, x0, phi, a, ck = _consts(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g)
    b1, b2 = dg * a / (1.0 + dg), a / (1.0 + dg)
    tot = 0.0
    for coef, p, q, j in _terms(K):
        D1, E1 = ck(p) + q * b1 / Om_g, rho_g * b1 / dg          # ck(p) + gam b1 = D1 + s E1
        D2, E2 = ck(j + 1) - q * b2 / Om_g, rho_g * b2            # ck(j+1) - beta b2 = D2 + s E2
        tot = tot + coef * np.exp(-(j + 1) * theta / Om_g) * (
            b1 * _vartheta(D1, E1, c) / ck(p) + b2 * _vartheta(D2, E2, c) / ck(j + 1))
    return gf_outage(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g) + (K * np.exp(-x0 / Om_b) / (Om_g * Om_b)) * tot


def gb_outage_residual(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g, c):
    """Theorem: GB outage with residual jamming, BS unaware."""
    db, dg, theta, xa, x0, phi, a, ck = _consts(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g)
    rg_a = rho_g * a
    # piece 1: xa < x < x0, gap = tau:  (e^{-xa/Om_b} - e^{-x0/Om_b} e^{-s dg}) / (Om_b (1/Om_b + s rg_a))
    p1 = np.exp(-xa / Om_b) * _vartheta(1.0 / Om_b, rg_a, c) / Om_b - np.exp(-x0 / Om_b) * _xi(1.0 / Om_b, dg, rg_a, c) / Om_b
    # piece 2: x > x0, nobody admissible, gap = tau
    p2 = sum(comb(K, j) * phi ** (K - j) * np.exp(-x0 / Om_b - j * theta / Om_g) * _xi(ck(j), dg, rg_a, c) / Om_b
             for j in range(K + 1))
    # piece 3: admitted user at gain m, gap = rho_g (psi - m)
    tot = 0.0
    for coef, p, q, j in _terms(K):
        D, E = ck(j + 1) - q * a / Om_g, rho_g * a                 # ck(j+1) - beta a = D + s E
        tot = tot + coef * np.exp(-(j + 1) * theta / Om_g) * a * _vartheta(D, E, c) / ck(j + 1)
    p3 = (K * np.exp(-x0 / Om_b) / (Om_g * Om_b)) * tot
    return gb_outage(rho_b, Om_b, R_b) + p1 + p2 + p3


# ---------------------------------------------------------------- first-order sensitivities (closed form)
def residual_slopes(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g):
    """dP_b/dc|0 and dP_g/dc|0 = f_gap(0)/2 (Proposition)."""
    db, dg, theta, xa, x0, phi, a, ck = _consts(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g)
    f_tau0 = (db / (rho_b * Om_b)) * np.exp(-db / (rho_b * Om_b))                       # tau -> 0
    f_top = (K / (rho_g * Om_g)) * np.exp(-x0 / Om_b - theta / Om_g) / (Om_b * ck(1))   # selected user at the top
    S = sum(comb(K - 1, j) * phi ** (K - 1 - j) * np.exp(-x0 / Om_b - j * theta / Om_g) / (Om_b * ck(j))
            for j in range(K))
    f_bot = (K * theta / Om_g) * np.exp(-theta / Om_g) * S                               # selected user at the bottom
    return 0.5 * (f_tau0 + f_top), 0.5 * (f_top + f_bot), dict(f_tau0=float(f_tau0), f_top=float(f_top), f_bot=float(f_bot))


def curvature_logpower(rho_b, rho_g_u, Om_b, Om_g, K, R_b, R_g, h=0.02):
    """H = d^2 P_g^out / d(ln rho_g)^2 at the unconstrained optimum (from Theorem 3)."""
    f = lambda v: gf_outage(rho_b, rho_g_u * np.exp(v), Om_b, Om_g, K, R_b, R_g)
    return (f(h) - 2 * f(0.0) + f(-h)) / h ** 2


def pj_opt_residual(rho_J_th, c_th, Dg, H):
    """zeta-aware jamming power (first order): rho_J^opt = rho_J^th exp(-Dg c_th / H), c_th = zeta rho_J^th Om_JB."""
    v = Dg * c_th / H
    return rho_J_th * np.exp(-v), v
