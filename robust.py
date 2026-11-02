"""Robust design under warden-location uncertainty (Section V-D).

Positions in the plane: GF cluster p_g, jammer p_J, nominal warden position p_w; the true warden
position w lies in the disc W(r) = {w: |w - p_w| <= r}.  Covertness depends on w only through the
distance ratio  varrho(w) = |w - p_g| / |w - p_J|  (kappa = (P_J/P_g) varrho^alpha), so the robust
covertness constraint is  P_g <= P_J^max varrho_min(r)^alpha / kappa_eps  with
varrho_min(r) = min over W(r) of varrho(w), attained on the boundary circle (ln varrho is harmonic).

Closed forms (Proposition 4 (iii) and Proposition 5):
* collinear jammer at distance d = d_Jw from p_w on the segment towards p_g (D = d_gw):
      varrho_min = (D + r)/(d + r)  for r^2 <= dD,   (D - r)/(r - d)  for dD <= r^2 < D^2;
* optimal placement (inverse point of p_g w.r.t. the circle): varrho_min = D/r  (for all r < D).
"""
import numpy as np
from model import rho_eps, optimise, dep_avg


def varrho_boundary(D, pJ, r, n=3601):
    """min_phi |w - p_g|/|w - p_J| over the circle |w - p_w| = r; p_g = (0,0), p_w = (D,0), pJ = (x,y)."""
    phi = np.linspace(0.0, 2 * np.pi, n)
    w = np.stack([D + r * np.cos(phi), r * np.sin(phi)], 1)
    return np.min(np.hypot(w[:, 0], w[:, 1]) / np.hypot(w[:, 0] - pJ[0], w[:, 1] - pJ[1]))


def varrho_disc(D, pJ, r, n=401):
    """brute-force min over the whole disc (check of the boundary property)."""
    x, y = np.meshgrid(np.linspace(D - r, D + r, n), np.linspace(-r, r, n))
    m = (x - D) ** 2 + y ** 2 <= r ** 2
    v = np.hypot(x, y) / np.hypot(x - pJ[0], y - pJ[1])
    return np.min(v[m])


def varrho_min_collinear(D, d, r):
    """Proposition 4 (iii): jammer on the segment p_g--p_w at distance d from p_w."""
    r = np.asarray(r, float)
    return np.where(r ** 2 <= d * D, (D + r) / (d + r), (D - r) / np.maximum(r - d, 1e-12))


def varrho_min_opt(D, r):
    """Proposition 5: jammer at the inverse point p_w + (r/D)^2 (p_g - p_w)."""
    return D / np.asarray(r, float)


def robust_design(rho_b, Om_b, Om_g, K, R_b, R_g_max, rho_g_pk, rho_J_pk, eps, varrho_min, alpha):
    """Solve (P3): (P2) with Om_J/Om_w replaced by varrho_min^alpha.  Returns
    (ECT, rho_g, R_g, rho_J^th,rob, rho_J^max used)."""
    ke = rho_eps(eps)
    vu, rgu, Rgu = optimise(rho_b, Om_b, Om_g, K, R_b, rho_g_pk, R_max=R_g_max)
    rJ_th = ke * rgu / varrho_min ** alpha
    if rJ_th <= rho_J_pk:
        return vu, rgu, Rgu, rJ_th, rJ_th
    cap = min(rho_J_pk * varrho_min ** alpha / ke, rho_g_pk)
    v, rg, Rg = optimise(rho_b, Om_b, Om_g, K, R_b, cap, R_max=R_g_max)
    return v, rg, Rg, rJ_th, rho_J_pk


def dep_worst(rho_g, rho_J, varrho_min, alpha):
    """worst-case average DEP over the disc: Theorem 2 with kappa = rho_J varrho_min^alpha / rho_g."""
    return dep_avg(rho_g, rho_J, 1.0, varrho_min ** alpha)


if __name__ == "__main__":
    D, d, alpha = 100.0, 30.0, 3.0
    # (i) boundary attainment and (iii) collinear closed form
    for r in [5, 20, 40, 54.77, 60, 80, 95]:
        pJ = (D - d, 0.0)
        b, full, cf = varrho_boundary(D, pJ, r), varrho_disc(D, pJ, r), float(varrho_min_collinear(D, d, r))
        assert abs(b - full) < 2e-2 * b and abs(b - cf) < 1e-3 * b, (r, b, full, cf)
        print(f"r={r:6.2f}: boundary {b:.4f} disc {full:.4f} closed form {cf:.4f}")
    # off-axis jammer: boundary property still holds
    for r in [20, 60]:
        pJ = (D - d, 15.0)
        assert abs(varrho_boundary(D, pJ, r) - varrho_disc(D, pJ, r)) < 2e-2, r
    # (Proposition 5) inverse point is optimal: 2-D search over jammer positions
    for r in [10, 30, 60, 90]:
        best = max(varrho_boundary(D, (x, y), r, n=721)
                   for x in np.linspace(0, D + r, 121) for y in np.linspace(0, r, 41))
        inv = varrho_boundary(D, (D - r ** 2 / D, 0.0), r)
        assert inv >= best - 1e-2 * best and abs(inv - D / r) < 1e-3 * D / r, (r, best, inv, D / r)
        print(f"r={r}: best over grid {best:.4f}, inverse point {inv:.4f}, D/r {D / r:.4f}")
    print("robust.py: all checks passed")
