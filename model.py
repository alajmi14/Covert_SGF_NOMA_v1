"""Analytical expressions and Monte-Carlo simulator for
Covert Semi-Grant-Free (SGF) NOMA with a random-power cooperative jammer.

Notation (all powers normalised by the noise power, i.e. transmit SNRs):
  rho_b : GB user transmit SNR          Om_b : E|h_b|^2  (GB  -> BS)
  rho_g : GF user transmit SNR          Om_g : E|h_k|^2  (GF  -> BS), i.i.d.
  rho_J : jammer max transmit SNR       Om_w : E|g_k|^2  (GF  -> Willie)
                                        Om_J : E|g_J|^2  (Jam -> Willie)
  K     : number of GF users,  R_b, R_g : target rates [bit/s/Hz]
Jammer power P_J ~ U[0, rho_J]; the BS removes the jamming (pre-shared seed)
up to a residual fraction zeta (zeta = 0 in the analysis; called beta in the code).
"""
import numpy as np
from scipy.special import lambertw, comb


# ---------------------------------------------------------------- covertness
def dep_min_conditional(X, Y):
    """Theorem 1: minimum DEP given X = rho_g|g_k|^2, Y = rho_J|g_J|^2."""
    return np.maximum(0.0, 1.0 - X / Y)


def dep_avg(rho_g, rho_J, Om_w, Om_J):
    """Theorem 2: E[xi*] = 1 - ln(1+r)/r,  r = rho_J Om_J / (rho_g Om_w)."""
    r = rho_J * Om_J / (rho_g * Om_w)
    return 1.0 - np.log1p(r) / r


def rho_eps(eps):
    """Corollary 1: smallest r with ln(1+r)/r <= eps (Lambert-W, k=-1 branch)."""
    w = lambertw(-eps * np.exp(-eps), k=-1).real
    return -w / eps - 1.0


def rho_g_covert_max(eps, rho_J, Om_w, Om_J):
    """Corollary 1: largest GF transmit SNR meeting E[xi*] >= 1 - eps."""
    return rho_J * Om_J / (rho_eps(eps) * Om_w)


# ---------------------------------------------------------------- reliability
def gb_outage(rho_b, Om_b, R_b):
    """Proposition 1: GB outage (identical to OMA)."""
    eb = 2.0 ** R_b - 1.0
    return 1.0 - np.exp(-eb / (rho_b * Om_b))


def gf_outage(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g):
    """Theorem 3 (simplified form): exact GF (strongest admissible user) outage,
    P_out = 1 - e^{-x0/Om_b} [1 - sum_m C(K,m) phi^{K-m} e^{-m theta/Om_g} / (Om_b c_m)]."""
    db, dg = 2.0 ** R_b - 1.0, 2.0 ** R_g - 1.0
    theta = dg / rho_g
    x0 = db * (1.0 + dg) / rho_b
    phi = 1.0 - np.exp(-theta / Om_g)
    S = 0.0
    for m in range(K + 1):
        cm = 1.0 / Om_b + m * rho_b / (db * rho_g * Om_g)
        S = S + comb(K, m) * phi ** (K - m) * np.exp(-m * theta / Om_g) / (Om_b * cm)
    return 1.0 - np.exp(-x0 / Om_b) * (1.0 - S)


def gf_outage_hybrid(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g):
    """Theorem 4: exact GF outage of the hybrid-SIC extension (Section IV-C).
    A user is served if its gain g lies in [theta, psi(x)] (decoded after the GB user) or in
    [g1(x), inf) with g1 = theta (rho_b x + 1) (decoded before the GB user); psi >= g1 iff x >= x1,
    x1 = x0/(1 - delta_b delta_g) (x1 = inf if delta_b delta_g >= 1).  Vectorised over rho_g."""
    from math import factorial
    rho_b, rho_g = np.broadcast_arrays(np.asarray(rho_b, float), np.asarray(rho_g, float))
    db, dg = 2.0 ** R_b - 1.0, 2.0 ** R_g - 1.0
    theta = dg / rho_g
    a = rho_b / (db * rho_g)
    w = theta * rho_b / Om_g
    x0 = db * (1.0 + dg) / rho_b
    phi = 1.0 - np.exp(-theta / Om_g)
    fin = db * dg < 1.0
    x1 = x0 / (1.0 - db * dg) if fin else np.inf * np.ones_like(x0)
    # region I: 0 <= x < x0, success only through the GF-first branch
    I1 = 0.0
    for i in range(K + 1):
        e = 1.0 / Om_b + i * w
        I1 = I1 + comb(K, i) * (-1) ** i * np.exp(-i * theta / Om_g) * (1.0 - np.exp(-e * x0)) / (Om_b * e)
    # region II: x0 <= x < x1, both branches, gap (psi, g1) between them
    I2 = 0.0
    for n in range(K + 1):
        for p in range(K + 1 - n):
            coef = factorial(K) / (factorial(K - n - p) * factorial(n) * factorial(p)) * (-1) ** p
            E = 1.0 / Om_b + n * a / Om_g + p * w
            t0 = np.exp(-x0 / Om_b - (n + p) * theta / Om_g - p * w * x0)
            if fin:
                psi1 = a * (x1 - db / rho_b)
                t1 = np.exp(-x1 / Om_b - (n + p) * psi1 / Om_g)
            else:
                t1 = 0.0
            I2 = I2 + coef * phi ** (K - n - p) * (t0 - t1) / (Om_b * E)
    # region III: x >= x1, success iff some user can support R_g at all
    I3 = phi ** K * np.exp(-x1 / Om_b) if fin else 0.0
    return I1 + I2 + I3


def gf_outage_hybrid_floor(Om_b, Om_g, K, R_b, R_g, mu=1.0):
    """Corollary 4: floor of the hybrid extension for rho_g = mu rho_b -> infinity (delta_b delta_g >= 1);
    zero if delta_b delta_g < 1.  The p = 0 term is the GB-first floor of Corollary 2."""
    db, dg = 2.0 ** R_b - 1.0, 2.0 ** R_g - 1.0
    if db * dg < 1.0:
        return 0.0
    return sum(comb(K, p) * (-1) ** p / (1.0 + (Om_b / (mu * Om_g)) * ((K - p) / db + p * dg)) for p in range(K + 1))


def gf_outage_random(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g, n_quad=4000):
    """Benchmark: a uniformly chosen admissible GF user transmits (no channel-aware
    contention).  Given |h_b|^2 = x > x0, the chosen user's gain is Exp(Om_g) truncated
    to [0, psi(x)], so  P(success|x) = [1 - e^{-K psi/Om_g}] (e^{-theta/Om_g} - e^{-psi/Om_g})
    / (1 - e^{-psi/Om_g}).  The average over x is taken by composite Simpson quadrature
    on x in [x0, x0 + 40 Om_b] (tail beyond that < e^{-40})."""
    rho_b, rho_g = np.broadcast_arrays(np.asarray(rho_b, float), np.asarray(rho_g, float))
    db, dg = 2.0 ** R_b - 1.0, 2.0 ** R_g - 1.0
    theta = dg / rho_g
    x0 = db * (1.0 + dg) / rho_b
    t = np.linspace(0.0, 40.0 * Om_b, n_quad + 1)          # x = x0 + t
    x = x0[..., None] + t
    psi = (rho_b[..., None] * x / db - 1.0) / rho_g[..., None]
    e_psi = np.exp(-psi / Om_g)
    e_th = np.exp(-theta / Om_g)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ps = (1.0 - e_psi ** K) * (e_th - e_psi) / (1.0 - e_psi)
    ps = np.where(psi > theta[..., None], ps, 0.0)
    w = np.ones(n_quad + 1); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0    # Simpson weights
    integrand = ps * np.exp(-x / Om_b) / Om_b
    succ = (integrand * w).sum(axis=-1) * (t[1] - t[0]) / 3.0
    return 1.0 - succ


def gf_outage_floor(Om_b, Om_g, K, R_b):
    """Corollary 2: high-SNR floor for rho_b = rho_g -> infinity."""
    eb = 2.0 ** R_b - 1.0
    return 1.0 / (1.0 + K * Om_b / (eb * Om_g))


def ect(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g, outage=gf_outage):
    """Effective covert throughput of the GF service."""
    return R_g * (1.0 - outage(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g))


def optimise(rho_b, Om_b, Om_g, K, R_b, rho_g_cap, R_max=4.0, R_grid=None, n_rho=400,
             outage=gf_outage):
    """Algorithm 1: maximise ECT over (rho_g, R_g) s.t. rho_g <= rho_g_cap, R_g <= R_max.
    Inner problem: 1-D grid + golden-section refinement over log(rho_g)."""
    if R_grid is None:
        R_grid = np.arange(0.05, R_max + 1e-9, 0.05)
    best = (-1.0, None, None)
    lo_db = 10 * np.log10(rho_g_cap) - 60.0
    hi_db = 10 * np.log10(rho_g_cap)
    grid_db = np.linspace(lo_db, hi_db, n_rho)
    for R in R_grid:
        vals = ect(rho_b, 10 ** (grid_db / 10), Om_b, Om_g, K, R_b, R, outage)
        i = int(np.argmax(vals))
        a = grid_db[max(i - 1, 0)]
        b = grid_db[min(i + 1, n_rho - 1)]
        f = lambda d: -ect(rho_b, 10 ** (d / 10), Om_b, Om_g, K, R_b, R, outage)
        gr = (np.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        for _ in range(60):
            if f(c) < f(d):
                b = d
            else:
                a = c
            c, d = b - gr * (b - a), a + gr * (b - a)
        xd = 0.5 * (a + b)
        v = float(-f(xd))
        if v > best[0]:
            best = (v, 10 ** (xd / 10), R)
    return best  # (ECT*, rho_g*, R_g*)


# ---------------------------------------------------------------- Monte Carlo
def mc_dep(rho_g, rho_J, Om_w, Om_J, n_ch=10000, n_pj=20000, n_eval=50000,
           rho_b=1.0, Om_bw=1.0, n=None, rng=None):
    """Empirical minimum DEP of Willie's radiometer.
    n=None: infinite block length (T equals the received power).  Finite n: the
    radiometer statistic T = (1/n) sum |y_w[i]|^2 of n i.i.d. CN(0, s) samples is
    s * Gamma(n, 1/n), which is drawn directly.
    For each channel realisation, Willie's statistic under H0/H1 is generated with
    n_pj independent jamming-power draws and the best threshold is found by
    exhaustive search on the empirical distributions.  The DEP of that threshold
    is then measured on n_eval *independent* draws: evaluating on the search
    samples would give the minimum of a noisy curve, which is biased low (most
    visibly where the true DEP curve is flat, i.e. near 1)."""
    rng = np.random.default_rng(rng)
    out = np.empty(n_ch)

    def stat(s, size):                      # radiometer statistic for variance s
        return s if n is None else s * rng.gamma(n, 1.0 / n, size)

    for i in range(n_ch):
        gw = rng.exponential(Om_w)
        gJ = rng.exponential(Om_J)
        base = 1.0 + rho_b * rng.exponential(Om_bw)  # noise + known GB power (irrelevant to xi*)
        shift = rho_g * gw
        # --- threshold search on one sample set
        T0 = stat(base + rng.uniform(0, rho_J, n_pj) * gJ, n_pj)
        T1 = stat(base + shift + rng.uniform(0, rho_J, n_pj) * gJ, n_pj)
        th = np.sort(np.concatenate([T0, T1, [T0.min() - 1, T1.max() + 1]]))
        s0, s1 = np.sort(T0), np.sort(T1)
        pfa = 1.0 - np.searchsorted(s0, th, side="right") / n_pj
        pmd = np.searchsorted(s1, th, side="right") / n_pj
        th_star = th[np.argmin(pfa + pmd)]
        # --- DEP of that threshold on an independent sample set
        E0 = stat(base + rng.uniform(0, rho_J, n_eval) * gJ, n_eval)
        E1 = stat(base + shift + rng.uniform(0, rho_J, n_eval) * gJ, n_eval)
        out[i] = np.mean(E0 > th_star) + np.mean(E1 <= th_star)
    return out.mean()


def mc_sgf(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g, n=10 ** 6, beta=0.0,
           rho_J=0.0, Om_JB=1.0, select="best", bs_knows_residual=False, rate_aware=True,
           rng=None):
    """Monte-Carlo of the SGF protocol. Returns (P_out_GF, P_out_GB).
    select="best": the strongest admissible user (proposed scheme); "margin": the admissible
    user maximising min(tau - rx, rx/eg - 1) (max-margin benchmark); "random": a uniformly
    chosen admissible user (benchmark without channel-aware selection).
    rate_aware=True applies both bounds of the admission interval (a user that cannot support
    R_g does not transmit); the random-admission benchmark of the paper is rate-unaware
    (any user below tau may be chosen), which is what gf_outage_random() computes.
    beta>0: residual jamming at the BS (not covered by the analysis).  The residual
    models estimation/hardware errors, so by default the BS does not know it and
    computes the admission threshold from the noise power alone
    (bs_knows_residual=True reproduces an oracle BS that subtracts the realised residual)."""
    rng = np.random.default_rng(rng)
    eb, eg = 2.0 ** R_b - 1.0, 2.0 ** R_g - 1.0
    hb = rng.exponential(Om_b, n)
    hk = rng.exponential(Om_g, (n, K))
    I = 1.0 + beta * rng.uniform(0, rho_J, n) * rng.exponential(Om_JB, n)   # actual noise + residual
    I_bs = I if bs_knows_residual else np.ones(n)                          # what the BS uses in (1)
    # admission threshold computed by the BS (received-SINR domain) and admissible set
    tau = np.maximum(0.0, rho_b * hb / eb - I_bs)
    rx = rho_g * hk
    adm = rx <= tau[:, None]                                   # does not exceed tau
    if rate_aware:
        adm &= rx >= eg * I_bs[:, None]                        # and can support R_g
    if select == "best":
        rxa = np.where(adm, rx, -1.0)
        sel = rxa.max(axis=1)
    elif select == "margin":
        # max-margin rule: the admissible user with the largest joint margin
        # min(Gamma_b, Gamma_g) = min(tau - rx, rx/eg - 1) against residual jamming (Remark on
        # selection in Sec. V-D); needs only tau and rx, not the residual
        m = np.where(adm, np.minimum(tau[:, None] - rx, rx / eg - 1.0), -np.inf)
        idx = m.argmax(axis=1)
        sel = np.where(adm.any(axis=1), rx[np.arange(n), idx], -1.0)
    else:  # random admissible user
        u = rng.random((n, K))
        u = np.where(adm, u, -1.0)
        idx = u.argmax(axis=1)
        sel = np.where(adm.any(axis=1), rx[np.arange(n), idx], -1.0)
    has = sel >= 0
    interf = np.where(has, sel, 0.0)
    gb_ok = rho_b * hb / (interf + I) >= eb
    gf_ok = has & gb_ok & (sel / I >= eg)
    return 1.0 - gf_ok.mean(), 1.0 - gb_ok.mean()


def mc_hybrid(rho_b, rho_g, Om_b, Om_g, K, R_b, R_g, n=10 ** 6, rng=None):
    """Idealised hybrid-SIC scheme [Ding 2021]: a GF user is served if either the GB-first
    branch (some user in the admission interval) or the GF-first branch (some user above tau
    that is decodable with the GB signal as interference) can decode one.  Returns
    (P_out_GF hybrid, P_out_GF GB-first only)."""
    rng = np.random.default_rng(rng)
    eb, eg = 2.0 ** R_b - 1.0, 2.0 ** R_g - 1.0
    hb = rng.exponential(Om_b, n)
    hk = rng.exponential(Om_g, (n, K))
    tau = np.maximum(0.0, rho_b * hb / eb - 1.0)
    rx = rho_g * hk
    gb_first = ((rx <= tau[:, None]) & (rx >= eg)).any(axis=1)
    gf_first = ((rx > tau[:, None]) & (rx / (rho_b * hb[:, None] + 1.0) >= eg)).any(axis=1)
    return 1.0 - (gb_first | gf_first).mean(), 1.0 - gb_first.mean()
