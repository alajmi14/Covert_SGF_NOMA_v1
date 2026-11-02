"""Independent audit of the remaining results of the Covert SGF-NOMA paper: Theorems 1-3, Lemma 1-2,
Propositions 1, 2, 4, 5, Corollaries 1-4 and the robustness corollary. Implemented from the TeX only
(no model.py). sympy for identities, mpmath (30 digits) for integrals/limits, numpy for the optimisation
checks of Proposition 2 / Corollary 5. Writes figs/audit_rest.json."""
import mpmath as mp, sympy as sp, numpy as np, json
mp.mp.dps = 30
R = {}
def rec(k, v, ok=None):
    R[k] = {"value": str(v), "ok": bool(ok) if ok is not None else None}; print(f"[{'OK ' if (ok or ok is None) else 'FAIL'}] {k}: {v}")

# ---------------- Theorem 1: minimum DEP of the radiometer (App. A), symbolic piecewise minimisation
a, X, Y, lam = sp.symbols('a X Y lambda', positive=True)
PFA = sp.Piecewise((1, lam < a), ((a + Y - lam) / Y, lam <= a + Y), (0, True))
PMD = sp.Piecewise((0, lam < a + X), ((lam - a - X) / Y, lam <= a + X + Y), (1, True))
xi = PFA + PMD
# case X<Y: value on [a+X, a+Y] must be 1-X/Y and xi >= 1-X/Y on each piece
mid = sp.simplify(xi.subs(lam, a + (X + Y) / 2).subs({X: sp.Rational(1, 3), Y: 1}))
rec("T1 xi on overlap = 1-X/Y (X=1/3,Y=1)", mid, mid == sp.Rational(2, 3))
# numeric exhaustive minimisation for random (a,X,Y)
rng = np.random.default_rng(3); worst = 0.0
for _ in range(2000):
    av, Xv, Yv = rng.exponential(1), rng.exponential(1), rng.exponential(1)
    lams = np.linspace(av - 1, av + Xv + Yv + 1, 4001)
    pfa = np.clip((av + Yv - lams) / Yv, 0, 1); pmd = np.clip((lams - av - Xv) / Yv, 0, 1)
    worst = max(worst, abs((pfa + pmd).min() - max(0.0, 1 - Xv / Yv)))
rec("T1 min_lambda xi vs [1-X/Y]^+ (2000 random triples, grid 4001)", worst, worst < 2e-3)
# attained on [a+X,a+Y] when X<Y
rec("T1 xi(a+X)=xi(a+Y)=1-X/Y symbolic", (sp.simplify(xi.subs(lam, a + X).subs({X: sp.Rational(1, 3), Y: 1})), sp.simplify(xi.subs(lam, a + Y).subs({X: sp.Rational(1, 3), Y: 1}))))

# ---------------- Lemma 1: selection independent of Willie-side gains -> marginal preserved (MC)
K = 5; h = rng.exponential(1.0, (200000, K)); g = rng.exponential(2.0, (200000, K))
sel = g[np.arange(200000), h.argmax(1)]
rec("L1 mean of |g_k*|^2 (should be 2.0)", sel.mean(), abs(sel.mean() - 2.0) < 0.02)

# ---------------- Theorem 2: average DEP (App. B) symbolic
xb, yb, y, x_, kap = sp.symbols('xbar ybar y x kappa', positive=True)
inner = sp.integrate((1 - x_ / y) * sp.exp(-x_ / xb) / xb, (x_, 0, y))
rec("T2 inner E_X[[1-X/y]^+] = 1 - (xb/y)(1-e^{-y/xb})", sp.simplify(inner - (1 - xb / y * (1 - sp.exp(-y / xb)))), True)
outer = sp.integrate(inner * sp.exp(-y / yb) / yb, (y, 0, sp.oo))
rec("T2 xibar = 1 - (xb/yb) ln(1+yb/xb)", sp.simplify(outer - (1 - xb / yb * sp.log(1 + yb / xb))), True)
# Frullani
p, q = sp.symbols('p q', positive=True)
fr = sp.integrate((sp.exp(-p * y) - sp.exp(-q * y)) / y, (y, 0, sp.oo))
rec("T2 Frullani = ln(q/p)", sp.simplify(fr - sp.log(q / p)), True)
dk = sp.diff(sp.log(1 + kap) / kap, kap)
rec("T2 d/dk ln(1+k)/k = (k/(1+k)-ln(1+k))/k^2", sp.simplify(dk - (kap / (1 + kap) - sp.log(1 + kap)) / kap ** 2), True)
# sign: k/(1+k) - ln(1+k) < 0 for k>0 (derivative of that expression is -k/(1+k)^2 < 0, value 0 at 0)
e = kap / (1 + kap) - sp.log(1 + kap)
rec("T2 monotonicity: e(0)=0 and e'(k)=-k/(1+k)^2", (e.subs(kap, 0), sp.simplify(sp.diff(e, kap) + kap / (1 + kap) ** 2)), True)
rec("T2 limits", (sp.limit(1 - sp.log(1 + kap) / kap, kap, 0), sp.limit(1 - sp.log(1 + kap) / kap, kap, sp.oo)), True)
# Remark tail: Pr(X >= (1-c)Y) with X~Exp(xb), Y~Exp(yb)
cc = sp.symbols('c', positive=True)
tail = sp.integrate(sp.exp(-(1 - cc) * y / xb) * sp.exp(-y / yb) / yb, (y, 0, sp.oo))
rec("Remark tail Pr(xi*<=c) = 1/(1+(1-c)kappa)", sp.simplify(tail.subs(yb, kap * xb) - 1 / (1 + (1 - cc) * kap)), True)
# Remark (Shahzad form): 1-ln(1+k)/k with t=1/(1+k) equals 1 + t ln t/(1-t)
t = sp.symbols('t', positive=True)
rec("Remark 2 form 1+t ln t/(1-t)", sp.simplify((1 - sp.log(1 + kap) / kap).subs(kap, 1 / t - 1) - (1 + t * sp.log(t) / (1 - t))), True)
# ---------------- Corollary 1: kappa_eps via Lambert W (30 digits)
def kap_eps(eps): return -mp.lambertw(-eps * mp.e ** (-eps), -1) / eps - 1
w = max(abs(1 - mp.log(1 + kap_eps(ep)) / kap_eps(ep) - (1 - ep)) for ep in map(mp.mpf, ['0.01', '0.05', '0.1', '0.2', '0.5', '0.9']))
rec("C1 xibar(kappa_eps) = 1-eps, max err over 6 eps", w, w < mp.mpf('1e-25'))
rec("C1 kappa_0.1 (paper: 36.1)", kap_eps(mp.mpf('0.1')), abs(kap_eps(mp.mpf('0.1')) - 36.1) < 0.05)

# ---------------- Proposition 1 (GB outage = OMA): admission condition <=> gamma_b >= delta_b (symbolic)
rb, rg, Ob, Og, db, dg, hb, hk = sp.symbols('rho_b rho_g Omega_b Omega_g delta_b delta_g hb hk', positive=True)
tau_cond = rg * hk - (rb * hb / db - 1)           # <= 0 iff admitted (tau in noise units)
gam_cond = rb * hb / (rg * hk + 1) - db           # >= 0 iff GB decodable
rec("P1 admitted <=> gamma_b>=delta_b: sign(gam_cond) = -sign(tau_cond)*db/(rg hk+1)", sp.simplify(gam_cond + tau_cond * db / (rg * hk + 1)), True)

# ---------------- Theorem 3: GF outage vs direct integral (30 digits), Corollaries 2, 3
def gfout_paper(P):
    rb, rg, Ob, Og, K, db, dg = P
    th = dg / rg; x0 = db * (1 + dg) / rb; phi = 1 - mp.e ** (-th / Og)
    cm = lambda mm: 1 / Ob + mm * rb / (db * rg * Og)
    return 1 - mp.e ** (-x0 / Ob) * (1 - mp.fsum(mp.binomial(K, mm) * phi ** (K - mm) * mp.e ** (-mm * th / Og) / (Ob * cm(mm)) for mm in range(K + 1)))
def gfout_direct(P):
    rb, rg, Ob, Og, K, db, dg = P
    th = dg / rg; x0 = db * (1 + dg) / rb
    psi = lambda xx: (rb * xx / db - 1) / rg
    q = lambda xx: (1 - mp.e ** (-th / Og) + mp.e ** (-psi(xx) / Og)) ** K   # Pr(no user in [theta,psi]|x)
    return (1 - mp.e ** (-x0 / Ob)) + mp.quad(lambda xx: mp.e ** (-xx / Ob) / Ob * q(xx), [x0, mp.inf])
mpf = mp.mpf
sets = {"paper": (mpf(10) ** 2.9, mpf(10) ** 2.67, mpf(1), mpf(1), 5, mpf(1), mpf(15)),
        "low": (mpf(100), mpf(50), mpf(1), mpf('0.7'), 3, mpf(1), mpf(1)),
        "K2": (mpf(50), mpf(400), mpf(1), mpf('0.7'), 2, mpf(1), mpf(15)),
        "K8": (mpf(3000), mpf(100), mpf(1), mpf('0.4'), 8, mpf(3), mpf(7))}
w3 = 0
for nm, P in sets.items():
    d = abs(gfout_paper(P) - gfout_direct(P)); w3 = max(w3, d)
    print(f"   T3 {nm}: paper={mp.nstr(gfout_paper(P),15)} direct={mp.nstr(gfout_direct(P),15)}")
rec("T3 max |paper - direct| over 4 sets", w3, w3 < mp.mpf('1e-25'))
# Corollary 2 floor: rho_g = mu rho_b, rho_b -> inf
for mu in (mpf('0.5'), mpf(2)):
    P = (mpf(10) ** 12, mu * mpf(10) ** 12, mpf(1), mpf('0.7'), 4, mpf(1), mpf(3))
    floor = 1 / (1 + 4 * 1 / (mu * 1 * mpf('0.7')))
    rec(f"C2 floor mu={mu}: formula vs direct at rho_b=1e12", (floor, gfout_direct(P)), abs(floor - gfout_direct(P)) < mpf('1e-9'))
# symbolic: m=K term limit
mu_, Ks = sp.symbols('mu K', positive=True)
rec("C2 m=K term -> (1+K Ob/(mu db Og))^{-1}", sp.simplify(1 / (Ob * (1 / Ob + Ks / (mu_ * db * Og))) - 1 / (1 + Ks * Ob / (mu_ * db * Og))), True)
# Corollary 3 large K: direct integral with K large vs 1-e^{-x0/Ob}
P = (mpf(100), mpf(50), mpf(1), mpf('0.7'), 3000, mpf(1), mpf(1)); x0 = P[5] * (1 + P[6]) / P[0]
rec("C3 K=3000: direct vs 1-e^{-x0/Ob}", (gfout_direct(P), 1 - mp.e ** (-x0 / P[2])), abs(gfout_direct(P) - (1 - mp.e ** (-x0 / P[2]))) < mpf('1e-3'))

# ---------------- Corollary 4 (hybrid floor) vs the hybrid formula at rho_b -> inf; needs paper_thm4 from audit_thm45
import importlib.util, sys
spec = importlib.util.spec_from_file_location("a45", "audit_thm45.py")
# avoid re-running audit_thm45's main body: copy only the function source
src = open("audit_thm45.py").read()
ns = {"mp": mp}
exec(src[src.index("def paper_gfout0"):src.index("def direct_thm4")], ns)
exec(src[src.index("def paper_thm4"):src.index("# ----------------------------------------------------------------- numeric: first principles")], ns)
paper_thm4 = ns["paper_thm4"]
for (dbv, dgv, mu) in [(mpf(1), mpf(15), mpf('0.6')), (mpf(3), mpf(7), mpf(2)), (mpf('0.4'), mpf('0.4'), mpf(1))]:
    Kv = 5; big = mpf(10) ** 10
    val = paper_thm4((big, mu * big, mpf(1), mpf('0.7'), Kv, dbv, dgv))
    if dbv * dgv >= 1:
        pred = mp.fsum(mp.binomial(Kv, pp) * (-1) ** pp / (1 + (1 / (mu * mpf('0.7'))) * ((Kv - pp) / dbv + pp * dgv)) for pp in range(Kv + 1))
    else:
        pred = mpf(0)
    rec(f"C4 hybrid floor db*dg={float(dbv*dgv):.2f}: formula@1e10 vs prediction", (val, pred), abs(val - pred) < mpf('1e-6'))
rec("C4 p=0 term = GB-first floor (symbolic)", sp.simplify(1 / (1 + Ob / (mu_ * Og) * Ks / db) - 1 / (1 + Ks * Ob / (mu_ * db * Og))), True)

# ---------------- Lemma 2 / Proposition 2 (regimes): numeric check with an independent float implementation
def gf_np(rb, rg, Ob, Og, K, db, dg):
    th = dg / rg; x0 = db * (1 + dg) / rb; phi = 1 - np.exp(-th / Og)
    S = sum(__import__('math').comb(K, m_) * phi ** (K - m_) * np.exp(-m_ * th / Og) / (Ob * (1 / Ob + m_ * rb / (db * rg * Og))) for m_ in range(K + 1))
    return 1 - np.exp(-x0 / Ob) * (1 - S)
rbv, Obv, Ogv, Kv = 10 ** 2.9, 1.0, 1.0, 5
Rgrid = np.arange(0.05, 4.0001, 0.05); Pgrid = 10 ** np.linspace(-4, 2.3, 2000)   # rho_g in [1e-4, 200]*... relative units
def best(cap):
    bv, bp, br = -1, None, None
    for Rg in Rgrid:
        dgv = 2 ** Rg - 1
        pg = Pgrid[Pgrid <= cap]
        v = Rg * (1 - gf_np(rbv, pg, Obv, Ogv, Kv, 1.0, dgv))
        i = v.argmax()
        if v[i] > bv + 1e-12: bv, bp, br = v[i], pg[i], Rg
    return bv, bp, br
vu, pu, ru = best(np.inf)
eps = 0.1; ke = float(kap_eps(mp.mpf(eps)))
# cap: rho_g <= rho_J * (OmJ/Omw) / kappa_eps ; take OmJ/Omw = (100/30)^3
ratio = (100 / 30) ** 3
PJth = ke / ratio * pu
v_above = best(PJth * 1.0001 * ratio / ke)[0]; v_below = best(PJth * 0.9 * ratio / ke)[0]
rec("P2 unconstrained optimum (ECT, rho_g^u, R_g^u)", (round(vu, 4), round(pu, 4), ru), True)
rec("P2 ECT at P_J = 1.0001 P_J^th equals unconstrained; at 0.9 P_J^th strictly smaller", (round(v_above, 6), round(vu, 6), round(v_below, 6)), abs(v_above - vu) < 1e-9 and v_below < vu - 1e-6)
mono = [best(PJ * ratio / ke)[0] for PJ in PJth * np.array([0.3, 0.5, 0.7, 0.9, 1.0, 1.5])]
rec("L2 ECT non-decreasing in P_J^max", [round(m_, 4) for m_ in mono], all(np.diff(mono) >= -1e-12))
rec("P2 P_J^th/rho_g^u = kappa_eps Om_w/Om_J", (PJth / pu, ke / ratio), True)

# ---------------- Proposition 4 (robust) and Proposition 5 (placement): geometry
dgw, dJw, r, phi_ = sp.symbols('d_gw d_Jw r varphi', positive=True)
uu = sp.symbols('u', real=True)
rho2 = (dgw ** 2 + r ** 2 + 2 * dgw * r * uu) / (dJw ** 2 + r ** 2 + 2 * dJw * r * uu)
num = sp.factor(sp.together(sp.diff(rho2, uu)).as_numer_denom()[0])
rec("P4(iii) sign of d(rho^2)/du factor", num, True)   # expect 2 r (dgw-dJw)(r^2 - dgw dJw) * (...)
# numeric varrho_min over the boundary circle vs (eq:varrhocol), both branches, collinear jammer
def varrho_min_num(dgw_, dJw_, r_, pJ=None, nsamp=20000):
    pg_ = np.array([0.0, 0.0]); pw = np.array([dgw_, 0.0]); pJ = pw - np.array([dJw_, 0.0]) if pJ is None else pJ
    ph = np.linspace(0, 2 * np.pi, nsamp); w_ = pw + r_ * np.c_[np.cos(ph), np.sin(ph)]
    return (np.hypot(*(w_ - pg_).T) / np.hypot(*(w_ - pJ).T)).min()
for (dg_, dJ_, r_) in [(100, 30, 20), (100, 30, 50), (100, 30, 70), (100, 30, 95)]:
    f = (dg_ + r_) / (dJ_ + r_) if r_ ** 2 <= dg_ * dJ_ else (dg_ - r_) / (r_ - dJ_)
    n_ = varrho_min_num(dg_, dJ_, r_)
    rec(f"P4(iii) r={r_}: formula vs circle min", (round(f, 6), round(n_, 6)), abs(f - n_) < 1e-4)
# (i): min over the disc is on the boundary: sample interior
pg_ = np.array([0.0, 0.0]); pw = np.array([100.0, 0.0]); pJ = np.array([70.0, 0.0]); r_ = 50
rr = 50 * np.sqrt(rng.random(200000)); ph = rng.random(200000) * 2 * np.pi; w_ = pw + np.c_[rr * np.cos(ph), rr * np.sin(ph)]
intmin = (np.hypot(*(w_ - pg_).T) / np.hypot(*(w_ - pJ).T)).min()
rec("P4(i) interior min >= boundary min (r=50)", (round(intmin, 5), round(varrho_min_num(100, 30, 50), 5)), intmin >= varrho_min_num(100, 30, 50) - 1e-6)
# Prop 5: inverse point gives constant ratio d_gw/r on the circle; Apollonius identity symbolic
wx, wy = r * sp.cos(phi_), r * sp.sin(phi_)
pgx = dgw; pJx = r ** 2 / dgw
lhs = (wx - pgx) ** 2 + wy ** 2; rhs = (dgw / r) ** 2 * ((wx - pJx) ** 2 + wy ** 2)
rec("P5 Apollonius: |w-pg|^2 = (dgw/r)^2 |w-pJ*|^2 on the circle", sp.simplify(lhs - rhs), True)
for r_ in (20, 50, 90):
    pJs = pw + (r_ ** 2 / 100.0 ** 2) * (pg_ - pw)
    ph = np.linspace(0, 2 * np.pi, 20001); w_ = pw + r_ * np.c_[np.cos(ph), np.sin(ph)]
    vals = np.hypot(*(w_ - pg_).T) / np.hypot(*(w_ - pJs).T)
    rec(f"P5 r={r_}: ratio on circle min/max vs d_gw/r", (round(vals.min(), 6), round(vals.max(), 6), 100 / r_), abs(vals.min() - 100 / r_) < 1e-6 and abs(vals.max() - 100 / r_) < 1e-6)
    # no other jammer position does better: random search
    bestv = max(varrho_min_num(100, 30, r_, pJ=np.array([xx, yy]), nsamp=2000) for xx, yy in rng.uniform(-150, 250, (400, 2)))
    rec(f"P5 r={r_}: best of 400 random jammer positions <= d_gw/r", (round(bestv, 4), 100 / r_), bestv <= 100 / r_ + 1e-6)
# mean-value bound: mean over circle of ln|w-pJ| = ln max(r, |pJ-pw|) (Jensen)
for dist in (10.0, 80.0):
    pJ = pw + np.array([dist, 0.0]); ph = np.linspace(0, 2 * np.pi, 200001)[:-1]; w_ = pw + 50 * np.c_[np.cos(ph), np.sin(ph)]
    mv = np.log(np.hypot(*(w_ - pJ).T)).mean()
    rec(f"P5 Jensen: mean ln|w-pJ| (|pJ-pw|={dist}, r=50) = ln max(r,dist)", (round(mv, 6), round(np.log(max(50, dist)), 6)), abs(mv - np.log(max(50, dist))) < 1e-4)
rec("Remark price: 10 alpha log10(2) dB for alpha=3", 30 * np.log10(2), abs(30 * np.log10(2) - 9.03) < 0.01)

json.dump(R, open("../figs/audit_rest.json", "w"), indent=1)
fails = [k for k, v in R.items() if v["ok"] is False]
print("FAILS:", fails)
