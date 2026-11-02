"""Independent algebraic/numerical audit of Theorem 4 (hybrid SIC, App. E) and Theorem 5 (residual
jamming, App. D) of the Covert SGF-NOMA paper. Everything here is implemented from the TeX of the
theorems and appendices, NOT from model.py/residual.py, and compared with first-principles integrals
evaluated by high-precision quadrature (mpmath) and with symbolic identities (sympy)."""
import mpmath as mp, sympy as sp, itertools, json
mp.mp.dps = 30
out = {}

# ----------------------------------------------------------------- symbolic identities
rb, rg, Ob, Og, db, dg, x, m, s, c, D, E, u, k = sp.symbols('rho_b rho_g Omega_b Omega_g delta_b delta_g x m s c D E u k', positive=True)
theta = dg / rg; x0 = db * (1 + dg) / rb; psi = (rb * x / db - 1) / rg
a = rb / (db * rg); b1 = dg * a / (1 + dg); b2 = a / (1 + dg)
ck = 1 / Ob + k * rb / (db * rg * Og)
mc = (psi + 1 / rg) * dg / (1 + dg)
ids = {
 "psi(x0)=theta": sp.simplify(psi.subs(x, x0) - theta),
 "psi-theta=a(x-x0)": sp.simplify(psi - theta - a * (x - x0)),
 "mc-theta=b1(x-x0)": sp.simplify(mc - theta - b1 * (x - x0)),
 "psi-mc=b2(x-x0)": sp.simplify(psi - mc - b2 * (x - x0)),
 "ck x0 - k/(rg Og) = x0/Ob + k theta/Og": sp.simplify(ck * x0 - k / (rg * Og) - x0 / Ob - k * theta / Og),
 "rho_b/delta_b = rho_g a": sp.simplify(rb / db - rg * a),
 "margins equal at mc: rg(psi-mc) = rg mc/dg - 1": sp.simplify(rg * (psi - mc) - (rg * mc / dg - 1)),
 "b1+b2=a": sp.simplify(b1 + b2 - a),
}
# hybrid identities
nu = dg * rb / (rg * Og); chi = theta * (1 + rb * x)
x1 = x0 / (1 - db * dg)
n_, p_ = sp.symbols('n p', positive=True)
cn = 1 / Ob + n_ * rb / (db * rg * Og)
ids.update({
 "chi/Og = theta/Og + nu x": sp.simplify(chi / Og - theta / Og - nu * x),
 "n psi/Og = n theta/Og + (c_n-1/Ob)(x-x0)": sp.simplify(n_ * psi / Og - n_ * theta / Og - (cn - 1 / Ob) * (x - x0)),
 "psi(x1)=chi(x1)": sp.simplify((psi - chi).subs(x, x1)),
 "chi>=theta": sp.simplify(chi - theta),   # = theta rb x >= 0
})
# region-II integrand of App. E: exp(-(n+p)theta/Og + (c_n-1/Ob)x0) exp(-(c_n+p nu)x)/Ob  ==  exp(-n psi/Og - p chi/Og) exp(-x/Ob)/Ob
lhs = sp.exp(-(n_ + p_) * theta / Og + (cn - 1 / Ob) * x0) * sp.exp(-(cn + p_ * nu) * x) / Ob
rhs = sp.exp(-n_ * psi / Og - p_ * chi / Og) * sp.exp(-x / Ob) / Ob
ids["AppE region-II integrand"] = sp.simplify(sp.log(lhs) - sp.log(rhs))
# exponents after integrating over [x0,x1]: at x0 -> -x0/Ob-(n+p)theta/Og - p nu x0 ; at x1 -> -x1/Ob-(n+p)chi(x1)/Og
expo = -(n_ + p_) * theta / Og + (cn - 1 / Ob) * x0 - (cn + p_ * nu) * x
ids["AppE exponent at x0"] = sp.simplify(expo.subs(x, x0) - (-x0 / Ob - (n_ + p_) * theta / Og - p_ * nu * x0))
ids["AppE exponent at x1"] = sp.simplify(expo.subs(x, x1) - (-x1 / Ob - (n_ + p_) * chi.subs(x, x1) / Og))
# u-integral of App. D: int_0^1 du/(D+E/(uc)) = vartheta(D,E)
I = sp.integrate(1 / (D + E / (u * c)), (u, 0, 1))
vt = (1 / D) * (1 - E / (D * c) * sp.log(1 + D * c / E))
ids["AppD int_0^1 du/(D+E/(uc)) = vartheta"] = sp.simplify(I - vt)
# int_1^inf e^{-alpha v}/(v+beta) dv = e^{alpha beta} E1(alpha(1+beta))
al, be, v = sp.symbols('alpha beta v', positive=True)
# sympy returns this one in terms of Ei with a polar branch (E1(z) = -Ei(-z) for z>0); check it at 30 digits instead
Jmax = max(abs(mp.quad(lambda vv: mp.e ** (-A * vv) / (vv + B), [1, mp.inf]) - mp.e ** (A * B) * mp.expint(1, A * (1 + B)))
           for A in (mp.mpf('0.05'), mp.mpf(1), mp.mpf(7)) for B in (mp.mpf('0.1'), mp.mpf(2), mp.mpf(30)))
ids["AppD int_1^inf e^{-av}/(v+b) = e^{ab}E1(a(1+b)) [30-digit check, max err]"] = 0 if Jmax < mp.mpf('1e-25') else Jmax
for kk, vv in ids.items():
    print(f"[sym] {kk:60s} -> {vv}")
out["symbolic"] = {kk: str(vv) for kk, vv in ids.items()}
assert all(vv == 0 for kk, vv in ids.items() if kk != "chi>=theta")

# ----------------------------------------------------------------- numeric: paper formulas (from TeX)
def paper_gfout0(P):
    rb, rg, Ob, Og, K, db, dg = P
    th = dg / rg; x0 = db * (1 + dg) / rb; phi = 1 - mp.e ** (-th / Og)
    cm = lambda mm: 1 / Ob + mm * rb / (db * rg * Og)
    S = mp.fsum(mp.binomial(K, mm) * phi ** (K - mm) * mp.e ** (-mm * th / Og) / (Ob * cm(mm)) for mm in range(K + 1))
    return 1 - mp.e ** (-x0 / Ob) * (1 - S)

def paper_thm5(P, cc):
    rb, rg, Ob, Og, K, db, dg = P
    th = dg / rg; x0 = db * (1 + dg) / rb; phi = 1 - mp.e ** (-th / Og)
    a = rb / (db * rg); b1 = dg * a / (1 + dg); b2 = a / (1 + dg)
    cm = lambda mm: 1 / Ob + mm * rb / (db * rg * Og)
    vt = lambda D, E: (1 / D) * (1 - E / (D * cc) * mp.log(1 + D * cc / E))
    E1 = lambda z: mp.expint(1, z)
    def vs(D):
        t1 = (1 / D) * (mp.e ** (-dg / cc) - (dg / cc) * E1(dg / cc))
        t2 = rb / (db * D ** 2 * cc) * (E1(dg / cc) - mp.e ** (db * dg * D / rb) * E1(dg / cc + db * dg * D / rb))
        return t1 - t2
    beta = lambda j, l: mp.binomial(K - 1, j) * mp.binomial(j, l) * (-1) ** l * mp.e ** (-(j + 1) * th / Og)
    pref = K * mp.e ** (-x0 / Ob) / (Ob * Og)
    Sg = Sb = mp.mpf(0)
    for j in range(K):
        for l in range(j + 1):
            v1 = vt(cm(j - l) + (l + 1) * b1 / Og, rg * b1 / dg)
            v2 = vt(cm(j + 1) - (l + 1) * b2 / Og, rg * b2)
            v3 = vt(cm(j + 1) - (l + 1) * a / Og, rg * a)
            Sg += beta(j, l) * (b1 * v1 / cm(j - l) + b2 * v2 / cm(j + 1))
            Sb += beta(j, l) * a * v3 / cm(j + 1)
    Pg = paper_gfout0(P) + pref * Sg
    Pb0 = 1 - mp.e ** (-db / (rb * Ob))
    Pb = Pb0 + mp.e ** (-db / (rb * Ob)) / Ob * vt(1 / Ob, rb / db) - mp.e ** (-x0 / Ob) / Ob * vs(1 / Ob) \
        + mp.fsum(mp.binomial(K, j) * phi ** (K - j) * mp.e ** (-x0 / Ob - j * th / Og) / Ob * vs(cm(j)) for j in range(K + 1)) \
        + pref * Sb
    return Pg, Pb

def paper_thm4(P):
    rb, rg, Ob, Og, K, db, dg = P
    th = dg / rg; x0 = db * (1 + dg) / rb; phi = 1 - mp.e ** (-th / Og)
    nu = dg * rb / (rg * Og); chi = lambda xx: th * (1 + rb * xx)
    cn = lambda n: 1 / Ob + n * rb / (db * rg * Og)
    fin = db * dg < 1
    x1 = x0 / (1 - db * dg) if fin else mp.inf
    S1 = mp.fsum(mp.binomial(K, i) * (-1) ** i * mp.e ** (-i * th / Og) * (1 - mp.e ** (-(1 / Ob + i * nu) * x0)) / (Ob * (1 / Ob + i * nu)) for i in range(K + 1))
    S2 = mp.mpf(0)
    for n in range(K + 1):
        for p in range(K + 1 - n):
            coef = mp.factorial(K) * (-1) ** p * phi ** (K - n - p) / (mp.factorial(K - n - p) * mp.factorial(n) * mp.factorial(p))
            t0 = mp.e ** (-x0 / Ob - (n + p) * th / Og - p * nu * x0)
            t1 = mp.e ** (-x1 / Ob - (n + p) * chi(x1) / Og) if fin else mp.mpf(0)
            S2 += coef * (t0 - t1) / (Ob * (cn(n) + p * nu))
    S3 = phi ** K * mp.e ** (-x1 / Ob) if fin else mp.mpf(0)
    return S1 + S2 + S3

# ----------------------------------------------------------------- numeric: first principles
def direct_thm4(P):
    rb, rg, Ob, Og, K, db, dg = P
    th = dg / rg
    def q(xx):   # Pr(a given user is NOT served | x): served iff gain in [theta,psi] (if psi>=theta) or gain >= chi
        psi = (rb * xx / db - 1) / rg; chi = th * (1 + rb * xx)
        F = lambda g: 1 - mp.e ** (-g / Og)
        served = mp.mpf(0)
        lo, hi = th, max(psi, th)               # branch A interval (empty if psi<theta)
        if hi > lo:
            served += F(min(hi, chi)) - F(lo) if chi > lo else 0   # part of [lo,hi] below chi
        served += 1 - F(max(chi, lo if hi > lo and chi < lo else chi))  # [chi, inf), avoid double count
        # simpler & exact: union of [lo,hi] and [chi,inf)
        if hi > lo:
            if chi <= hi:  served = 1 - F(min(lo, chi)) if chi < lo else 1 - F(lo)
            else:          served = (F(hi) - F(lo)) + (1 - F(chi))
        else:
            served = 1 - F(chi)
        return 1 - served
    f = lambda xx: mp.e ** (-xx / Ob) / Ob * q(xx) ** K
    x0 = db * (1 + dg) / rb
    pts = [0, x0] + ([x0 / (1 - db * dg)] if db * dg < 1 else []) + [mp.inf]
    return mp.quad(f, pts)

def g_tail(w, cc):   # Pr(W > w) = int_0^1 exp(-w/(uc)) du = e^{-w/c} - (w/c) E1(w/c)
    if w <= 0: return mp.mpf(1)
    z = w / cc
    return mp.e ** (-z) - z * mp.expint(1, z)

def direct_thm5(P, cc):
    rb, rg, Ob, Og, K, db, dg = P
    th = dg / rg; x0 = db * (1 + dg) / rb; xa = db / rb
    fx = lambda xx: mp.e ** (-xx / Ob) / Ob
    psi = lambda xx: (rb * xx / db - 1) / rg
    def fM(mm, xx):
        return K / Og * mp.e ** (-mm / Og) * (1 - mp.e ** (-mm / Og) + mp.e ** (-psi(xx) / Og)) ** (K - 1)
    # GF: outage increase = E[ Pr(W > min(Gb,Gg)) ; A nonempty ]
    def inner_g(xx):
        p = psi(xx); mc = (p + 1 / rg) * dg / (1 + dg)
        f1 = lambda mm: g_tail(rg * mm / dg - 1, cc) * fM(mm, xx)      # m <= mc: GF margin binds
        f2 = lambda mm: g_tail(rg * (p - mm), cc) * fM(mm, xx)         # m >= mc: GB margin binds
        return mp.quad(f1, [th, mc]) + mp.quad(f2, [mc, p])
    Pg = paper_gfout0(P) + mp.quad(lambda xx: fx(xx) * inner_g(xx), [x0, mp.inf])
    # GB: Pr(x<xa) + E[Pr(W > Gb)]
    def inner_b(xx):
        p = psi(xx)
        if xx < x0:
            return g_tail(rg * p, cc)
        pA0 = (1 - mp.e ** (-th / Og) + mp.e ** (-p / Og)) ** K
        return pA0 * g_tail(rg * p, cc) + mp.quad(lambda mm: g_tail(rg * (p - mm), cc) * fM(mm, xx), [th, p])
    Pb = (1 - mp.e ** (-xa / Ob)) + mp.quad(lambda xx: fx(xx) * inner_b(xx), [xa, x0, mp.inf])
    return Pg, Pb

def slopes(P):   # Proposition 3
    rb, rg, Ob, Og, K, db, dg = P
    th = dg / rg; x0 = db * (1 + dg) / rb; phi = 1 - mp.e ** (-th / Og)
    cm = lambda mm: 1 / Ob + mm * rb / (db * rg * Og)
    ft = db / (rb * Ob) * mp.e ** (-db / (rb * Ob))
    ftop = K * mp.e ** (-x0 / Ob - th / Og) / (rg * Og * Ob * cm(1))
    fbot = K * th / Og * mp.e ** (-th / Og) * mp.fsum(mp.binomial(K - 1, j) * phi ** (K - 1 - j) * mp.e ** (-x0 / Ob - j * th / Og) / (Ob * cm(j)) for j in range(K))
    return (ft + ftop) / 2, (ftop + fbot) / 2

# parameter sets (rb, rg, Ob, Og, K, db, dg): paper point (29 dB GB SNR, 26.7 dB GF, K=5, Rb=1, Rg=4) + two others
mpf = mp.mpf
sets = {
 "paper_default(K5,Rb1,Rg4)": (mpf(10) ** 2.9, mpf(10) ** 2.67, mpf(1), mpf(1), 5, mpf(1), mpf(15)),
 "lowrate(K3,Rb1,Rg1)":       (mpf(100), mpf(50), mpf(1), mpf('0.7'), 3, mpf(1), mpf(1)),
 "dbdg<1(K4,Rb0.5,Rg0.5)":    (mpf(2000), mpf(100), mpf(1), mpf('0.7'), 4, mpf(2) ** mpf('0.5') - 1, mpf(2) ** mpf('0.5') - 1),
 "K2 high rg":                (mpf(50), mpf(400), mpf(1), mpf('0.7'), 2, mpf(1), mpf(15)),
}
res = {}
worst4 = worst5 = 0
for name, P in sets.items():
    r = {}
    t4p, t4d = paper_thm4(P), direct_thm4(P)
    r["thm4"] = dict(paper=float(t4p), direct=float(t4d), absdiff=float(abs(t4p - t4d)))
    worst4 = max(worst4, abs(t4p - t4d))
    print(f"[T4] {name:28s} paper={mp.nstr(t4p,12)} direct={mp.nstr(t4d,12)} diff={mp.nstr(abs(t4p-t4d),3)}")
    for cc in (mpf('0.01'), mpf('0.2'), mpf('2'), mpf('20')):
        Pg_p, Pb_p = paper_thm5(P, cc); Pg_d, Pb_d = direct_thm5(P, cc)
        dgf, dgb = abs(Pg_p - Pg_d), abs(Pb_p - Pb_d)
        worst5 = max(worst5, dgf, dgb)
        r[f"thm5_c{float(cc)}"] = dict(Pg_paper=float(Pg_p), Pg_direct=float(Pg_d), Pb_paper=float(Pb_p), Pb_direct=float(Pb_d),
                                      diff_g=float(dgf), diff_b=float(dgb))
        print(f"[T5] {name:28s} c={float(cc):<5} Pg paper={mp.nstr(Pg_p,12)} direct={mp.nstr(Pg_d,12)} | Pb paper={mp.nstr(Pb_p,12)} direct={mp.nstr(Pb_d,12)} | diffs {mp.nstr(dgf,3)} {mp.nstr(dgb,3)}")
    # Proposition 3 slopes vs numerical derivative of the direct integrals at c -> 0
    Db, Dg = slopes(P); h = mpf('1e-4')
    Pg_h, Pb_h = direct_thm5(P, h); Pg_0, Pb_0 = paper_gfout0(P), 1 - mp.e ** (-P[5] / (P[0] * P[2]))
    r["prop3"] = dict(Db=float(Db), Db_num=float((Pb_h - Pb_0) / h), Dg=float(Dg), Dg_num=float((Pg_h - Pg_0) / h))
    print(f"[P3] {name:28s} Db={mp.nstr(Db,6)} num={mp.nstr((Pb_h-Pb_0)/h,6)} | Dg={mp.nstr(Dg,6)} num={mp.nstr((Pg_h-Pg_0)/h,6)}")
    res[name] = r
out["numeric"] = res; out["worst_abs_diff"] = dict(thm4=float(worst4), thm5=float(worst5))
print("WORST", float(worst4), float(worst5))
json.dump(out, open("../figs/audit_thm45.json", "w"), indent=1)
