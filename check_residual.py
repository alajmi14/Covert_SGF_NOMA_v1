"""Verification of residual.py against the Monte-Carlo simulator (BS unaware of the residual)."""
import numpy as np, json, time
from model import *
from residual import *

beta0_dB, alpha, N0_dBm = -30.0, 3.0, -90.0
d = dict(b=50.0, g=60.0, w=100.0, JW=30.0, JB=80.0, bw=120.0)
Om = lambda dist: 10 ** (beta0_dB / 10) * dist ** (-alpha)
snr = lambda P_dBm: 10 ** ((P_dBm - N0_dBm) / 10)
dB = lambda x: 10 * np.log10(x)
Om_b, Om_g, Om_w, Om_J, Om_JB = Om(d["b"]), Om(d["g"]), Om(d["w"]), Om(d["JW"]), Om(d["JB"])
Pb, Pmax, K, Rb, eps, RMAX = 20.0, 23.0, 5, 1.0, 0.1, 4.0
rng = np.random.default_rng(5)
out = {}

print("=== A. exact quadrature vs first-order vs MC on the Fig. 5 grid (operating point from Algorithm 1)")
rows = []
for PJ in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]:
    cap = min(rho_g_covert_max(eps, snr(PJ), Om_w, Om_J), snr(Pmax))
    _, rg, Rg = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap, n_rho=200)
    Db, Dg, parts = residual_slopes(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg)
    pb0 = gb_outage(snr(Pb), Om_b, Rb); pg0 = gf_outage(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg)
    for zeta in [1e-3, 1e-2]:
        c = zeta * snr(PJ) * Om_JB
        t = time.time()
        pb_ex = gb_outage_residual(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg, c)
        pg_ex = gf_outage_residual(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg, c)
        tq = time.time() - t
        po_mc, pb_mc = mc_sgf(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg, n=10 ** 6, beta=zeta, rho_J=snr(PJ), Om_JB=Om_JB, rng=rng)
        row = dict(PJ=PJ, zeta=zeta, c=c, Pg_dBm=dB(rg) + N0_dBm, Rg=Rg,
                   gb_mc=pb_mc, gb_exact=pb_ex, gb_lin=pb0 + Db * c,
                   gf_mc=po_mc, gf_exact=pg_ex, gf_lin=pg0 + Dg * c,
                   ect_mc=Rg * (1 - po_mc), ect_exact=Rg * (1 - pg_ex), ect_lin=Rg * (1 - pg0 - Dg * c))
        rows.append(row)
        print(f"PJ={PJ:4.0f} zeta={zeta:.0e} c={c:8.3f} | GB: mc {pb_mc:.2e} exact {pb_ex:.2e} lin {row['gb_lin']:.2e}"
              f" | GF: mc {po_mc:.4f} exact {pg_ex:.4f} lin {row['gf_lin']:.4f} | ECT mc {row['ect_mc']:.3f} ex {row['ect_exact']:.3f} ({tq:.1f}s)")
out["grid"] = rows

print("\n=== B. slopes at the unconstrained optimum and the zeta-aware jamming power")
vu, rgu, Rgu = optimise(snr(Pb), Om_b, Om_g, K, Rb, snr(Pmax))
Db, Dg, parts = residual_slopes(snr(Pb), rgu, Om_b, Om_g, K, Rb, Rgu)
H = curvature_logpower(snr(Pb), rgu, Om_b, Om_g, K, Rb, Rgu)
rho_J_th = rho_eps(eps) * Om_w / Om_J * rgu
print("Db", Db, "Dg", Dg, parts, "H", H, "PJth dBm", dB(rho_J_th) + N0_dBm)
out["slopes"] = dict(Db=Db, Dg=Dg, H=H, PJth_dBm=dB(rho_J_th) + N0_dBm, **{k: float(v) for k, v in parts.items()})
for zeta in [1e-4, 1e-3, 1e-2]:
    c_th = zeta * rho_J_th * Om_JB
    rJo, v = pj_opt_residual(rho_J_th, c_th, Dg, H)
    # brute force: exact ECT(P_J) with the covertness cap active below P_J^th
    PJs = np.linspace(dB(rho_J_th) + N0_dBm - 3, dB(rho_J_th) + N0_dBm + 0.5, 36)
    ects = []
    for PJ in PJs:
        cap = min(rho_g_covert_max(eps, snr(PJ), Om_w, Om_J), snr(Pmax))
        rg = min(cap, rgu)
        c = zeta * snr(PJ) * Om_JB
        ects.append(Rgu * (1 - gf_outage_residual(snr(Pb), rg, Om_b, Om_g, K, Rb, Rgu, c)))
    i = int(np.argmax(ects))
    print(f"zeta={zeta:.0e}: c_th={c_th:.4f} v*={v:.4f} -> P_J^opt={dB(rJo)+N0_dBm:.2f} dBm ; brute force max at {PJs[i]:.2f} dBm, "
          f"ECT {ects[i]:.4f} vs at P_J^th {ects[-6]:.4f} (loss vs zeta=0: {vu-ects[i]:.4f}; linear pred {Rgu*Dg*c_th:.4f})")
    out[f"opt_zeta{zeta:g}"] = dict(c_th=c_th, v=v, PJopt_dBm=dB(rJo) + N0_dBm, brute_dBm=float(PJs[i]), ect_brute=float(ects[i]))
json.dump(out, open("../figs/residual_check.json", "w"), indent=1)
