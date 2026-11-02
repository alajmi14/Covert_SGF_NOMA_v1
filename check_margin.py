"""Third benchmark (reviewer round 2, point 3): max-margin selection versus the strongest-user rule
and uniform selection within A, under residual jamming. Writes figs/margin_select_check.json."""
import json
import numpy as np
from model import *

beta0_dB, alpha, N0_dBm = -30.0, 3.0, -90.0
Om = lambda dist: 10 ** (beta0_dB / 10) * dist ** (-alpha)
snr = lambda P_dBm: 10 ** ((P_dBm - N0_dBm) / 10)
Om_b, Om_g, Om_w, Om_J, Om_JB = Om(50.), Om(60.), Om(100.), Om(30.), Om(80.)
Pb, Pmax, K, Rb = 20.0, 23.0, 5, 1.0

out = {}
for PJv, e in [(20, 0.01), (20, 0.1), (10, 0.01), (10, 0.1)]:
    cap = min(rho_g_covert_max(e, snr(PJv), Om_w, Om_J), snr(Pmax))
    v, rg, Rg = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap)
    for z in (0.0, 0.01, 0.1):
        row = {"Rg": float(Rg)}
        for s in ("best", "margin", "random"):
            pg, pb = mc_sgf(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg, n=10 ** 6, beta=z, rho_J=snr(PJv),
                            Om_JB=Om_JB, select=s, rng=11)
            row[s] = dict(ect=float(Rg * (1 - pg)), Pout_gf=float(pg), Pout_gb=float(pb))
        out[f"PJ{PJv}_eps{e}_zeta{z}"] = row
        print(PJv, e, z, {s: (round(row[s]["ect"], 3), round(row[s]["Pout_gb"], 5)) for s in ("best", "margin", "random")})
json.dump(out, open("../figs/margin_select_check.json", "w"), indent=1)
