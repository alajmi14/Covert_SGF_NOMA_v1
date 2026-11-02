"""Closed-form numbers added in the audit revision (Remarks on the DEP tail, warden-position
robustness, GB-power randomisation, Table III sensitivity, jammer cost).  Writes figs/extra.json."""
import json
import numpy as np
from scipy.special import exp1
from model import *

beta0_dB, alpha, N0_dBm = -30.0, 3.0, -90.0
d = dict(b=50.0, g=60.0, w=100.0, JW=30.0, JB=80.0, bw=120.0)
Om = lambda dist: 10 ** (beta0_dB / 10) * dist ** (-alpha)
snr = lambda P_dBm: 10 ** ((P_dBm - N0_dBm) / 10)
dB = lambda x: 10 * np.log10(x)
Om_b, Om_g, Om_w, Om_J, Om_JB, Om_bw = (Om(d["b"]), Om(d["g"]), Om(d["w"]), Om(d["JW"]),
                                        Om(d["JB"]), Om(d["bw"]))
Pb, PJ, Pmax, K, Rb, eps, RMAX = 20.0, 20.0, 23.0, 5, 1.0, 0.1, 4.0
out = {}

# ---- B1: tail of the per-block DEP, P(xi* <= c) = 1/(1+(1-c)kappa)
ke = rho_eps(eps)
out["kappa_eps"] = ke
out["P_xi0"] = 1 / (1 + ke)                       # detected with certainty
out["P_xi_le_0.9"] = 1 / (1 + 0.1 * ke)
out["P_xi_le_0.5"] = 1 / (1 + 0.5 * ke)
out["N_for_half"] = np.log(0.5) / np.log(ke / (1 + ke))   # accesses until P(>=1 certain detection)=1/2
out["kappa_outage_eps0.1_delta0.05"] = (1 - 0.05) / (0.05 * 0.1)   # Pr(xi*<1-eps')<=delta -> kappa >= (1-delta)/(delta eps')

# ---- B3: warden-position robustness (design d_gw=100, d_Jw=30, eps=0.1, P_g at the cap)
rob = {}
for dgw, djw in [(100, 30), (100, 45), (100, 60), (100, 100), (60, 60), (80, 30), (60, 30)]:
    kap = ke * (Om(djw) / Om(d["JW"])) * (Om(d["w"]) / Om(dgw))   # kappa_true = kappa_eps * (OmJ_true/OmJ)*(Omw/Omw_true)
    rob[f"dgw{dgw}_dJw{djw}"] = dict(kappa=float(kap), dep=float(1 - np.log1p(kap) / kap), P_xi0=float(1 / (1 + kap)))
out["robust"] = rob
# robust design for d_Jw in [30, 60] m: use the smallest Om_J -> cap reduced by (60/30)^3 = 9 dB; ECT cost
cap_nom = min(rho_g_covert_max(eps, snr(PJ), Om_w, Om_J), snr(Pmax))
cap_rob = min(rho_g_covert_max(eps, snr(PJ), Om_w, Om(60.0)), snr(Pmax))
v_nom = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap_nom)
v_rob = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap_rob)
out["robust_design"] = dict(cap_nom_dBm=float(dB(cap_nom) + N0_dBm), cap_rob_dBm=float(dB(cap_rob) + N0_dBm),
                            ect_nom=float(v_nom[0]), ect_rob=float(v_rob[0]),
                            Pg_nom_dBm=float(dB(v_nom[1]) + N0_dBm), Pg_rob_dBm=float(dB(v_rob[1]) + N0_dBm),
                            PJth_rob_dBm=float(dB(ke * Om_w / Om(60.0) * v_nom[1]) + N0_dBm))

# ---- B4: GB-power randomisation P_b ~ U[0, P_b^max] as the cover
cap_gb = snr(PJ) * Om_bw / (ke * Om_w)             # P_J^max -> P_b^max (same 20 dBm), Om_J -> Om_bw
out["gbrand_cap_dBm"] = float(dB(cap_gb) + N0_dBm)
out["jammer_cap_dBm"] = float(dB(rho_g_covert_max(eps, snr(PJ), Om_w, Om_J)) + N0_dBm)
out["jammer_advantage_dB"] = float(dB(Om_J / Om_bw))
u = (2 ** Rb - 1) / (snr(Pb) * Om_b)               # OMA outage exponent at P_b^max
out["gb_outage_OMA"] = float(1 - np.exp(-u))
out["gb_outage_rand"] = float(1 - np.exp(-u) + u * exp1(u))   # E_P[1-exp(-c/P)], P~U[0,Pmax]
# GB-side randomisation with the range needed for the cap to reach P_g^u = 9.7 dBm
vu = optimise(snr(Pb), Om_b, Om_g, K, Rb, snr(Pmax))
out["Pgu_dBm"] = float(dB(vu[1]) + N0_dBm)
out["gbrand_range_needed_dBm"] = float(dB(ke * Om_w / Om_bw * vu[1]) + N0_dBm)

# ---- B5: sensitivity table (closed-form optimiser)
tab = {}
for Rm in [2.0, 3.0, 4.0, 6.0, 12.0]:
    v, rg, Rg = optimise(snr(Pb), Om_b, Om_g, K, Rb, snr(Pmax), R_max=Rm)
    tab[f"Rmax{Rm:g}"] = dict(ect=float(v), Rg=float(Rg), Pg_dBm=float(dB(rg) + N0_dBm),
                              pout=float(gf_outage(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg)),
                              PJth_dBm=float(dB(ke * Om_w / Om_J * rg) + N0_dBm))
for Rbv in [1.0, 2.0, 3.0]:
    v, rg, Rg = optimise(snr(Pb), Om_b, Om_g, K, Rbv, snr(Pmax), R_max=RMAX)
    tab[f"Rb{Rbv:g}"] = dict(ect=float(v), Rg=float(Rg), Pg_dBm=float(dB(rg) + N0_dBm),
                             pout=float(gf_outage(snr(Pb), rg, Om_b, Om_g, K, Rbv, Rg)),
                             PJth_dBm=float(dB(ke * Om_w / Om_J * rg) + N0_dBm),
                             gb_out=float(gb_outage(snr(Pb), Om_b, Rbv)))
out["table3"] = tab

# ---- B7: jammer cost at P_J^max = P_J^th
PJth = tab["Rmax4"]["PJth_dBm"]
out["jammer_avg_mW_at_PJth"] = float(10 ** (PJth / 10) / 2)
out["gf_power_mW_at_Pgu"] = float(10 ** (tab["Rmax4"]["Pg_dBm"] / 10))
out["P_H1_at_opt"] = float(1 - tab["Rmax4"]["pout"])

# ---- GF-first branch activity and no-admissible-user probability at the operating points (MC)
rng = np.random.default_rng(11)
act = {}
for Pg in [tab["Rmax4"]["Pg_dBm"], 16.13, 20.11, 23.0]:
    n = 10 ** 6
    hb = rng.exponential(Om_b, n); hk = rng.exponential(Om_g, (n, K))
    tau = np.maximum(0.0, snr(Pb) * hb / (2 ** Rb - 1) - 1.0)
    rx = snr(Pg) * hk
    act[f"Pg{Pg:.1f}"] = dict(above_tau=float((rx > tau[:, None]).any(axis=1).mean()),
                              none_adm=float((~(rx <= tau[:, None])).all(axis=1).mean()))
out["gf_first_activity"] = act

json.dump(out, open("../figs/extra.json", "w"), indent=1)
print(json.dumps(out, indent=1))
