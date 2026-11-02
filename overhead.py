"""Signaling overhead of the keyed sounding (reviewer round 2, point 2).
Effective ECT eta_eff = eta * n/(n + n_0) with n_0 = L*ceil(N_reg/M)/T + n_gr(N_reg) + n_p, versus the number of
registered GF users N_reg, with the number of active users K fixed. Writes figs/fig_overhead.pdf and
figs/overhead.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from model import *

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.3, "legend.fontsize": 6.5, "lines.linewidth": 1.1,
                     "figure.figsize": (3.5, 2.6), "savefig.bbox": "tight", "mathtext.fontset": "cm"})
beta0_dB, alpha, N0_dBm = -30.0, 3.0, -90.0
Om = lambda dist: 10 ** (beta0_dB / 10) * dist ** (-alpha)
snr = lambda P_dBm: 10 ** ((P_dBm - N0_dBm) / 10)
Om_b, Om_g, Om_w, Om_J = Om(50.), Om(60.), Om(100.), Om(30.)
Pb, Pmax, K, Rb = 20.0, 23.0, 5, 1.0
n, L, M, n_p, b_tag = 1000, 64, 8, 8, 32   # data channel uses, sounding length, users per slot, jammer pilots, tag bits
# grant field: index of the selected user among N_reg registered users + tag, at a net 2 bit per channel use
b_idx = lambda N: np.maximum(1, np.ceil(np.log2(N)))
n_gr = lambda N: np.ceil((b_idx(N) + b_tag) / 2)
Nreg = np.arange(5, 501, 5)
n0 = lambda N, T: L * np.ceil(N / M) / T + n_gr(N) + n_p
eff = lambda N, T: n / (n + n0(N, T))

log = dict(n=n, L=L, M=M, n_p=n_p, b_tag=b_tag, n_gr={int(N): int(n_gr(N)) for N in (5, 100, 256, 257, 500, 632)})
fig, ax = plt.subplots()
for (PJv, e, c) in [(20, 0.1, "C0"), (10, 0.01, "C2")]:
    cap = min(rho_g_covert_max(e, snr(PJv), Om_w, Om_J), snr(Pmax))
    eta, rg, Rg = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap)
    for T, ls, mk in [(1, "-", "o"), (8, "--", "s")]:
        ax.plot(Nreg, eta * eff(Nreg, T), c + ls, marker=mk, mfc="none", ms=3, markevery=10)
        log[f"PJ{PJv}_eps{e}_T{T}"] = {int(N): float(eta * eff(N, T)) for N in (5, 20, 50, 100, 200, 500)}
    log[f"PJ{PJv}_eps{e}_eta"] = float(eta)
    ax.axhline(eta, color=c, lw=0.6, ls=":")
h = [Line2D([], [], c="C0", label=r"$P_J^{\max}=20$ dBm, $\epsilon=0.1$"),
     Line2D([], [], c="C2", label=r"$P_J^{\max}=10$ dBm, $\epsilon=0.01$"),
     Line2D([], [], c="gray", ls="-", marker="o", mfc="none", ms=3, label=r"sounding every block ($T=1$)"),
     Line2D([], [], c="gray", ls="--", marker="s", mfc="none", ms=3, label=r"sounding every 8 blocks ($T=8$)"),
     Line2D([], [], c="gray", ls=":", lw=0.6, label=r"no overhead ($\eta$)")]
ax.legend(handles=h, loc="upper center", ncol=2, fontsize=6)
ax.set_xlim(0, 500); ax.set_ylim(0, 5.2)
ax.set_xlabel(r"Registered GF users $N_{\mathrm{reg}}$ ($K=5$ active)")
ax.set_ylabel(r"Effective throughput $\eta_{\mathrm{eff}}$ (bit/s/Hz)")
fig.savefig("../figs/fig_overhead.pdf"); plt.close(fig)
# efficiency factors and activity-detection error bound Q(sqrt(L*delta_g)) for an admissible user
log["eff_T1"] = {int(N): float(eff(N, 1)) for N in (5, 20, 50, 100, 200, 500)}
log["eff_T8"] = {int(N): float(eff(N, 8)) for N in (5, 20, 50, 100, 200, 500)}
log["Pe_admissible_bound_L64_dg15"] = float(norm.sf(np.sqrt(L * 15.0)))
log["Pe_admissible_bound_L16_dg15"] = float(norm.sf(np.sqrt(16 * 15.0)))
log["sounding_duty_T1"] = L / (n + n0(100, 1)); log["sounding_duty_T8"] = L / (8 * (n + n0(100, 8)))
json.dump(log, open("../figs/overhead.json", "w"), indent=1)
print(json.dumps(log, indent=1))
cap = min(rho_g_covert_max(0.01, snr(20), Om_w, Om_J), snr(Pmax))
eta01 = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap)[0]
Ns = np.arange(5, 5001)
for T in (1, 8):
    ok = Ns[eta01 * eff(Ns, T) >= 2.30]
    log[f"crossover_vs_benchmark_ii_T{T}"] = int(ok.max())
log["eta_PJ20_eps0.01"] = float(eta01)
log["false_accept_network"] = {int(N): float(N * 2.0 ** -32) for N in (100, 500)}
json.dump(log, open("../figs/overhead.json", "w"), indent=1)
print({k: log[k] for k in ("n_gr", "eff_T1", "eff_T8", "sounding_duty_T1", "sounding_duty_T8", "crossover_vs_benchmark_ii_T1", "crossover_vs_benchmark_ii_T8", "false_accept_network", "eta_PJ20_eps0.01")})
