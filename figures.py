"""Generates all numerical-result figures of the paper (analysis + Monte Carlo).
Every number quoted in Section VI of the paper is written to figs/numbers.json.
Block numbers below are historical and differ from the paper's figure numbers (submission version, 14 pages):
block 2 = paper Fig. 2 (DEP), block 3 = Fig. 3 (outage vs SNR), block 8 = Fig. 4 (hybrid SIC), block 7 = Fig. 5
(residual), block 5 = Fig. 6 (ECT vs eps), block 9 = Fig. 7 (robust design).  Blocks 4 (outage vs P_g, R_g = 1)
and 6 (ECT vs P_J^max) are no longer shown in the paper; their numbers are quoted in the text (numbers.json)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model import *

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.3, "legend.fontsize": 6.5, "lines.linewidth": 1.1,
                     "figure.figsize": (3.5, 2.6), "savefig.bbox": "tight",
                     "mathtext.fontset": "cm"})
FIG = "../figs/"
# one independent random stream per figure, so that rerun_figs.py reproduces every figure bit-exactly
rng_fig = {k: np.random.default_rng(2026 + k) for k in range(3, 10)}

# ------------------------------------------------------------ default setup (Table II of the paper)
beta0_dB, alpha, N0_dBm = -30.0, 3.0, -90.0
d = dict(b=50.0, g=60.0, w=100.0, JW=30.0, JB=80.0, bw=120.0)
Om = lambda dist: 10 ** (beta0_dB / 10) * dist ** (-alpha)
snr = lambda P_dBm: 10 ** ((P_dBm - N0_dBm) / 10)
dB = lambda x: 10 * np.log10(x)
Om_b, Om_g, Om_w, Om_J, Om_JB, Om_bw = (Om(d["b"]), Om(d["g"]), Om(d["w"]), Om(d["JW"]),
                                        Om(d["JB"]), Om(d["bw"]))
Pb, PJ, Pmax, K, Rb, eps, RMAX = 20.0, 20.0, 23.0, 5, 1.0, 0.1, 4.0
NMC = 10 ** 6                      # channel realisations for every outage/ECT simulation
log = {"R_max": RMAX, "NMC": NMC}

# ------------------------------------------------------------ Fig. 2: DEP
rng_dep = np.random.default_rng(7)  # separate stream: other figures do not depend on it
fig, ax = plt.subplots()
PJ_ax = np.linspace(0, 30, 61)
PJ_mc = np.arange(0, 31, 5)
for Pg, c in zip([10, 15, 20], ["C0", "C1", "C2"]):
    ax.plot(PJ_ax, dep_avg(snr(Pg), snr(PJ_ax), Om_w, Om_J), c,
            label=rf"Analysis, $P_g={Pg}$ dBm")
    mc = [mc_dep(snr(Pg), snr(p), Om_w, Om_J, rho_b=snr(Pb), Om_bw=Om_bw, rng=rng_dep)
          for p in PJ_mc]
    ax.plot(PJ_mc, mc, c + "o", mfc="none", ms=4)
    log[f"dep_mc_Pg{Pg}"] = list(map(float, mc))
    log[f"dep_an_Pg{Pg}"] = list(map(float, dep_avg(snr(Pg), snr(PJ_mc), Om_w, Om_J)))
# fading average as printed in [shahzad2018achieving, Lemma 2] (P_min = 0, unweighted DEP), for P_g = 15 dBm
t_sh = 1.0 / (1.0 + snr(PJ_ax) * Om_J / (snr(15) * Om_w))
ax.plot(PJ_ax, 1 + t_sh * np.log(t_sh) - t_sh ** 2, "C1--", lw=0.9,
        label=r"Shahzad et al., Lemma 2, $P_g=15$ dBm")
log["shahzad_lemma2_max_gap"] = float(np.max(1 + t_sh * np.log(t_sh) - t_sh ** 2
                                             - dep_avg(snr(15), snr(PJ_ax), Om_w, Om_J)))
# finite blocklength (simulation only), P_g = 15 dBm
for nn, mk in [(100, "^"), (1000, "v")]:
    mc = [mc_dep(snr(15), snr(p), Om_w, Om_J, rho_b=snr(Pb), Om_bw=Om_bw, n=nn, rng=rng_dep)
          for p in PJ_mc]
    ax.plot(PJ_mc, mc, "C1:", marker=mk, mfc="none", ms=4, lw=0.8, label=rf"$n={nn}$, $P_g=15$ dBm (sim.)")
    log[f"dep_mc_Pg15_n{nn}"] = list(map(float, mc))
ax.plot([], [], "ko", mfc="none", ms=4, label=r"Simulation, $n\to\infty$")
ax.axhline(1 - eps, ls=":", c="k", lw=0.8)
ax.text(1, 1 - eps + 0.015, r"$1-\epsilon$ ($\epsilon=0.1$)", fontsize=6.5)
ax.set_xlabel(r"Maximum jamming power $P_J^{\max}$ (dBm)")
ax.set_ylabel(r"Average minimum DEP $\overline{\xi}^{*}$")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")
fig.savefig(FIG + "fig_dep.pdf"); plt.close(fig)

# ------------------------------------------------------------ Fig. 3: GF outage vs P_b (P_g = P_b)
fig, ax = plt.subplots()
Pb_ax = np.linspace(0, 40, 81)
Pb_mc = np.arange(0, 41, 5)
for Kk, c in zip([1, 3, 5, 10], ["C0", "C1", "C2", "C3"]):
    ax.semilogy(Pb_ax, gf_outage(snr(Pb_ax), snr(Pb_ax), Om_b, Om_g, Kk, Rb, 1.0), c,
                label=rf"$K={Kk}$")
    mc = [mc_sgf(snr(p), snr(p), Om_b, Om_g, Kk, Rb, 1.0, n=NMC, rng=rng_fig[3])[0] for p in Pb_mc]
    ax.semilogy(Pb_mc, mc, c + "o", mfc="none", ms=4)
    log[f"fig3_mc_K{Kk}"] = list(map(float, mc)); log["fig3_P_mc_dBm"] = list(map(float, Pb_mc))
    ax.axhline(gf_outage_floor(Om_b, Om_g, Kk, Rb), c=c, ls="--", lw=0.7)
    log[f"floor_K{Kk}"] = float(gf_outage_floor(Om_b, Om_g, Kk, Rb))
ax.semilogy(Pb_ax, gb_outage(snr(Pb_ax), Om_b, Rb), "k-.", label="GB user (= OMA)")
ax.plot([], [], "ko", mfc="none", ms=4, label="Simulation")
ax.plot([], [], "k--", lw=0.7, label="Error floor (Cor. 2)")
ax.set_xlabel(r"Transmit power $P_b=P_g$ (dBm)")
ax.set_ylabel("Outage probability")
ax.set_ylim(1e-3, 1.1)
ax.legend(loc="lower left", ncol=2)
fig.savefig(FIG + "fig_outage_snr.pdf"); plt.close(fig)

# ------------------------------------------------------------ Fig. 4: GF outage vs P_g, covert limits
fig, ax = plt.subplots()
Pg_ax = np.linspace(-5, 35, 161)
Pg_mc = np.arange(-5, 36, 5)
for (Kk, Pbb), c in zip([(1, 15), (1, 25), (5, 15), (5, 25)], ["C0", "C1", "C2", "C3"]):
    an = gf_outage(snr(Pbb), snr(Pg_ax), Om_b, Om_g, Kk, Rb, 1.0)
    ax.semilogy(Pg_ax, an, c, label=rf"$K={Kk}$, $P_b={Pbb}$ dBm")
    log[f"fig4_argmin_K{Kk}_Pb{Pbb}_dBm"] = float(Pg_ax[np.argmin(an)])
    mc = [mc_sgf(snr(Pbb), snr(p), Om_b, Om_g, Kk, Rb, 1.0, n=NMC, rng=rng_fig[4])[0] for p in Pg_mc]
    log[f"fig4_mc_K{Kk}_Pb{Pbb}"] = list(map(float, mc)); log["fig4_Pg_mc_dBm"] = list(map(float, Pg_mc))
    ax.semilogy(Pg_mc, mc, c + "o", mfc="none", ms=4)
for e, ls in [(0.05, ":"), (0.1, "-.")]:
    cap = dB(rho_g_covert_max(e, snr(PJ), Om_w, Om_J)) + N0_dBm
    ax.axvline(cap, c="k", ls=ls, lw=0.8)
    ax.text(cap + 0.4, 1.3e-3, rf"$\overline{{P}}_g(\epsilon={e})$", fontsize=6.5, rotation=90)
    log[f"Pg_cap_eps{e}_dBm"] = float(cap)
ax.plot([], [], "ko", mfc="none", ms=4, label="Simulation")
ax.set_xlabel(r"GF transmit power $P_g$ (dBm)")
ax.set_ylabel("GF outage probability")
ax.set_ylim(1e-3, 1.1)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6)
fig.savefig(FIG + "fig_outage_pg.pdf"); plt.close(fig)

# ------------------------------------------------------------ Fig. 5: ECT vs eps, benchmarks
eps_ax = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3])
fig, ax = plt.subplots(figsize=(3.5, 2.9))
res = {}
for PJv, ls in [(20, "-"), (10, "--")]:
    opt, fixd, rnd, mc_opt, mc_rnd = [], [], [], [], []
    for e in eps_ax:
        cap = min(rho_g_covert_max(e, snr(PJv), Om_w, Om_J), snr(Pmax))
        v, rg, Rg = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap)
        opt.append(v)
        # benchmark (i): GF user always transmits at the largest covert power, R_g optimised
        fixd.append(max(ect(snr(Pb), cap, Om_b, Om_g, K, Rb, R) for R in np.arange(0.05, RMAX + 1e-9, 0.05)))
        # benchmark (ii): random admitted GF user, with ITS OWN optimal (P_g, R_g)
        vr, rgr, Rgr = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap, outage=gf_outage_random, n_rho=120)
        rnd.append(vr)
        po_r, _ = mc_sgf(snr(Pb), rgr, Om_b, Om_g, K, Rb, Rgr, n=NMC, select="random", rate_aware=False, rng=rng_fig[5])
        mc_rnd.append(Rgr * (1 - po_r))
        po, _ = mc_sgf(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg, n=NMC, rng=rng_fig[5])
        mc_opt.append(Rg * (1 - po))
        res[f"{PJv}dBm_eps{e}"] = dict(ect=float(v), Pg_dBm=float(dB(rg) + N0_dBm), Rg=float(Rg),
                                       pout=float(gf_outage(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg)),
                                       maxpow=float(fixd[-1]), random=float(vr),
                                       random_Pg_dBm=float(dB(rgr) + N0_dBm), random_Rg=float(Rgr),
                                       mc_ect=float(mc_opt[-1]), mc_random=float(mc_rnd[-1]))
    ax.plot(eps_ax, opt, "C0" + ls)
    ax.plot(eps_ax, fixd, "C1" + ls, marker="s", mfc="none", ms=3)
    ax.plot(eps_ax, rnd, "C2" + ls, marker="^", mfc="none", ms=3)
    ax.plot(eps_ax, mc_opt, "k+", ms=5)
    ax.plot(eps_ax, mc_rnd, "kx", ms=4)
h = [Line2D([], [], c="C0", label="Proposed (Algorithm 1)"),
     Line2D([], [], c="C1", marker="s", mfc="none", ms=3, label=r"Max. covert power $P_g=\widetilde{P}_g$"),
     Line2D([], [], c="C2", marker="^", mfc="none", ms=3, label=r"Uniform selection below $\tau$ (optimized)"),
     Line2D([], [], c="k", marker="+", ls="none", label="Simulation (proposed)"),
     Line2D([], [], c="k", marker="x", ls="none", ms=4, label="Simulation (uniform selection)"),
     Line2D([], [], c="gray", ls="-", label=r"$P_J^{\max}=20$ dBm"),
     Line2D([], [], c="gray", ls="--", label=r"$P_J^{\max}=10$ dBm")]
ax.legend(handles=h, loc="lower right", fontsize=5.8)
ax.set_xscale("log")
ax.set_ylim(0, 4.2)
ax.set_xlabel(r"Covertness requirement $\epsilon$")
ax.set_ylabel(r"Effective covert throughput $\eta$ (bit/s/Hz)")
fig.savefig(FIG + "fig_ect_eps.pdf"); plt.close(fig)
log["fig5"] = res
# idealised hybrid-SIC scheme [ding2021new] at the operating points (GB-first + GF-first branches)
hyb = {}
for key, (PJv, e) in {"20dBm_eps0.1": (20, 0.1), "20dBm_eps0.01": (20, 0.01), "10dBm_eps0.01": (10, 0.01)}.items():
    r = res[f"{PJv}dBm_eps{e}"]
    h, g = mc_hybrid(snr(Pb), snr(r["Pg_dBm"]), Om_b, Om_g, K, Rb, r["Rg"], n=NMC, rng=rng_fig[5])
    hyb[key] = dict(Pg_dBm=r["Pg_dBm"], Rg=r["Rg"], hybrid=float(h), gb_first=float(g))
cap01 = dB(rho_g_covert_max(0.1, snr(20), Om_w, Om_J)) + N0_dBm
h, g = mc_hybrid(snr(Pb), snr(cap01), Om_b, Om_g, K, Rb, RMAX, n=NMC, rng=rng_fig[5])
hyb["at_cap_eps0.1_Rg4"] = dict(Pg_dBm=float(cap01), Rg=RMAX, hybrid=float(h), gb_first=float(g))
log["hybrid"] = hyb

# ------------------------------------------------------------ Fig. 6: ECT* and P_g* vs P_J^max
PJ_ax2 = np.linspace(-10, 30, 41)
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.grid(False)
for dw, c in [(60.0, "C0"), (100.0, "C3")]:
    for Kk, ls in [(2, "--"), (5, "-")]:
        v_l, p_l = [], []
        for p in PJ_ax2:
            cap = min(rho_g_covert_max(eps, snr(p), Om(dw), Om_J), snr(Pmax))
            v, rg, Rg = optimise(snr(Pb), Om_b, Om_g, Kk, Rb, cap, n_rho=200)
            v_l.append(v); p_l.append(dB(rg) + N0_dBm)
        ax.plot(PJ_ax2, v_l, c + ls, label=rf"$d_{{gw}}={dw:.0f}$ m, $K={Kk}$")
        if Kk == 5:
            ax2.plot(PJ_ax2, p_l, c + ":", lw=0.9)
        log[f"fig6_Pgopt_dw{dw:.0f}_K{Kk}"] = list(map(float, p_l))
        log[f"fig6_ect_dw{dw:.0f}_K{Kk}"] = list(map(float, v_l))
        # unconstrained (reliability-limited) optimum and the jamming threshold P_J^th
        vu, rgu, Rgu = optimise(snr(Pb), Om_b, Om_g, Kk, Rb, snr(Pmax), n_rho=400)
        log[f"fig6_Pgu_dBm_K{Kk}"] = float(dB(rgu) + N0_dBm); log[f"fig6_Rgu_K{Kk}"] = float(Rgu)
        log[f"fig6_ectu_K{Kk}"] = float(vu)
        log[f"fig6_PJth_dBm_dw{dw:.0f}_K{Kk}"] = float(dB(rho_eps(eps) * Om(dw) / Om_J * rgu) + N0_dBm)
ax.plot([], [], "k:", lw=0.9, label=r"$P_g^{*}$ (right axis, $K=5$)")
ax.set_xlabel(r"Maximum jamming power $P_J^{\max}$ (dBm)")
ax.set_ylabel(r"Maximum ECT $\eta^{*}$ (bit/s/Hz)")
ax2.set_ylabel(r"Optimal GF power $P_g^{*}$ (dBm)")
ax.legend(loc="lower right", fontsize=6)
fig.savefig(FIG + "fig_ect_pj.pdf"); plt.close(fig)

# ------------------------------------------------------------ Fig. 7: imperfect jamming cancellation
from residual import gb_outage_residual, gf_outage_residual, residual_slopes
PJ_ax3 = np.arange(-10, 41, 2.5)
PJ_an = np.linspace(-10, 40, 101)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.5))
for zeta, c in [(0.0, "C0"), (1e-4, "C1"), (1e-3, "C2"), (1e-2, "C3")]:
    vals, gbo = [], []
    for p in PJ_ax3:
        cap = min(rho_g_covert_max(eps, snr(p), Om_w, Om_J), snr(Pmax))
        _, rg, Rg = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap, n_rho=200)
        po, pb = mc_sgf(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg, n=NMC, beta=zeta,
                        rho_J=snr(p), Om_JB=Om_JB, rng=rng_fig[7])   # BS unaware of the residual
        vals.append(Rg * (1 - po)); gbo.append(pb)
    # closed forms (Theorem 5); zeta = 0 reduces to Theorem 3 / Proposition 1
    an_e, an_b = [], []
    for p in PJ_an:
        cap = min(rho_g_covert_max(eps, snr(p), Om_w, Om_J), snr(Pmax))
        _, rg, Rg = optimise(snr(Pb), Om_b, Om_g, K, Rb, cap, n_rho=200)
        cc = zeta * snr(p) * Om_JB
        if cc > 0:
            an_e.append(Rg * (1 - gf_outage_residual(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg, cc)))
            an_b.append(gb_outage_residual(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg, cc))
        else:
            an_e.append(Rg * (1 - gf_outage(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg)))
            an_b.append(gb_outage(snr(Pb), Om_b, Rb))
    axa.plot(PJ_an, an_e, c + "-", label=rf"$\zeta={zeta:g}$")
    axb.semilogy(PJ_an, an_b, c + "-", label=rf"$\zeta={zeta:g}$")
    axa.plot(PJ_ax3, vals, c + "o", mfc="none", ms=3)
    axb.semilogy(PJ_ax3, gbo, c + "o", mfc="none", ms=3)
    log[f"fig7_ect_zeta{zeta:g}"] = list(map(float, vals))
    log[f"fig7_gbout_zeta{zeta:g}"] = list(map(float, gbo))
    log[f"fig7_ect_an_zeta{zeta:g}"] = list(map(float, an_e))
    log[f"fig7_gbout_an_zeta{zeta:g}"] = list(map(float, an_b))
axb.axhline(gb_outage(snr(Pb), Om_b, Rb), c="k", ls="--", lw=0.8, label="OMA (Prop. 1)")
axa.plot([], [], "ko", mfc="none", ms=3, label="Simulation")
log["fig7_PJ_dBm"] = list(map(float, PJ_ax3)); log["fig7_PJ_an_dBm"] = list(map(float, PJ_an))
log["gb_outage_OMA_20dBm"] = float(gb_outage(snr(Pb), Om_b, Rb))
# first-order sensitivities at the threshold (Proposition 3)
vu, rgu, Rgu = optimise(snr(Pb), Om_b, Om_g, K, Rb, snr(Pmax), n_rho=400)
Db, Dg, parts = residual_slopes(snr(Pb), rgu, Om_b, Om_g, K, Rb, Rgu)
rJth = rho_eps(eps) * Om_w / Om_J * rgu
log["residual_slopes"] = dict(Db=float(Db), Dg=float(Dg), **parts, PJth_dBm=float(dB(rJth) + N0_dBm))
for zeta in [1e-4, 1e-3, 1e-2]:
    cth = zeta * rJth * Om_JB
    log[f"residual_at_th_zeta{zeta:g}"] = dict(
        c_th=float(cth), ect_loss_lin=float(Rgu * Dg * cth), gb_inc_lin=float(Db * cth),
        ect_exact=float(Rgu * (1 - gf_outage_residual(snr(Pb), rgu, Om_b, Om_g, K, Rb, Rgu, cth))),
        gb_exact=float(gb_outage_residual(snr(Pb), rgu, Om_b, Om_g, K, Rb, Rgu, cth)))
axa.set_xlabel(r"Maximum jamming power $P_J^{\max}$ (dBm)"); axa.set_ylabel(r"ECT $\eta$ (bit/s/Hz)")
axb.set_xlabel(r"Maximum jamming power $P_J^{\max}$ (dBm)"); axb.set_ylabel("GB outage probability")
axa.set_title("(a)", fontsize=8); axb.set_title("(b)", fontsize=8)
axa.legend(loc="lower center"); axb.legend(loc="upper left")
fig.savefig(FIG + "fig_ect_beta.pdf"); plt.close(fig)

# ------------------------------------------------------------ Fig. 8: GB-first vs hybrid SIC (Theorem 3 vs Theorem 4)
fig, ax = plt.subplots()
Pg_ax8 = np.linspace(-5, 35, 161)
Pg_mc8 = np.arange(-5, 36, 5)
hyb = {}
for Kk, c in [(2, "C0"), (5, "C2")]:
    an_g = gf_outage(snr(Pb), snr(Pg_ax8), Om_b, Om_g, Kk, Rb, RMAX)
    an_h = gf_outage_hybrid(snr(Pb), snr(Pg_ax8), Om_b, Om_g, Kk, Rb, RMAX)
    ax.semilogy(Pg_ax8, an_g, c + "-", label=rf"GB-first, $K={Kk}$")
    ax.semilogy(Pg_ax8, an_h, c + "--", label=rf"Hybrid SIC, $K={Kk}$")
    mc_g = [mc_sgf(snr(Pb), snr(p), Om_b, Om_g, Kk, Rb, RMAX, n=NMC, rng=rng_fig[8])[0] for p in Pg_mc8]
    mc_h = [mc_hybrid(snr(Pb), snr(p), Om_b, Om_g, Kk, Rb, RMAX, n=NMC, rng=rng_fig[8])[0] for p in Pg_mc8]
    ax.semilogy(Pg_mc8, mc_g, c + "o", mfc="none", ms=4)
    ax.semilogy(Pg_mc8, mc_h, c + "s", mfc="none", ms=4)
    vg = optimise(snr(Pb), Om_b, Om_g, Kk, Rb, snr(Pmax), R_max=RMAX)
    vh = optimise(snr(Pb), Om_b, Om_g, Kk, Rb, snr(Pmax), R_max=RMAX, outage=gf_outage_hybrid)
    hyb[f"K{Kk}"] = dict(gbfirst=dict(ect=float(vg[0]), Pg_dBm=float(dB(vg[1]) + N0_dBm), Rg=float(vg[2]),
                                      PJth_dBm=float(dB(rho_eps(eps) * Om_w / Om_J * vg[1]) + N0_dBm),
                                      floor=float(gf_outage_floor(Om_b, Om_g, Kk, Rb))),
                         hybrid=dict(ect=float(vh[0]), Pg_dBm=float(dB(vh[1]) + N0_dBm), Rg=float(vh[2]),
                                     PJth_dBm=float(dB(rho_eps(eps) * Om_w / Om_J * vh[1]) + N0_dBm),
                                     floor=float(gf_outage_hybrid_floor(Om_b, Om_g, Kk, Rb, RMAX))),
                         at_gbfirst_opt=dict(pout_gb=float(gf_outage(snr(Pb), vg[1], Om_b, Om_g, Kk, Rb, vg[2])),
                                             pout_hyb=float(gf_outage_hybrid(snr(Pb), vg[1], Om_b, Om_g, Kk, Rb, vg[2]))),
                         mc_gbfirst=[float(v) for v in mc_g], mc_hybrid=[float(v) for v in mc_h])
for e, ls in [(0.05, ":"), (0.1, "-.")]:
    cap = dB(rho_g_covert_max(e, snr(PJ), Om_w, Om_J)) + N0_dBm
    ax.axvline(cap, c="k", ls=ls, lw=0.8)
    ax.text(cap + 0.4, 1.3e-2, rf"$\overline{{P}}_g(\epsilon={e})$", fontsize=6.5, rotation=90)
ax.plot([], [], "ko", mfc="none", ms=4, label="Simulation (GB-first)")
ax.plot([], [], "ks", mfc="none", ms=4, label="Simulation (hybrid)")
ax.set_xlabel(r"GF transmit power $P_g$ (dBm)")
ax.set_ylabel("GF outage probability")
ax.set_ylim(1e-2, 1.1)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=6)
fig.savefig(FIG + "fig_outage_hybrid.pdf"); plt.close(fig)
log["fig8_hybrid"] = hyb
log["fig8_Pg_mc_dBm"] = [float(p) for p in Pg_mc8]

# ------------------------------------------------------------ Fig. 9: robust design under warden-location uncertainty
from robust import varrho_min_collinear, varrho_min_opt, robust_design, dep_worst
D9, d9 = d["w"], d["JW"]                     # nominal GF-warden and jammer-warden distances (collinear geometry)
r_an = np.linspace(2, 90, 177)
r_mc = np.array([10, 20, 30, 40, 50, 60, 70, 80])
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.5))
rob = {}
vu9, rgu9, Rgu9 = optimise(snr(Pb), Om_b, Om_g, K, Rb, snr(Pmax), R_max=RMAX)
for name, vfun, ls in [("fixed", lambda r: varrho_min_collinear(D9, d9, r), "-"),
                       ("opt", lambda r: varrho_min_opt(D9, r), "--")]:
    vr = vfun(r_an)
    PJth = dB(rho_eps(eps) * rgu9 / vr ** alpha) + N0_dBm
    axb.plot(r_an, PJth, "k" + ls, label=("Fixed jammer" if name == "fixed" else "Optimal placement"))
    rob[name] = dict(r=list(map(float, r_an)), PJth_rob_dBm=list(map(float, PJth)), varrho_min=list(map(float, vr)))
    for PJpk, c in [(20.0, "C0"), (30.0, "C1")]:
        ects = [robust_design(snr(Pb), Om_b, Om_g, K, Rb, RMAX, snr(Pmax), snr(PJpk), eps, float(v), alpha)[0] for v in vr]
        axa.plot(r_an, ects, c + ls, label=rf"$P_J^{{\mathrm{{pk}}}}={PJpk:.0f}$ dBm, " + ("fixed" if name == "fixed" else "optimal"))
        mc = []
        for r in r_mc:
            v, rg, Rg, _, _ = robust_design(snr(Pb), Om_b, Om_g, K, Rb, RMAX, snr(Pmax), snr(PJpk), eps, float(vfun(r)), alpha)
            po, _ = mc_sgf(snr(Pb), rg, Om_b, Om_g, K, Rb, Rg, n=NMC, rng=rng_fig[9])
            mc.append(Rg * (1 - po))
        axa.plot(r_mc, mc, c + ("o" if name == "fixed" else "s"), mfc="none", ms=3)
        # largest uncertainty radius without ECT loss: varrho_min(r) = (kappa_eps rho_g^u / rho_J^pk)^(1/alpha)
        vstar = (rho_eps(eps) * rgu9 / snr(PJpk)) ** (1.0 / alpha)
        if name == "fixed":
            r1 = (D9 - vstar * d9) / (vstar - 1.0) if vstar > 1 else np.inf
            rmax = r1 if r1 ** 2 <= d9 * D9 else (D9 + vstar * d9) / (vstar + 1.0)
        else:
            rmax = D9 / vstar
        rob[name][f"PJpk{PJpk:.0f}"] = dict(ect=list(map(float, ects)), mc_r=list(map(float, r_mc)), mc_ect=list(map(float, mc)),
                                            r_max_no_loss=float(rmax))
# worst-case DEP of the nominal design (jammer fixed, P_g = P_g^u, P_J^max = P_J^th) versus r
rJth9 = rho_eps(eps) * Om_w / Om_J * rgu9
vr_f = varrho_min_collinear(D9, d9, r_an)
dep_nom = dep_worst(rgu9, rJth9, vr_f, alpha)
axb2 = axb.twinx()
axb2.plot(r_an, dep_nom, "C3-", label="Nominal design")
axb2.axhline(1 - eps, c="C3", ls=":", lw=0.8)
dep_mc = []
for r in r_mc:
    v = float(varrho_min_collinear(D9, d9, r))
    # worst-case position: distances d_gw', d_Jw' with d_gw'/d_Jw' = v (far side for r^2 <= dD, near side otherwise)
    dgw, dJw = (D9 + r, d9 + r) if r ** 2 <= d9 * D9 else (D9 - r, r - d9)
    dep_mc.append(mc_dep(rgu9, rJth9, Om(dgw), Om(dJw), rho_b=snr(Pb), Om_bw=Om_bw, n_ch=4000, rng=rng_fig[9]))
axb2.plot(r_mc, dep_mc, "C3o", mfc="none", ms=3)
rob["nominal_dep_worst"] = list(map(float, dep_nom)); rob["nominal_dep_worst_mc"] = list(map(float, dep_mc))
rob["nominal"] = dict(Pg_u_dBm=float(dB(rgu9) + N0_dBm), PJth_dBm=float(dB(rJth9) + N0_dBm), ect=float(vu9))
axa.set_xlabel(r"Uncertainty radius $r$ (m)"); axa.set_ylabel(r"Robust ECT $\eta^{*}$ (bit/s/Hz)")
axb.set_xlabel(r"Uncertainty radius $r$ (m)"); axb.set_ylabel(r"$P_J^{\mathrm{th,rob}}$ (dBm)")
axb2.set_ylabel(r"Worst-case $\overline{\xi}^{*}$, nominal design", color="C3"); axb2.tick_params(axis="y", colors="C3")
axb2.set_ylim(0, 1.05)
axb.axhline(20, c="C0", ls=":", lw=0.7); axb.axhline(30, c="C1", ls=":", lw=0.7)
axa.plot([], [], "ko", mfc="none", ms=3, label="Simulation (fixed)"); axa.plot([], [], "ks", mfc="none", ms=3, label="Simulation (optimal)")
axb.plot([], [], "C3-", label=r"Nominal design, worst-case $\overline{\xi}^{*}$"); axb.plot([], [], "C3o", mfc="none", ms=3, label="Simulation")
axa.set_title("(a)", fontsize=8); axb.set_title("(b)", fontsize=8)
axb.set_ylim(-15, 45)
axa.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=5.5)
axb.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=5.5)
fig.savefig(FIG + "fig_robust.pdf"); plt.close(fig)
log["fig9_robust"] = rob

# ------------------------------------------------------------ numbers quoted in the text
log["rho_eps"] = {str(e): float(rho_eps(e)) for e in [0.01, 0.05, 0.1, 0.2]}
log["mean_rx_snr_dB"] = {"GB": float(dB(snr(Pb) * Om_b)), "GF@20dBm": float(dB(snr(20) * Om_g))}
log["dep_crossing_Pg10_dBm"] = float(10 + dB(rho_eps(eps) * Om_w / Om_J))
json.dump(log, open(FIG + "numbers.json", "w"), indent=1)
print(json.dumps({k: v for k, v in log.items() if not isinstance(v, list)}, indent=1))
