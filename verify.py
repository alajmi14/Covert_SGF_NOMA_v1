import numpy as np
from model import *
# self-check of the paper's closed forms against Monte Carlo; exits non-zero on failure
rng=np.random.default_rng(1)
# DEP check
for rg,rJ,Ow,OJ in [(10,100,1,1),(100,100,1,1),(10,1000,0.2,1),(50,20,1,0.5)]:
    print("DEP",rg,rJ, round(dep_avg(rg,rJ,Ow,OJ),4), round(mc_dep(rg,rJ,Ow,OJ,n_ch=3000,n_pj=3000,rng=rng),4))
# conditional DEP via direct numeric average of Theorem1
X=rng.exponential(10,10**6);Y=rng.exponential(100,10**6);print("T1avg",dep_min_conditional(X,Y).mean(),dep_avg(10,100,1,1))
for e in [0.01,0.05,0.1,0.2]:
    r=rho_eps(e); print("rho_eps",e,r, np.log1p(r)/r)
for K in [1,3,5]:
  for rb,rg in [(100,50),(1000,300),(10,10)]:
    a=gf_outage(rb,rg,1,0.7,K,1,1); s,gb=mc_sgf(rb,rg,1,0.7,K,1,1,n=400000,rng=rng)
    print("GF",K,rb,rg,round(a,4),round(s,4),"GB",round(gb_outage(rb,1,1),4),round(gb,4))
print("floor",gf_outage(1e7,1e7,1,0.7,3,1,1),gf_outage_floor(1,0.7,3,1))

# hard checks
rng=np.random.default_rng(7)
for K,rb,rg,R in [(1,100,50,1.0),(5,1000,300,2.0),(3,300,30,3.0)]:
    a=gf_outage(rb,rg,1,0.7,K,1,R); m,_=mc_sgf(rb,rg,1,0.7,K,1,R,n=400000,rng=rng); assert abs(a-m)<3e-3,(a,m)
    a=float(gf_outage_random(rb,rg,1,0.7,K,1,R)); m,_=mc_sgf(rb,rg,1,0.7,K,1,R,n=400000,select="random",rate_aware=False,rng=rng); assert abs(a-m)<3e-3,(a,m)
# hybrid scheme (Theorem 4, Corollary 4): closed form vs simulator, GB-first component reproduces Theorem 3,
# hybrid outage never larger; cases with delta_b delta_g < 1 (floor 0) and >= 1 (floor of Corollary 4)
for K,rb,rg,Rb,R in [(5,1000,300,1,2.0),(3,300,30,1,3.0),(4,2000,100,0.5,0.5),(2,50,400,1,4.0)]:
    h,g=mc_hybrid(rb,rg,1,0.7,K,Rb,R,n=400000,rng=rng); a=gf_outage(rb,rg,1,0.7,K,Rb,R); ah=float(gf_outage_hybrid(rb,rg,1,0.7,K,Rb,R))
    assert abs(a-g)<3e-3 and abs(ah-h)<3e-3 and ah<=a+1e-9,(a,g,ah,h)
for K,Rb,R in [(3,1,2.0),(3,0.5,0.5)]:
    assert abs(float(gf_outage_hybrid(1e9,2e9,1,0.7,K,Rb,R))-gf_outage_hybrid_floor(1,0.7,K,Rb,R,mu=2.0))<1e-6
# residual jamming: an oracle BS (knows the residual) never does worse for the GB user than an unaware BS
po1,pb1=mc_sgf(1000,30,1,0.7,5,1,2.0,n=400000,beta=1e-2,rho_J=1000,Om_JB=0.5,bs_knows_residual=True,rng=rng)
po0,pb0=mc_sgf(1000,30,1,0.7,5,1,2.0,n=400000,beta=1e-2,rho_J=1000,Om_JB=0.5,bs_knows_residual=False,rng=rng)
assert pb1<=pb0+1e-3,(pb1,pb0)
# finite-blocklength radiometer: DEP within a few percent of the n->infinity value for n=1000
d_inf=mc_dep(10,100,1,1,n_ch=2000,n_pj=3000,rng=rng); d_fin=mc_dep(10,100,1,1,n_ch=2000,n_pj=3000,n=1000,rng=rng)
assert abs(d_inf-d_fin)<0.05,(d_inf,d_fin)
# residual jamming (Theorem 4): closed forms vs the simulator with an unaware BS
from residual import gb_outage_residual, gf_outage_residual, residual_slopes
rb_, rg_, K_, Rb_, Rg_, Ob_, Og_, OJB_ = 1000.0, 30.0, 5, 1.0, 2.0, 1.0, 0.7, 0.5
assert abs(gf_outage_residual(rb_, rg_, Ob_, Og_, K_, Rb_, Rg_, 1e-9) - gf_outage(rb_, rg_, Ob_, Og_, K_, Rb_, Rg_)) < 1e-6
assert abs(gb_outage_residual(rb_, rg_, Ob_, Og_, K_, Rb_, Rg_, 1e-9) - gb_outage(rb_, Ob_, Rb_)) < 1e-6
for zeta_, rJ_ in [(1e-2, 100.0), (1e-2, 2000.0)]:
    c_ = zeta_ * rJ_ * OJB_
    po_, pb_ = mc_sgf(rb_, rg_, Ob_, Og_, K_, Rb_, Rg_, n=10**6, beta=zeta_, rho_J=rJ_, Om_JB=OJB_, rng=rng)
    pg_cl, pb_cl = gf_outage_residual(rb_, rg_, Ob_, Og_, K_, Rb_, Rg_, c_), gb_outage_residual(rb_, rg_, Ob_, Og_, K_, Rb_, Rg_, c_)
    assert abs(pg_cl - po_) < 3e-3 and abs(pb_cl - pb_) < 0.1 * pb_ + 3e-4, (c_, pg_cl, po_, pb_cl, pb_)
    Db_, Dg_, _ = residual_slopes(rb_, rg_, Ob_, Og_, K_, Rb_, Rg_)
    print("residual c=%.3f: GF closed %.4f MC %.4f (lin %.4f) | GB closed %.5f MC %.5f (lin %.5f)" % (
        c_, pg_cl, po_, gf_outage(rb_, rg_, Ob_, Og_, K_, Rb_, Rg_) + Dg_ * c_, pb_cl, pb_, gb_outage(rb_, Ob_, Rb_) + Db_ * c_))
# robust design (Propositions 4 and 5): worst case on the boundary, collinear closed form, inverse point optimal
from robust import varrho_boundary, varrho_disc, varrho_min_collinear, varrho_min_opt
for r_ in [20.0, 60.0, 85.0]:
    b_, f_, c_ = varrho_boundary(100.0, (70.0, 0.0), r_), varrho_disc(100.0, (70.0, 0.0), r_), float(varrho_min_collinear(100.0, 30.0, r_))
    assert abs(b_ - f_) < 2e-2 * b_ and abs(b_ - c_) < 1e-3 * b_, (r_, b_, f_, c_)
    inv_ = varrho_boundary(100.0, (100.0 - r_ ** 2 / 100.0, 0.0), r_)
    grid_ = max(varrho_boundary(100.0, (x_, y_), r_, n=361) for x_ in np.linspace(0, 100 + r_, 61) for y_ in np.linspace(0, r_, 21))
    assert abs(inv_ - float(varrho_min_opt(100.0, r_))) < 1e-3 * inv_ and inv_ >= grid_ - 1e-2 * grid_, (r_, inv_, grid_)
print("verify.py: all checks passed")
