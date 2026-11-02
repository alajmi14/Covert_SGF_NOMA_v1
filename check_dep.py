"""Self-check: held-out DEP estimate must match Theorem 2 within MC noise."""
import time, numpy as np
from model import dep_avg, mc_dep
rng = np.random.default_rng(11)
worst = 0.0
for rg, rJ, Ow, OJ in [(10, 1000, 0.2, 1), (10, 100, 1, 1), (100, 100, 1, 1), (50, 20, 1, 0.5)]:
    t = time.time()
    a, m = dep_avg(rg, rJ, Ow, OJ), mc_dep(rg, rJ, Ow, OJ, n_ch=2000, rng=rng)
    worst = max(worst, abs(a - m))
    print(f"analysis {a:.4f}  MC {m:.4f}  diff {m-a:+.4f}  ({time.time()-t:.0f}s)")
assert worst < 0.01, worst
print("OK, worst abs diff", round(worst, 4))
