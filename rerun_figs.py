"""Re-run selected figure blocks of figures.py (by their '# ---- Fig. N' markers) and
merge their numbers into figs/numbers.json.  Usage: python3 rerun_figs.py 6 7"""
import sys, re, json
src = open("figures.py").read()
parts = re.split(r"(?m)^# -{10,} ", src)
head = parts[0]
blocks = {p.split(":")[0].strip(): p for p in parts[1:]}
want = sys.argv[1:]
code = head
for k, p in blocks.items():
    m = re.match(r"Fig\. (\d+)", k)
    if (m and m.group(1) in want) or (not m):
        code += "\n# " + p
old = json.load(open("../figs/numbers.json"))
ns = {}
exec(compile(code, "figures_subset", "exec"), ns)
old.update(ns["log"])
json.dump(old, open("../figs/numbers.json", "w"), indent=1)
print("merged")
