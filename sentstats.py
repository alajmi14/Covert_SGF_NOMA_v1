"""Sentence-length and invariant report for a LaTeX manuscript.

Usage:  python3 sentstats.py main.tex [--list N] [--dump file.txt]
        python3 sentstats.py before.tex after.tex        (adds an invariant check)

Prose = abstract + body + appendices + captions + theorem/proof text, up to the bibliography.
Excluded: preamble/title block, display math, tikz, algorithmic, tabular, comments.
Inline math, a \cite and a \ref/\eqref each count as ONE word.  Sentences break at . ? ! followed by
space and an upper-case letter or a command, at \item, at section headings and at paragraph breaks;
common abbreviations (e.g., i.e., et al., Fig., Eq., Sec., Prop., Cor., Th., cf., vs.) do not break.

Invariant check (two files): the multisets of inline-math strings, display-math blocks, \cite/\ref/
\eqref/\label commands and numerals in prose must be identical; content words added or removed are
listed for manual review (function words are ignored).
"""
import re, sys, statistics as st
from collections import Counter

ABBR = ["e.g.", "i.e.", "et al.", "Fig.", "Figs.", "Eq.", "Eqs.", "Sec.", "Prop.", "Props.", "Cor.",
        "Th.", "Thm.", "cf.", "vs.", "resp.", "approx.", "Ref.", "Refs.", "No."]
DISPLAY = ["equation", "equation*", "align", "align*", "subequations", "tikzpicture", "algorithmic",
           "tabular", "thebibliography"]
STOP = set("""a an the of to in on at by for with from and or but nor so as is are was were be been being
this that these those it its which who whom whose then than thus hence therefore also only both either
not no yes we our us they their there here when where while whereas if because since into onto over under
such each every all any one two three first second third same other another more most less least very
can cannot may might must should would will shall does do did has have had via per within without between
among about above below up down out off again further once just still yet so""".split())


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def body(s):
    a = s.index(r"\begin{abstract}")
    b = s.index(r"\bibliographystyle") if r"\bibliographystyle" in s else len(s)
    return s[a:b]


def captions(s):
    out, i = [], 0
    while True:
        i = s.find(r"\caption{", i)
        if i < 0: return out
        j, d = i + 9, 1
        while d:
            d += {"{": 1, "}": -1}.get(s[j], 0); j += 1
        out.append(s[i + 9:j - 1]); i = j


def display_blocks(s):
    blocks = []
    for env in DISPLAY:
        e = re.escape(env)
        blocks += re.findall(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}", s, re.S)
    blocks += re.findall(r"\\\[.*?\\\]", s, re.S)
    return blocks


def prose(s):
    s = strip_comments(body(s))
    caps = captions(s)
    s = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "\n\n", s, flags=re.S)
    s = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", "\n\n", s, flags=re.S)
    for env in DISPLAY:
        e = re.escape(env)
        s = re.sub(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}", " DISPLAYMATH ", s, flags=re.S)
    s = s + "\n\n" + "\n\n".join(caps)
    return s


def normalise(p):
    p = p.replace(r"\$", "USD")                                            # escaped dollar is text, not math
    p = re.sub(r"\$[^$]*\$", " MATH ", p)                                   # inline math -> one word
    p = re.sub(r"~?\\cite(\[[^\]]*\])?\{[^}]*\}", " CITE ", p)
    p = re.sub(r"\\(eq)?ref\{[^}]*\}", "REF", p)
    p = re.sub(r"\\(section|subsection|appendices|section\*)\*?(\{[^}]*\})?", "\n\n", p)
    p = re.sub(r"\\item", "\n\n", p)
    p = re.sub(r"\\begin\{(theorem|proposition|corollary|lemma|remark|IEEEproof|itemize|enumerate|abstract|IEEEkeywords)\}(\[[^\]]*\])?", "\n\n", p)
    p = re.sub(r"\\end\{[a-zA-Z*]+\}", "\n\n", p)
    p = re.sub(r"\\label\{[^}]*\}", " ", p)
    p = re.sub(r"\\(emph|textit|textbf|mathrm|textsc)\{([^}]*)\}", r"\2", p)
    p = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w+)\}", r"\1\2", p)
    p = re.sub(r"\\hfill\s*MATH", " ", p)
    p = re.sub(r"\\[a-zA-Z]+\*?", " ", p)
    p = p.replace("~", " ").replace("{", " ").replace("}", " ").replace("``", '"').replace("''", '"')
    return p


def sentences(s):
    p = normalise(prose(s))
    out = []
    for para in re.split(r"\n\s*\n", p):
        t = " ".join(para.split())
        if not t: continue
        for k, a in enumerate(ABBR):
            t = t.replace(a, a.replace(".", "<D>"))
        for sent in re.split(r"(?<=[.?!])\s+(?=[A-Z(\"]|MATH|CITE|REF)", t):
            sent = sent.replace("<D>", ".").strip()
            w = [x for x in sent.split() if re.search(r"[A-Za-z0-9]", x)]
            if len(w) >= 3 and not sent.startswith("DISPLAYMATH"):
                out.append((len(w), sent))
    return out


def report(path, n_list=0, dump=None):
    s = open(path).read()
    S = sentences(s)
    L = [n for n, _ in S]
    q = st.quantiles(L, n=10)
    print(f"{path}: {len(L)} sentences | median {st.median(L):.0f} | mean {st.mean(L):.1f} | p90 {q[-1]:.0f} | "
          f"max {max(L)} | >35 words: {sum(n > 35 for n in L)} | >45: {sum(n > 45 for n in L)} | "
          f"em-dashes (--- or \\textemdash): {s.count('---') + s.count(chr(92) + 'textemdash')}")
    if n_list:
        for n, x in sorted(S, reverse=True)[:n_list]:
            print(f"  [{n}] {x[:160]}")
    if dump:
        open(dump, "w").write("\n".join(f"{n}\t{x}" for n, x in S))
    return S


def invariants(a, b):
    A, B = open(a).read(), open(b).read()
    ba, bb = strip_comments(body(A)), strip_comments(body(B))
    checks = {
        "inline math": (Counter(re.findall(r"\$[^$]*\$", ba)), Counter(re.findall(r"\$[^$]*\$", bb))),
        "display math/tikz/tabular": (Counter(display_blocks(ba)), Counter(display_blocks(bb))),
        "cite/ref/eqref/label": (Counter(re.findall(r"\\(?:cite|eqref|ref|label)(?:\[[^\]]*\])?\{[^}]*\}", ba)),
                                 Counter(re.findall(r"\\(?:cite|eqref|ref|label)(?:\[[^\]]*\])?\{[^}]*\}", bb))),
    }
    num = lambda s: Counter(re.findall(r"(?<![A-Za-z_\\])\d+(?:\.\d+)?", re.sub(r"\$[^$]*\$", " ", normalise(prose(s)))))
    checks["numerals in prose"] = (num(A), num(B))
    ok = True
    for k, (x, y) in checks.items():
        same = x == y
        ok &= same
        print(f"  {k:28s}: {'identical' if same else 'DIFFERENT'} ({sum(x.values())} vs {sum(y.values())})")
        if not same:
            print("     only before:", list((x - y).elements())[:10]); print("     only after :", list((y - x).elements())[:10])
    words = lambda s: Counter(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]+", normalise(prose(s)))
                              if w.lower() not in STOP and w not in ("MATH", "CITE", "REF", "DISPLAYMATH"))
    wa, wb = words(A), words(B)
    print("  content words removed:", dict((wa - wb).most_common(40)))
    print("  content words added  :", dict((wb - wa).most_common(40)))
    return ok


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    n_list = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 0
    dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
    if "--list" in sys.argv: args = [a for a in args if a != str(n_list)]
    if dump: args = [a for a in args if a != dump]
    for a in args:
        report(a, n_list, dump)
    if len(args) == 2:
        print("Invariants (before -> after):")
        print("ALL INVARIANTS HOLD" if invariants(*args) else "INVARIANT VIOLATION")
