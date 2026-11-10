"""
compare_glioma_vs_wm.py - put the glioma cross-size pattern next to the working-memory one.

For each domain (working memory, glioma) and each model (LDA, HDP) it reads the seeded real runs
and reports, per corpus size, the same three things the single-domain size report uses:
  1. coherence (c_v) mean and spread across seeds,
  2. within-corpus seed jitter (mean Optimal-Transport distance between two seeds of the SAME size),
  3. adjacent cross-size OT (100 vs 1k, 1k vs 10k) with a plain "reproduces / diverges" read made
     against the jitter floor.

It does NOT compare glioma topics to WM topics directly (different vocabularies, not meaningful).
It compares the SHAPE of each domain's size curve, which is the question: does glioma behave like
working memory as the corpus grows?

Reads run dirs written by controls.py (--control real). Reuses comparison_utils. No em/en dashes.

Usage (from lda/):
    venv/bin/python compare_glioma_vs_wm.py --seeds 0-4 --out results/glioma/report_vs_wm
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from comparison_utils import calculate_optimal_transport_distance  # noqa: E402

# (label, results-root, [ (size-label, corpus-dir-name), ... ])
DOMAINS = [
    ("Working memory", "results/controls",
     [("100", "100_sorted_pdf"), ("1k", "1000_sorted_pdf"), ("10k", "10000_sorted_pdf")]),
    ("Glioma", "results/glioma/controls",
     [("100", "glioma_100"), ("1k", "glioma_1000"), ("10k", "glioma_10000")]),
]


def parse_seeds(spec):
    out = []
    for p in spec.split(","):
        p = p.strip()
        if "-" in p:
            a, b = p.split("-")
            out += list(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


def load(root, corpus, model, seed):
    """Return (topics, prevalence, coherence) for one real run, or None if absent."""
    d = os.path.join(root, corpus, model, "real", f"s{seed:03d}")
    dist = os.path.join(d, "topic_word_distributions.csv")
    if not os.path.exists(dist):
        return None
    df = pd.read_csv(dist)
    prevp = os.path.join(d, "topic_prevalence.csv")
    pm = pd.read_csv(prevp).set_index("Topic_ID")["Prevalence"].to_dict() if os.path.exists(prevp) else {}
    import json
    rj = os.path.join(d, "run.json")
    coh = None
    if os.path.exists(rj):
        coh = (json.load(open(rj)).get("metrics") or {}).get("coherence")
    topics, prev = [], []
    for tid, g in df.groupby("Topic_ID"):
        topics.append({"topic_name": f"T{tid}",
                       "word_probabilities": {str(r.Word): float(r.Probability) for r in g.itertuples()}})
        prev.append(float(pm.get(tid, 1.0 / df["Topic_ID"].nunique())))
    return {"topics": topics, "prev": prev, "coherence": coh}


def ot(a, b):
    vocab = sorted({w for t in a["topics"] + b["topics"] for w in t["word_probabilities"]})
    return calculate_optimal_transport_distance(a["topics"], b["topics"], a["prev"], b["prev"], vocab)


def ms(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), 0.0)


def domain_stats(root, sizes, model, seeds):
    """Per domain+model: coherence per size, within-size jitter per size, adjacent cross-size OT."""
    runs = {sl: {s: load(root, cd, model, s) for s in seeds} for sl, cd in sizes}
    runs = {sl: {s: v for s, v in d.items() if v} for sl, d in runs.items()}

    coh = {sl: ms([runs[sl][s]["coherence"] for s in runs[sl]]) for sl in runs if runs[sl]}
    within = {sl: ms([ot(runs[sl][i], runs[sl][j]) for i, j in itertools.combinations(runs[sl], 2)])
              for sl in runs if len(runs[sl]) >= 2}

    cross = {}
    order = [sl for sl, _ in sizes]
    for a, b in zip(order, order[1:]):
        if runs.get(a) and runs.get(b):
            vals = [ot(runs[a][i], runs[b][j]) for i in runs[a] for j in runs[b]]
            cross[(a, b)] = ms(vals)
    return {"coh": coh, "within": within, "cross": cross,
            "nseeds": {sl: len(runs[sl]) for sl in runs}}


def verdict(cross_mean, jitter_mean):
    if np.isnan(cross_mean) or np.isnan(jitter_mean):
        return "n/a"
    return "reproduces (cross ~ seed noise)" if cross_mean <= jitter_mean * 1.5 else "diverges (cross > seed noise)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--models", default="lda,hdp")
    ap.add_argument("--out", default="results/glioma/report_vs_wm")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    models = [m.strip() for m in args.models.split(",")]
    os.makedirs(args.out, exist_ok=True)

    order = ["100", "1k", "10k"]
    sections = []
    text_lines = []
    for model in models:
        text_lines.append(f"\n==== {model.upper()} ====")
        rows_coh, rows_cross = "", ""
        stats = {}
        for dlabel, root, sizes in DOMAINS:
            stats[dlabel] = domain_stats(root, sizes, model, seeds)

        # coherence + jitter table (one row per domain, columns per size)
        for dlabel in [d[0] for d in DOMAINS]:
            st = stats[dlabel]
            cells = ""
            line = f"  {dlabel:<15s} coherence:"
            for sl in order:
                c = st["coh"].get(sl)
                w = st["within"].get(sl)
                nseed = st["nseeds"].get(sl, 0)
                ctxt = f"{c[0]:.3f}&plusmn;{c[1]:.3f}" if c else "-"
                wtxt = f"{w[0]:.3f}" if w else "-"
                cells += f"<td>{ctxt}<br><span style='color:#888;font-size:.8em'>jitter {wtxt} &middot; n={nseed}</span></td>"
                line += f"  {sl}={ctxt.replace('&plusmn;','+/-') if c else '-'}(jit {wtxt})"
            rows_coh += f"<tr><td><b>{dlabel}</b></td>{cells}</tr>"
            text_lines.append(line)

        # cross-size OT table + verdict
        for dlabel in [d[0] for d in DOMAINS]:
            st = stats[dlabel]
            cells = ""
            line = f"  {dlabel:<15s} cross-size OT:"
            for a, b in [("100", "1k"), ("1k", "10k")]:
                cr = st["cross"].get((a, b))
                jit = max(st["within"].get(a, (float('nan'), 0))[0], st["within"].get(b, (float('nan'), 0))[0])
                if cr:
                    v = verdict(cr[0], jit)
                    cells += f"<td>{cr[0]:.3f}&plusmn;{cr[1]:.3f}<br><span style='color:#888;font-size:.8em'>{v}</span></td>"
                    line += f"  {a}-{b}={cr[0]:.3f} [{v}]"
                else:
                    cells += "<td>-</td>"
                    line += f"  {a}-{b}=-"
            rows_cross += f"<tr><td><b>{dlabel}</b></td>{cells}</tr>"
            text_lines.append(line)

        sections.append(f"""<h2>{model.upper()}</h2>
<h3>Coherence (c_v) and within-size seed jitter</h3>
<table><tr><th>Domain</th><th>100</th><th>1k</th><th>10k</th></tr>{rows_coh}</table>
<h3>Adjacent cross-size Optimal-Transport distance (0 = identical, 1 = disjoint)</h3>
<table><tr><th>Domain</th><th>100 vs 1k</th><th>1k vs 10k</th></tr>{rows_cross}</table>""")

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Glioma vs Working Memory pattern</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;line-height:1.5;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.4rem 0 1.2rem}} th,td{{border:1px solid #ddd;padding:7px 9px;text-align:left;vertical-align:top}} th{{background:#f6f6f6}}
h2{{border-bottom:2px solid #eee;margin-top:2rem}} .note{{background:#eef6ff;border-left:4px solid #5b9bd5;padding:8px 12px;font-size:.9rem;margin:1rem 0}}</style></head><body>
<h1>Glioma vs Working Memory: same pipeline, side by side</h1>
<p>Both domains run through the identical LDA (k=5) and HDP pipeline, same stopwords and stemming,
on nested 100 / 1k / 10k corpora. Numbers are means over {len(seeds)} seeds (10k may use fewer).</p>
<div class="note"><b>How to read it:</b> coherence shows topic quality at each size; the jitter under
it is how much the model moves across seeds on the SAME corpus (the noise floor). Cross-size OT is
meaningful only against that floor: if the two sizes are about as far apart as one size is from
itself across seeds, the small corpus reproduces the large one; if cross-size OT is well above the
jitter, the topics genuinely shift with scale. This report compares the SHAPE of the two domains'
curves, not glioma topics against working-memory topics.</div>
{''.join(sections)}
</body></html>"""
    out = os.path.join(args.out, "report.html")
    with open(out, "w") as fh:
        fh.write(html)
    print("\n".join(text_lines))
    print(f"\n  report -> {out}")


if __name__ == "__main__":
    main()
