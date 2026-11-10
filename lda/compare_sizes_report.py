"""
compare_sizes_report.py - cross-corpus comparison (e.g. 100 vs 1k) for the REAL runs.

The research question (prediction #1): does a small corpus reproduce the topics of a larger
one? A single cross-size distance is meaningless on its own, so we compare it to the
*seed-to-seed jitter* of each corpus with itself:

  - within-A : OT between two different seeds of corpus A  (how much A moves on its own)
  - within-B : same for corpus B
  - cross    : OT between a seed of A and a seed of B

If cross ~ within jitter, the corpora are basically the same up to noise (small reproduces
large). If cross >> jitter, they genuinely diverge. We also report coverage at TVD<0.3 (the
pre-registered threshold) and the mean best-match TVD. Reuses comparison_utils.

Usage (from lda/):
    python compare_sizes_report.py --a 100_sorted_pdf --b 1000_sorted_pdf \
        --seeds 0-9 --models lda,hdp --out results/report_compare_100_1k
"""
import argparse, base64, io, itertools, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "src")
from comparison_utils import (calculate_optimal_transport_distance, calculate_best_match_metrics,
                              calculate_tvd)

ROOT = "results/controls"


def parse_seeds(spec):
    out = []
    for p in spec.split(","):
        p = p.strip()
        if "-" in p:
            a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


def load(model, corpus, seed):
    d = os.path.join(ROOT, corpus, model, "real", f"s{seed:03d}")
    dist = os.path.join(d, "topic_word_distributions.csv")
    if not os.path.exists(dist):
        return None, None
    df = pd.read_csv(dist)
    prevp = os.path.join(d, "topic_prevalence.csv")
    prev_map = pd.read_csv(prevp).set_index("Topic_ID")["Prevalence"].to_dict() if os.path.exists(prevp) else {}
    topics, prev = [], []
    for tid, g in df.groupby("Topic_ID"):
        topics.append({"topic_name": f"T{tid}",
                       "word_probabilities": {str(r.Word): float(r.Probability) for r in g.itertuples()}})
        prev.append(float(prev_map.get(tid, 1.0 / df["Topic_ID"].nunique())))
    return topics, prev


def ot_between(ta, pa, tb, pb):
    """OT divergence between two topic sets (signature: topics1, topics2, prev1, prev2, vocab)."""
    vocab = sorted({w for t in ta + tb for w in t["word_probabilities"]})
    return calculate_optimal_transport_distance(ta, tb, pa, pb, vocab)


def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def cross_heatmap(ta, tb, a_name, b_name, title):
    vocab = sorted({w for t in ta + tb for w in t["word_probabilities"]})
    M = np.array([[calculate_tvd(x["word_probabilities"], y["word_probabilities"], vocab)
                   for y in tb] for x in ta])
    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(tb), 1.2 + 0.55 * len(ta)))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1, ax=ax,
                xticklabels=[t["topic_name"] for t in tb], yticklabels=[t["topic_name"] for t in ta],
                cbar_kws={"label": "TVD (0=identical, 1=disjoint)"})
    ax.set_xlabel(b_name); ax.set_ylabel(a_name); ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return b64(fig)


def model_block(model, corpusA, corpusB, seeds):
    A = {s: load(model, corpusA, s) for s in seeds}
    B = {s: load(model, corpusB, s) for s in seeds}
    A = {s: v for s, v in A.items() if v[0]}
    B = {s: v for s, v in B.items() if v[0]}
    if not A or not B:
        return None

    within_a = [ot_between(*A[i], *A[j]) for i, j in itertools.combinations(A, 2)]
    within_b = [ot_between(*B[i], *B[j]) for i, j in itertools.combinations(B, 2)]
    cross = [ot_between(*A[i], *B[j]) for i in A for j in B]
    within_a = [x for x in within_a if x is not None]
    within_b = [x for x in within_b if x is not None]
    cross = [x for x in cross if x is not None]

    # coverage + best-match TVD, cross pairs (A-topics -> nearest B-topic)
    covs, bests = [], []
    for i in A:
        for j in B:
            ta, _ = A[i]; tb, _ = B[j]
            vocab = sorted({w for t in ta + tb for w in t["word_probabilities"]})
            bm = calculate_best_match_metrics(ta, tb, vocab, tvd_threshold=0.3)
            if bm:
                covs.append(bm["coverage_1to2"])
                bests.append(np.mean([m["tvd"] for m in bm["best_matches_1to2"]]))

    s0a, s0b = sorted(A)[0], sorted(B)[0]
    hm = cross_heatmap(A[s0a][0], B[s0b][0], f"{corpusA}", f"{corpusB}",
                       f"{model.upper()}: {corpusA} (rows) vs {corpusB} (cols) - best seed pair, TVD")

    def ms(x):
        return (np.mean(x), np.std(x)) if x else (float("nan"), 0.0)
    return {
        "model": model, "nA": len(A), "nB": len(B),
        "within_a": ms(within_a), "within_b": ms(within_b), "cross": ms(cross),
        "coverage": ms(covs), "best_tvd": ms(bests), "heatmap": hm,
        "ntopicsA": len(A[s0a][0]), "ntopicsB": len(B[s0b][0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="100_sorted_pdf")
    ap.add_argument("--b", default="1000_sorted_pdf")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--models", default="lda,hdp")
    ap.add_argument("--out", default="results/report_compare")
    ap.add_argument("--root", default="results/controls",
                    help="results root holding <corpus>/<model>/real/sNNN/ (default: the WM runs)")
    args = ap.parse_args()
    global ROOT
    ROOT = args.root
    os.makedirs(args.out, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    blocks = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        print(f"  comparing {m} ...")
        b = model_block(m, args.a, args.b, seeds)
        if b:
            blocks.append(b)

    def fmt(t):
        return f"{t[0]:.3f}±{t[1]:.3f}"

    rows = ""
    cards = ""
    for b in blocks:
        ca, cb, cx = b["within_a"][0], b["within_b"][0], b["cross"][0]
        jitter = max(ca, cb)
        verdict = ("cross ≈ seed noise → small reproduces large" if cx <= jitter * 1.5
                   else "cross &gt; seed noise → genuinely diverge")
        rows += (f"<tr><td><b>{b['model'].upper()}</b></td>"
                 f"<td>{fmt(b['within_a'])}</td><td>{fmt(b['within_b'])}</td>"
                 f"<td><b>{fmt(b['cross'])}</b></td>"
                 f"<td>{b['coverage'][0]*100:.0f}%</td><td>{fmt(b['best_tvd'])}</td>"
                 f"<td>{verdict}</td></tr>")
        cards += (f"<h3>{b['model'].upper()} - {args.a} ({b['ntopicsA']} topics) vs {args.b} ({b['ntopicsB']} topics)</h3>"
                  f"<img src='data:image/png;base64,{b['heatmap']}'>")

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Cross-size comparison {args.a} vs {args.b}</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;line-height:1.5;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{border:1px solid #ddd;padding:6px 9px;text-align:left}} th{{background:#f6f6f6}}
img{{max-width:100%;border:1px solid #eee;border-radius:6px;margin:.4rem 0}} .note{{background:#eef6ff;border-left:4px solid #5b9bd5;padding:8px 12px;font-size:.9rem;margin:1rem 0}}</style></head><body>
<h1>Cross-size comparison: {args.a} vs {args.b}</h1>
<p>Optimal-Transport divergence between the REAL topic models of the two corpus sizes (10 seeds each),
compared to each corpus's own seed-to-seed jitter. OT: 0 = identical, 1 = disjoint.</p>
<table>
<tr><th>Model</th><th>within-{args.a} (jitter)</th><th>within-{args.b} (jitter)</th>
<th>cross {args.a}↔{args.b}</th><th>coverage TVD&lt;0.3</th><th>mean best-match TVD</th><th>verdict</th></tr>
{rows}
</table>
<div class="note"><b>How to read it:</b> compare the <b>cross</b> column to the two <b>jitter</b> columns.
If cross is about the same as a corpus's own seed-to-seed wobble, the small corpus reproduces the large one.
If cross is much larger, the topics genuinely shift with scale. "coverage TVD&lt;0.3" is the fraction of
small-corpus topics whose nearest large-corpus topic is within the pre-registered 0.3 threshold (note: full
word-distribution TVD rarely drops below ~0.4 even for related topics, so a low % here does not by itself mean
"unrelated" - read it alongside the OT-vs-jitter comparison).</div>
<h2>Cross-size topic heatmaps</h2>
<p>Each cell = TVD between a {args.a} topic (row) and a {args.b} topic (column). Dark/low cells mark a
corresponding theme found at both sizes.</p>
{cards}
</body></html>"""
    out = os.path.join(args.out, "report.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"  report -> {out}")
    for b in blocks:
        print(f"  {b['model'].upper()}: within-A={fmt(b['within_a'])} within-B={fmt(b['within_b'])} "
              f"cross={fmt(b['cross'])} coverage<0.3={b['coverage'][0]*100:.0f}% bestTVD={fmt(b['best_tvd'])}")


if __name__ == "__main__":
    main()
