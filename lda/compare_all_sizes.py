"""
compare_all_sizes.py - capstone cross-size report over 100 / 1k / 10k for the REAL runs.

Shows, per model:
  1. the real-corpus coherence trend vs corpus size (does topic quality plateau?),
  2. within-corpus seed jitter (OT) vs between-corpus OT for every pair (does a smaller corpus
     reproduce a larger one, beyond its own seed noise?),
  3. pairwise cross-size TVD heatmaps (adjacent sizes).

Reuses comparison_utils. Uses a common seed set across corpora.

Usage (from lda/):
    python compare_all_sizes.py --corpora 100_sorted_pdf,1000_sorted_pdf,10000_sorted_pdf \
        --labels 100,1k,10k --seeds 0-4 --models lda,hdp --out results/report_sizes_all
"""
import argparse, base64, io, itertools, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
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
        return None
    df = pd.read_csv(dist)
    prevp = os.path.join(d, "topic_prevalence.csv")
    pm = pd.read_csv(prevp).set_index("Topic_ID")["Prevalence"].to_dict() if os.path.exists(prevp) else {}
    rj = os.path.join(d, "run.json")
    coh = json.load(open(rj)).get("metrics", {}).get("coherence") if os.path.exists(rj) else None
    topics, prev = [], []
    for tid, g in df.groupby("Topic_ID"):
        topics.append({"topic_name": f"T{tid}",
                       "word_probabilities": {str(r.Word): float(r.Probability) for r in g.itertuples()}})
        prev.append(float(pm.get(tid, 1.0 / df["Topic_ID"].nunique())))
    return {"topics": topics, "prev": prev, "coherence": coh}


def ot(a, b):
    vocab = sorted({w for t in a["topics"] + b["topics"] for w in t["word_probabilities"]})
    return calculate_optimal_transport_distance(a["topics"], b["topics"], a["prev"], b["prev"], vocab)


def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def heatmap(a, b, la, lb, title):
    vocab = sorted({w for t in a["topics"] + b["topics"] for w in t["word_probabilities"]})
    M = np.array([[calculate_tvd(x["word_probabilities"], y["word_probabilities"], vocab)
                   for y in b["topics"]] for x in a["topics"]])
    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(b["topics"]), 1.2 + 0.55 * len(a["topics"])))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1, ax=ax,
                xticklabels=[t["topic_name"] for t in b["topics"]],
                yticklabels=[t["topic_name"] for t in a["topics"]],
                cbar_kws={"label": "TVD"})
    ax.set_xlabel(lb); ax.set_ylabel(la); ax.set_title(title, fontsize=11); fig.tight_layout()
    return b64(fig)


def ms(x):
    x = [v for v in x if v is not None]
    return (float(np.mean(x)), float(np.std(x))) if x else (float("nan"), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="100_sorted_pdf,1000_sorted_pdf,10000_sorted_pdf")
    ap.add_argument("--labels", default="100,1k,10k")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--models", default="lda,hdp")
    ap.add_argument("--out", default="results/report_sizes_all")
    ap.add_argument("--root", default="results/controls",
                    help="results root holding <corpus>/<model>/real/sNNN/ (default: the WM runs)")
    args = ap.parse_args()
    global ROOT
    ROOT = args.root
    os.makedirs(args.out, exist_ok=True)
    corpora = [c.strip() for c in args.corpora.split(",")]
    labels = [l.strip() for l in args.labels.split(",")]
    lab = dict(zip(corpora, labels))
    seeds = parse_seeds(args.seeds)
    models = [m.strip() for m in args.models.split(",")]

    sections = ""
    for model in models:
        data = {c: {s: load(model, c, s) for s in seeds} for c in corpora}
        data = {c: {s: v for s, v in d.items() if v} for c, d in data.items()}

        # coherence trend
        cohs = {c: ms([data[c][s]["coherence"] for s in data[c]]) for c in corpora if data[c]}
        fig, ax = plt.subplots(figsize=(5, 3.2))
        xs = [lab[c] for c in corpora if c in cohs]
        ys = [cohs[c][0] for c in corpora if c in cohs]
        es = [cohs[c][1] for c in corpora if c in cohs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, color="steelblue")
        ax.set_ylabel("real coherence (c_v)"); ax.set_xlabel("corpus size")
        ax.set_title(f"{model.upper()}: coherence vs corpus size"); ax.set_ylim(0, 1); ax.grid(alpha=.3)
        trend_img = b64(fig)

        # within jitter + pairwise cross
        within = {c: ms([ot(data[c][i], data[c][j]) for i, j in itertools.combinations(data[c], 2)])
                  for c in corpora if len(data[c]) >= 2}
        pair_rows, heat_imgs = "", ""
        for a, b in itertools.combinations([c for c in corpora if data[c]], 2):
            cross = ms([ot(data[a][i], data[b][j]) for i in data[a] for j in data[b]])
            covs = []
            for i in data[a]:
                for j in data[b]:
                    ta, tb = data[a][i], data[b][j]
                    vocab = sorted({w for t in ta["topics"] + tb["topics"] for w in t["word_probabilities"]})
                    bm = calculate_best_match_metrics(ta["topics"], tb["topics"], vocab, 0.3)
                    if bm: covs.append(bm["coverage_1to2"])
            jit = max(within.get(a, (0, 0))[0], within.get(b, (0, 0))[0])
            verdict = "≈ seed noise" if cross[0] <= jit * 1.3 else "&gt; seed noise (diverge)"
            pair_rows += (f"<tr><td>{lab[a]} ↔ {lab[b]}</td><td><b>{cross[0]:.3f}±{cross[1]:.3f}</b></td>"
                          f"<td>{ms(covs)[0]*100:.0f}%</td><td>{verdict}</td></tr>")
            s0a, s0b = sorted(data[a])[0], sorted(data[b])[0]
            heat_imgs += (f"<h4>{lab[a]} (rows) vs {lab[b]} (cols)</h4>"
                          f"<img src='data:image/png;base64,{heatmap(data[a][s0a], data[b][s0b], lab[a], lab[b], '')}'>")

        within_row = " · ".join(f"{lab[c]}: {within[c][0]:.3f}±{within[c][1]:.3f}" for c in corpora if c in within)
        sections += f"""<h2>{model.upper()}</h2>
<img src="data:image/png;base64,{trend_img}">
<p><b>Within-corpus seed jitter (OT):</b> {within_row} &nbsp; <span style="color:#666">(how much the model moves across seeds on the SAME corpus - the noise floor)</span></p>
<table><tr><th>Pair</th><th>cross-size OT</th><th>coverage TVD&lt;0.3</th><th>vs seed noise</th></tr>{pair_rows}</table>
<h3>Cross-size TVD heatmaps</h3>{heat_imgs}
"""

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Cross-size topic comparison</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;line-height:1.5;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.5rem 0}} th,td{{border:1px solid #ddd;padding:6px 9px;text-align:left}} th{{background:#f6f6f6}}
img{{max-width:100%;border:1px solid #eee;border-radius:6px;margin:.3rem 0}} h2{{border-bottom:2px solid #eee;margin-top:2rem}}
.note{{background:#eef6ff;border-left:4px solid #5b9bd5;padding:8px 12px;font-size:.9rem;margin:1rem 0}}</style></head><body>
<h1>Cross-size comparison ({' / '.join(labels)})</h1>
<p>Real topic models at three corpus sizes, {len(seeds)} seeds each. OT: 0 = identical, 1 = disjoint.</p>
<div class="note"><b>Read it as:</b> the <b>coherence-vs-size</b> plot shows whether topic quality plateaus.
The <b>cross-size OT</b> is meaningful only next to the <b>within-corpus seed jitter</b> - if cross ≈ jitter,
the sizes agree up to noise; if cross is much larger, topics genuinely shift with scale. Coverage TVD&lt;0.3 is
the pre-registered bar (strict for full word-distributions - read alongside OT, not alone).</div>
{sections}
</body></html>"""
    out = os.path.join(args.out, "report.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"  report -> {out}")


if __name__ == "__main__":
    main()
