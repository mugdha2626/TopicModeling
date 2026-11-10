"""
make_report.py - build a self-contained HTML results report for one corpus: per-topic word
clouds, a topic x topic TVD heatmap per model, a cross-model (LDA vs HDP) heatmap, topic word
tables, and the headline metrics. Reuses calculate_tvd from comparison_utils and the stem->word
display map from preprocessing (so labels read as real words, not Snowball stems).

Usage (from lda/):
    python make_report.py --lda results/lda/100_sorted_pdf \
        --hdp results/controls/100_sorted_pdf/hdp/real/s042 \
        --title "100 corpus (74 papers)" --out results/report_100
"""
import argparse, base64, io, json, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

sys.path.insert(0, "src")
from preprocessing import unstem_words


def load_topics(d):
    dist = pd.read_csv(os.path.join(d, "topic_word_distributions.csv"))
    prev_path = os.path.join(d, "topic_prevalence.csv")
    prev = pd.read_csv(prev_path).set_index("Topic_ID")["Prevalence"].to_dict() if os.path.exists(prev_path) else {}
    topics = []
    for tid, g in dist.groupby("Topic_ID"):
        wp = {str(r.Word): float(r.Probability) for r in g.itertuples()}
        topics.append({"id": int(tid), "wp": wp, "prevalence": float(prev.get(tid, 1.0 / dist["Topic_ID"].nunique()))})
    topics.sort(key=lambda t: -t["prevalence"])
    return topics


def disp(words):
    """Stem -> readable word for display."""
    return unstem_words(list(words))


def read_coh(d):
    """Coherence (c_v) from a run.json in dir d, else None."""
    p = os.path.join(d, "run.json")
    if os.path.exists(p):
        try:
            return json.load(open(p)).get("metrics", {}).get("coherence")
        except Exception:
            return None
    return None


def coh_str(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def wordcloud_grid(topics, title):
    n = len(topics)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for idx, t in enumerate(topics):
        ax = axes[idx // cols][idx % cols]
        freqs = {disp([w])[0]: p for w, p in sorted(t["wp"].items(), key=lambda x: -x[1])[:30]}
        wc = WordCloud(width=400, height=300, background_color="white",
                       colormap="viridis", prefer_horizontal=0.95).generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        top3 = ", ".join(disp([w for w, _ in sorted(t["wp"].items(), key=lambda x: -x[1])[:3]]))
        ax.set_title(f"T{t['id']} ({t['prevalence']:.0%})  {top3}", fontsize=9)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return b64(fig)


def topic_table(topics):
    rows = []
    for t in topics:
        top = sorted(t["wp"].items(), key=lambda x: -x[1])[:10]
        words = ", ".join(f"{dw}" for dw in disp([w for w, _ in top]))
        rows.append(f"<tr><td>T{t['id']}</td><td>{t['prevalence']:.1%}</td><td>{words}</td></tr>")
    return ("<table><tr><th>Topic</th><th>Prevalence</th><th>Top words</th></tr>"
            + "".join(rows) + "</table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lda", required=True)
    ap.add_argument("--hdp", required=True)
    ap.add_argument("--title", default="Topic model results")
    ap.add_argument("--out", default="results/report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lda = load_topics(args.lda)
    hdp = load_topics(args.hdp)
    lda_coh = coh_str(read_coh(args.lda))
    hdp_coh = coh_str(read_coh(args.hdp))

    print("  rendering word clouds...")
    wc_lda = wordcloud_grid(lda, f"LDA topics ({len(lda)})")
    wc_hdp = wordcloud_grid(hdp, f"HDP topics ({len(hdp)})")

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{args.title}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1a1a1a;line-height:1.5}}
 h1{{font-size:1.5rem}} h2{{margin-top:2rem;border-bottom:2px solid #eee;padding-bottom:4px}}
 table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.5rem 0}}
 th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}} th{{background:#f6f6f6}}
 img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:6px}}
 .note{{background:#fff8e6;border-left:4px solid #f0c000;padding:8px 12px;font-size:.9rem;margin:1rem 0}}
 .metrics{{display:flex;gap:16px;flex-wrap:wrap}} .metric{{background:#f6f6f6;border-radius:8px;padding:10px 16px}}
 .metric b{{display:block;font-size:1.3rem}}
</style></head><body>
<h1>{args.title}</h1>
<p>Topic-model results for LDA and HDP. Word labels are de-stemmed for readability (e.g. <i>memori&rarr;memory</i>);
the models operate on Snowball stems.</p>

<div class="metrics">
 <div class="metric"><b>{len(lda)}</b>LDA topics</div>
 <div class="metric"><b>{lda_coh}</b>LDA coherence (c_v)*</div>
 <div class="metric"><b>{len(hdp)}</b>HDP topics</div>
 <div class="metric"><b>{hdp_coh}</b>HDP coherence (c_v)*</div>
</div>
<div class="note">HDP uses the fixed multi-pass training (it previously collapsed to one topic and failed the
known-topic positive control; it now recovers planted topics at c_v&nbsp;0.73). *Coherence shown is the
single real-corpus run; the control sweep reports mean&plusmn;SD over seeds.</div>

<h2>LDA topics</h2>
{topic_table(lda)}
<img src="data:image/png;base64,{wc_lda}" alt="LDA word clouds">

<h2>HDP topics</h2>
{topic_table(hdp)}
<img src="data:image/png;base64,{wc_hdp}" alt="HDP word clouds">

<p style="color:#888;font-size:.85rem;margin-top:2rem">TVD / Optimal-Transport comparisons are cross-corpus
(100 vs 1k vs 10k) and live in the size-comparison report, not here - within one corpus there is nothing to
compare against.</p>
</body></html>"""

    out_html = os.path.join(args.out, "report.html")
    with open(out_html, "w") as fh:
        fh.write(html)
    print(f"  report -> {out_html}")


if __name__ == "__main__":
    main()
