"""
run_size_curve.py - run LDA and HDP on nested slices of the 10k corpus to map how topic quality
and stability change with corpus size. All sizes are cut from the SAME 10k stemmed cache (top-N by
PubMed rank) so preprocessing is identical across the curve. LDA uses a fixed k so coherence is
comparable across sizes. Saves to results/size_curve/<N>/<model>/sNNN/.

Usage (from lda/):
    python run_size_curve.py --sizes 100,500,1000,3000,5000,6720 --seeds 0-2 --k 5
"""
import argparse, json, os, sys, time
sys.path.insert(0, "src")
sys.path.insert(0, ".")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from test_models_all import make_vectorizer, run_lda, run_hdp
from preprocessing import get_all_stopwords

CACHE = "cache/10000_sorted_pdf/stemmed_texts.json"
OUT = "results/size_curve"


def parse_list(spec):
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_seeds(spec):
    out = []
    for p in spec.split(","):
        p = p.strip()
        if "-" in p:
            a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


def run_one(texts, stopwords, model, size, seed, k):
    out_dir = os.path.join(OUT, str(size), model, f"s{seed:03d}")
    os.makedirs(out_dir, exist_ok=True)
    label = f"size={size} {model} seed={seed}"
    vec = make_vectorizer(len(texts), stopwords); vec.fit(texts)
    t0 = time.time()
    if model == "lda":
        m = run_lda(texts, stopwords, label, num_topics=k, out_dir=out_dir, seed=seed, vectorizer=vec)
    else:
        m = run_hdp(texts, stopwords, label, out_dir=out_dir, seed=seed, vectorizer=vec)
    run = {"size": size, "model": model, "seed": seed, "k": k if model == "lda" else None,
           "n_docs": len(texts), "n_topics": m["n_topics"],
           "metrics": {kk: vv for kk, vv in m.items() if kk not in ("topic_words", "overlaps")},
           "fit_sec": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(run, fh, indent=2, default=float)
    print(f"  {model} size={size} seed={seed}: topics={m['n_topics']} coh={m['coherence']:.3f} ({run['fit_sec']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100,500,1000,3000,5000,6720")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--models", default="lda,hdp")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    with open(CACHE) as fh:
        full = json.load(fh)
    print(f"loaded {len(full)} docs from {CACHE}")
    stopwords = get_all_stopwords()
    sizes = [min(s, len(full)) for s in parse_list(args.sizes)]
    seeds = parse_seeds(args.seeds)
    models = [m.strip() for m in args.models.split(",")]

    for size in sizes:
        texts = full[:size]
        print(f"\n=== size {size} ({len(texts)} docs) ===")
        for model in models:
            for seed in seeds:
                run_one(texts, stopwords, model, size, seed, args.k)


if __name__ == "__main__":
    main()
