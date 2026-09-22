"""
controls.py — seeded run harness for LDA / HDP (control runs, Task 2 of the
synthetic-data / null-model work).

SCOPE, DELIBERATELY NARROW: only --control real is implemented. It runs the corpus
exactly as it stands, varying only the random seed. This exists to:

  (a) let LDA and HDP be re-run with any seed instead of the hardcoded 42
      (test_models_all.py's run_lda / run_lda_auto / run_hdp now all take seed=),
  (b) give a stable, consistent output layout for a run of a (corpus, model, control,
      seed) combination, so whoever writes the scrambling functions next (word
      shuffle across the corpus, count resampling, etc.) has somewhere to plug in.

No assumptions about that future code are baked in here. apply_control() below is a
single, clearly marked extension point — see its docstring for the (intentionally
loose) contract a scrambler needs to satisfy. Nothing else in this file should need
to change when that code lands.

Usage (from lda/):
    python controls.py --corpus 100 --model lda --seeds 42 --lda-k 5
    python controls.py --corpus 100 --model hdp --seeds 42
    python controls.py --corpus 1k  --model both --seeds 0-4 --lda-k 5
    python controls.py --corpus 100 --model lda --seeds 42 --dry-run

Output: results/controls/<corpus>/<model>/<control>/s<NNN>/
    run.json, topic_word_distributions.csv, topic_prevalence.csv
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from test_models_all import corpus_name, load_corpus, run_lda_auto, run_hdp  # noqa: E402

CORPUS_ZIPS = {
    "100": "/Users/mugdha/Desktop/research/100_sorted_pdf.zip",
    "1k": "/Users/mugdha/Desktop/research/1000_sorted_pdf.zip",
    "10k": "/Users/mugdha/Desktop/research/10000_sorted_pdf.zip",
}


def resolve_corpus(name):
    """Accept a shorthand (100 / 1k / 10k) or a direct zip path."""
    if name in CORPUS_ZIPS:
        return CORPUS_ZIPS[name]
    if os.path.exists(name):
        return name
    raise ValueError(f"unknown corpus {name!r}; use 100, 1k, 10k, or a zip path")


def apply_control(control, texts, *, corpus, seed):
    """
    Transform the stemmed document list (one whitespace-joined string per paper,
    exactly what load_corpus() returns) according to `control`.

    Only "real" is implemented right now: it returns `texts` unchanged, so a run
    differs from the existing results/{lda,hdp}/<corpus> output only by its random
    seed (and not at all, at seed=42 with the same fixed_k -- see the reproduction
    check in this repo's notes for how that was verified).

    EXTENSION POINT. When the scrambling functions exist (shuffle words across the
    corpus, resample each paper's words from the corpus-wide distribution, replace
    words with random real words, etc.), add their dispatch here by name. The only
    informal contract a scrambler needs to satisfy to plug in: given the `texts`
    list and a seed, return a same-length list of strings. Nothing else in this file
    needs to change -- `run_one` below already passes the transformed `texts`
    straight into run_lda_auto / run_hdp.
    """
    if control == "real":
        return texts
    raise NotImplementedError(
        f"control={control!r} is not built yet -- only 'real' works right now. "
        "The scrambling functions haven't been written; when they are, wire them "
        "in here (see this function's docstring for the contract)."
    )


def run_one(corpus_arg, model, control, seed, *, out_root="results/controls",
            cache_root="cache", lda_k=None, log=print):
    zip_path = resolve_corpus(corpus_arg)
    cname = corpus_name(zip_path)
    label = f"{cname} · {control} · seed {seed}"

    t0 = time.time()
    real_texts, stopwords = load_corpus(zip_path, cache_root=cache_root)
    texts = apply_control(control, real_texts, corpus=cname, seed=seed)
    load_s = time.time() - t0

    out_dir = os.path.join(out_root, cname, model, control, f"s{seed:03d}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    if model == "lda":
        metrics = run_lda_auto(texts, stopwords, label, out_dir=out_dir, fixed_k=lda_k, seed=seed)
    elif model == "hdp":
        metrics = run_hdp(texts, stopwords, label, out_dir=out_dir, seed=seed)
    else:
        raise ValueError(f"unknown model {model!r}; use lda or hdp")
    fit_s = time.time() - t0

    run = {
        "schema": 1,
        "control": control, "corpus": cname, "model": model, "seed": seed,
        "n_docs": len(texts), "n_topics": metrics["n_topics"],
        "metrics": {k: v for k, v in metrics.items() if k not in ("topic_words", "overlaps")},
        "params": {"lda_k": lda_k} if model == "lda" else {},
        "timings_sec": {"load": round(load_s, 2), "fit": round(fit_s, 2)},
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(run, fh, indent=2, default=float)
    log(f"  -> {out_dir}")
    return run


def parse_seeds(spec):
    """'42' -> [42]; '0-29' -> [0..29]; '0,5,10' -> [0, 5, 10]"""
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="100 | 1k | 10k | a zip path")
    ap.add_argument("--model", required=True, choices=["lda", "hdp", "both"])
    ap.add_argument("--control", default="real", help="only 'real' works right now")
    ap.add_argument("--seeds", default="42", help="'42', '0-29', or '0,5,10'")
    ap.add_argument("--lda-k", type=int, default=None, help="fixed LDA k (omit for the auto-k coherence sweep)")
    ap.add_argument("--out-root", default="results/controls")
    ap.add_argument("--cache-root", default="cache")
    ap.add_argument("--dry-run", action="store_true", help="print the run plan without fitting anything")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    models = ["lda", "hdp"] if args.model == "both" else [args.model]
    cname = corpus_name(resolve_corpus(args.corpus))

    print(f"PLAN: corpus={cname} control={args.control} models={models} seeds={seeds}"
          + (f" lda_k={args.lda_k}" if args.lda_k else " lda_k=auto"))
    if args.dry_run:
        for model in models:
            for seed in seeds:
                print(f"  would run -> results/controls/{cname}/{model}/{args.control}/s{seed:03d}/")
        return

    for model in models:
        for seed in seeds:
            print(f"\n{'=' * 70}\n  {cname} | {model} | {args.control} | seed {seed}\n{'=' * 70}")
            run_one(args.corpus, model, args.control, seed, out_root=args.out_root,
                   cache_root=args.cache_root, lda_k=args.lda_k)


if __name__ == "__main__":
    main()
