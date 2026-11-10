"""
controls.py: seeded run harness for LDA / HDP control and null-model runs.

It runs a (corpus, model, control, seed) combination: load the corpus, apply a control
transform, vectorize, fit LDA or HDP, and save a consistent output layout. Controls live in
controls_synthetic.py; the dispatch is apply_control() below.

Controls (--control):
    real       the corpus unchanged (at seed=42 + matching --lda-k this reproduces the
               existing results/{lda,hdp}/<corpus> output byte-for-byte; commit 19bc54a)
    shuffle    corpus-wide word shuffle (destroys within-paper co-occurrence)
    resample   redraw each paper from the corpus-wide word distribution
    uniform    replace every token with a uniform-random vocabulary word
    gibberish  interleave random non-vocabulary noise words
    nowm       remove the "working memory" stems, then model as usual
    synthetic  POSITIVE control: a corpus with KNOWN planted topics (saves ground_truth.json)

Vocabulary is frozen on the REAL corpus for real/shuffle/resample/uniform/nowm (keeps real vs
null comparable and prevents the empty-vocab crash that killed scrambled corpora); gibberish
and synthetic are vectorized on their own corpus. Controls that collapse to too few non-empty
documents are recorded as status="skipped" rather than crashing a seed sweep.

Usage (from lda/):
    python controls.py --corpus 100 --model both --seeds 42 --lda-k 5          # reproduce real
    python controls.py --corpus 100 --model lda  --seeds 0-4  --control real   # seed jitter
    python controls.py --corpus 100 --model both --seeds 0-29 --control all --lda-k 5
    python controls.py --corpus 1k   --model lda  --seeds 0-2  --control shuffle,uniform
    python controls.py --corpus 100 --model both --seeds 42 --control all --dry-run

Output: results/controls/<corpus>/<model>/<control>/s<NNN>/
    run.json (+ topic_word_distributions.csv, topic_prevalence.csv when the run succeeds)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from test_models_all import (  # noqa: E402
    corpus_name, load_corpus, make_vectorizer, run_lda_auto, run_hdp,
)
from controls_synthetic import (  # noqa: E402
    TEXT_CONTROLS, FRESH_VOCAB_CONTROLS, ALL_CONTROLS, make_synthetic_known_topics,
)

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

    - "real"  -> texts unchanged (at seed=42 with the same fixed_k this reproduces the
      existing results/{lda,hdp}/<corpus> output byte-for-byte; see commit 19bc54a).
    - any name in controls_synthetic.TEXT_CONTROLS (shuffle / resample / uniform /
      gibberish / nowm) -> fn(texts, seed), a same-length list of strings.

    "synthetic" (the known-topic positive control) is generative, not a text->text
    transform, so it is handled in run_one (it needs out_dir to save ground truth).
    """
    if control == "real":
        return texts
    if control in TEXT_CONTROLS:
        return TEXT_CONTROLS[control](texts, seed)
    raise NotImplementedError(
        f"control={control!r} is not a known control. Known: {ALL_CONTROLS}."
    )


def _write_skip(out_dir, *, control, corpus, model, seed, n_docs, reason, log):
    """Record a control that could not be modelled (e.g. too few non-empty docs) instead
    of crashing, so a seed sweep keeps going."""
    run = {"schema": 1, "status": "skipped", "reason": reason,
           "control": control, "corpus": corpus, "model": model, "seed": seed,
           "n_docs": n_docs, "out_dir": out_dir}
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(run, fh, indent=2)
    log(f"  -> SKIPPED ({reason})")
    return run


def run_one(corpus_arg, model, control, seed, *, out_root="results/controls",
            cache_root="cache", lda_k=None, synth_k=5, min_live_docs=3, log=print):
    zip_path = resolve_corpus(corpus_arg)
    cname = corpus_name(zip_path)
    label = f"{cname} · {control} · seed {seed}"

    out_dir = os.path.join(out_root, cname, model, control, f"s{seed:03d}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    real_texts, stopwords = load_corpus(zip_path, cache_root=cache_root)
    ground_truth = None
    if control == "synthetic":
        texts, ground_truth = make_synthetic_known_topics(len(real_texts), n_topics=synth_k, seed=seed)
    else:
        texts = apply_control(control, real_texts, corpus=cname, seed=seed)
    load_s = time.time() - t0

    # Vocabulary: freeze on the REAL corpus for structure-destroying controls (keeps real vs
    # null comparable + avoids the empty-vocab crash); fit on the control corpus itself when
    # the control introduces new words (gibberish, synthetic).
    fit_on = texts if control in FRESH_VOCAB_CONTROLS else real_texts
    vocab_source = "control" if control in FRESH_VOCAB_CONTROLS else "real"
    try:
        vectorizer = make_vectorizer(len(fit_on), stopwords)
        vectorizer.fit(fit_on)
    except ValueError as e:
        return _write_skip(out_dir, control=control, corpus=cname, model=model, seed=seed,
                           n_docs=len(texts), reason=f"empty vocabulary at fit: {e}", log=log)

    # Guard: make sure the control corpus still has enough non-empty documents to model.
    dtm_check = vectorizer.transform(texts)
    live_docs = int((dtm_check.sum(axis=1) > 0).sum())
    if dtm_check.nnz == 0 or live_docs < min_live_docs:
        return _write_skip(out_dir, control=control, corpus=cname, model=model, seed=seed,
                           n_docs=len(texts),
                           reason=f"only {live_docs} non-empty docs after vectorizing", log=log)

    t0 = time.time()
    if model == "lda":
        metrics = run_lda_auto(texts, stopwords, label, out_dir=out_dir, fixed_k=lda_k,
                               seed=seed, vectorizer=vectorizer)
    elif model == "hdp":
        metrics = run_hdp(texts, stopwords, label, out_dir=out_dir, seed=seed, vectorizer=vectorizer)
    else:
        raise ValueError(f"unknown model {model!r}; use lda or hdp")
    fit_s = time.time() - t0

    if ground_truth is not None:
        with open(os.path.join(out_dir, "ground_truth.json"), "w") as fh:
            json.dump(ground_truth, fh, indent=2)

    run = {
        "schema": 1, "status": "ok",
        "control": control, "corpus": cname, "model": model, "seed": seed,
        "n_docs": len(texts), "n_topics": metrics["n_topics"],
        "vocab_size": len(vectorizer.vocabulary_), "vocab_source": vocab_source,
        "live_docs": live_docs,
        "metrics": {k: v for k, v in metrics.items() if k not in ("topic_words", "overlaps")},
        "params": {"lda_k": lda_k, "synth_k": synth_k if control == "synthetic" else None},
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


def parse_controls(spec):
    """'real' -> ['real']; 'shuffle,uniform' -> [...]; 'all' -> every control."""
    if spec.strip() == "all":
        return list(ALL_CONTROLS)
    out = []
    for part in spec.split(","):
        c = part.strip()
        if c and c not in ALL_CONTROLS:
            raise ValueError(f"unknown control {c!r}; choose from {ALL_CONTROLS} or 'all'")
        if c:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="100 | 1k | 10k | a zip path")
    ap.add_argument("--model", required=True, choices=["lda", "hdp", "both"])
    ap.add_argument("--control", default="real",
                    help="'real', a comma list (e.g. real,shuffle,uniform), or 'all'. "
                         f"Known: {ALL_CONTROLS}")
    ap.add_argument("--seeds", default="42", help="'42', '0-29', or '0,5,10'")
    ap.add_argument("--lda-k", type=int, default=None, help="fixed LDA k (omit for the auto-k coherence sweep)")
    ap.add_argument("--synth-k", type=int, default=5, help="number of planted topics for control=synthetic")
    ap.add_argument("--out-root", default="results/controls")
    ap.add_argument("--cache-root", default="cache")
    ap.add_argument("--dry-run", action="store_true", help="print the run plan without fitting anything")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    controls = parse_controls(args.control)
    models = ["lda", "hdp"] if args.model == "both" else [args.model]
    cname = corpus_name(resolve_corpus(args.corpus))

    print(f"PLAN: corpus={cname} controls={controls} models={models} seeds={seeds}"
          + (f" lda_k={args.lda_k}" if args.lda_k else " lda_k=auto"))
    if args.dry_run:
        for control in controls:
            for model in models:
                for seed in seeds:
                    print(f"  would run -> {args.out_root}/{cname}/{model}/{control}/s{seed:03d}/")
        return

    for control in controls:
        for model in models:
            for seed in seeds:
                print(f"\n{'=' * 70}\n  {cname} | {model} | {control} | seed {seed}\n{'=' * 70}")
                run_one(args.corpus, model, control, seed, out_root=args.out_root,
                        cache_root=args.cache_root, lda_k=args.lda_k, synth_k=args.synth_k)


if __name__ == "__main__":
    main()
