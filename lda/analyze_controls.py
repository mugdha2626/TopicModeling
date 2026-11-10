"""
analyze_controls.py - summarise control / null-model runs and compare them to the real corpus.

Reads the run.json files that controls.py writes under
    results/controls/<corpus>/<model>/<control>/sNNN/run.json
and, for each (corpus, model, control), reports the distribution of a metric across seeds
(mean, std, n). It then tests each null control against the REAL runs with a Mann-Whitney U
test, so you can say whether the real topics score better than chance and by how much.

This is what makes the control runs mean something: on their own they are folders of numbers;
here we ask "do real topics beat the null, for this metric?" and "how much do the models jump
seed to seed?" (the spread of the real runs).

Usage (from lda/):
    python analyze_controls.py --corpus 100
    python analyze_controls.py --corpus 100 --metric diversity
    python analyze_controls.py --corpus 100 --metric all
"""
import argparse
import glob
import json
import math
import os

try:
    from scipy.stats import mannwhitneyu
    SCIPY = True
except ImportError:
    SCIPY = False

METRICS = ["coherence", "diversity", "avg_overlap"]
# For each metric, does a HIGHER value mean "more real structure"?
HIGHER_IS_MORE_REAL = {"coherence": True, "diversity": True, "avg_overlap": False}


def load_runs(out_root, corpus):
    """Return {model: {control: {'ok': [run,...], 'skipped': n}}}."""
    runs = {}
    pattern = os.path.join(out_root, corpus, "*", "*", "s*", "run.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            d = json.load(fh)
        model, control = d.get("model"), d.get("control")
        slot = runs.setdefault(model, {}).setdefault(control, {"ok": [], "skipped": 0})
        if d.get("status") == "skipped":
            slot["skipped"] += 1
        else:
            slot["ok"].append(d)
    return runs


def values(run_list, metric):
    """Finite metric values across runs (NaN/inf dropped - degenerate scrambled topics can
    make c_v coherence NaN)."""
    out = []
    for d in run_list:
        v = d.get("metrics", {}).get(metric)
        if v is not None and math.isfinite(float(v)):
            out.append(float(v))
    return out


def mean(xs):
    return sum(xs) / len(xs)


def pstdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def summarize(vals):
    if not vals:
        return "n=0"
    if len(vals) == 1:
        return f"n=1 {vals[0]:.3f}"
    return f"n={len(vals)} {mean(vals):.3f}±{pstdev(vals):.3f} [{min(vals):.3f},{max(vals):.3f}]"


def report_metric(runs, metric, log):
    higher_real = HIGHER_IS_MORE_REAL[metric]
    log(f"\n{'='*86}\n  METRIC: {metric}   ({'higher' if higher_real else 'lower'} = more real structure)\n{'='*86}")
    summary = {}
    for model in sorted(runs):
        controls = runs[model]
        real_vals = values(controls.get("real", {}).get("ok", []), metric)
        real_mean = mean(real_vals) if real_vals else None
        log(f"\n  [{model.upper()}]   real: {summarize(real_vals)}"
            + (f"   (seed-to-seed spread = {pstdev(real_vals):.3f})" if len(real_vals) > 1 else ""))
        log(f"    {'control':10s} | {'seeds':>24s} | {'vs real (Δmean)':>15s} | {'Mann-Whitney p':>14s} | verdict")
        log(f"    {'-'*10}-+-{'-'*24}-+-{'-'*15}-+-{'-'*14}-+--------")
        for control in sorted(controls):
            if control == "real":
                continue
            ok = controls[control]["ok"]
            skipped = controls[control]["skipped"]
            vals = values(ok, metric)
            if not vals:
                log(f"    {control:10s} | {'(no successful runs, '+str(skipped)+' skipped)':>24s} |")
                continue
            dmean = (mean(vals) - real_mean) if real_mean is not None else None
            p = None
            if SCIPY and real_vals and len(vals) >= 1 and len(real_vals) >= 1 and (len(vals) + len(real_vals)) > 2:
                try:
                    p = float(mannwhitneyu(real_vals, vals, alternative="two-sided").pvalue)
                except ValueError:
                    p = None
            # verdict: is real clearly better than this null?
            verdict = "?"
            if dmean is not None:
                real_better = (dmean > 0) if higher_real else (dmean < 0)
                if p is not None:
                    verdict = ("real>null" if real_better else "null>=real") + (" (sig)" if p < 0.05 else " (ns)")
                else:
                    verdict = "real>null" if real_better else "null>=real"
            skip_note = f" +{skipped} skipped" if skipped else ""
            log(f"    {control:10s} | {summarize(vals)+skip_note:>24s} | "
                f"{(f'{dmean:+.3f}' if dmean is not None else '-'):>15s} | "
                f"{(f'{p:.4f}' if p is not None else '-'):>14s} | {verdict}")
            summary.setdefault(model, {})[control] = {
                "n": len(vals), "mean": mean(vals),
                "delta_vs_real": dmean, "mannwhitney_p": p, "skipped": skipped,
            }
        summary.setdefault(model, {})["real"] = {"n": len(real_vals),
                                                 "mean": real_mean,
                                                 "seed_spread": pstdev(real_vals) if len(real_vals) > 1 else None}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus name, e.g. 100_sorted_pdf (or 100 / 1k / 10k)")
    ap.add_argument("--out-root", default="results/controls")
    ap.add_argument("--metric", default="coherence", help="coherence | diversity | avg_overlap | all")
    args = ap.parse_args()

    alias = {"100": "100_sorted_pdf", "1k": "1000_sorted_pdf", "10k": "10000_sorted_pdf"}
    corpus = alias.get(args.corpus, args.corpus)

    runs = load_runs(args.out_root, corpus)
    if not runs:
        print(f"No run.json files under {os.path.join(args.out_root, corpus)}. "
              f"Run controls.py first.")
        return
    if not SCIPY:
        print("  (scipy not available - Mann-Whitney p-values will be blank)")

    metrics = METRICS if args.metric == "all" else [args.metric]
    full = {}
    lines = []
    log = lambda s="": (print(s), lines.append(s))
    log(f"\nCONTROL ANALYSIS - corpus={corpus}")
    n_models = len(runs)
    n_controls = len({c for m in runs.values() for c in m})
    log(f"  {n_models} model(s), {n_controls} control(s) found under {args.out_root}/{corpus}/")
    for metric in metrics:
        full[metric] = report_metric(runs, metric, log)

    out_dir = os.path.join(args.out_root, corpus)
    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as fh:
        json.dump(full, fh, indent=2)
    with open(os.path.join(out_dir, "analysis_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  saved -> {out_dir}/analysis_summary.json  and  analysis_report.txt")


if __name__ == "__main__":
    main()
