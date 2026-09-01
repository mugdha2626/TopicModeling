#!/usr/bin/env python3
"""
fetch_papers.py — bulk-fetch full text for a PubMed CSV export.

Two legal, sanctioned sources:
  Stage 1  PMC Open Access Web Service  (articles with a PMCID that are in the OA subset)
  Stage 2  Unpaywall                    (everything else that has a legally-posted free copy)

Anything neither stage can get is written to still_missing.csv for your
library's interlibrary loan / document delivery service.

Usage:
    pip install requests pandas
    python fetch_papers.py --csv csv-workingmem-set.csv --email you@university.edu

    # try a few first
    python fetch_papers.py --csv csv-workingmem-set.csv --email you@university.edu --limit 25

    # just Unpaywall, skip PMC
    python fetch_papers.py --csv ... --email ... --stage 2

Resumable: state is kept in pdfs/_state.json. Ctrl-C and rerun any time.
"""

import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

# NCBI retired the old OA Web Service + FTP endpoints in Aug 2026. The OA
# subset now lives in the world-readable S3 bucket below (us-east-1, no login),
# reorganized as one folder per article: PMC<id>.<version>/PMC<id>.<version>.pdf
PMC_S3 = "https://pmc-oa-opendata.s3.amazonaws.com/"
S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"

# Be a good citizen. NCBI asks for <=3 req/s without an API key; Unpaywall
# asks for <=100k/day and a real email. These delays keep you well under both.
PMC_DELAY = 0.4
UNPAYWALL_DELAY = 0.15
DOWNLOAD_DELAY = 0.3
TIMEOUT = 60


def log(msg):
    print(msg, flush=True)


class State:
    """Tracks what's already done so reruns are cheap."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {"done": {}, "failed": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                log("! state file unreadable, starting fresh")
        self._n = 0

    def is_done(self, key):
        return key in self.data["done"]

    def mark_done(self, key, path, source):
        self.data["done"][key] = {"file": str(path), "source": source}
        self.data["failed"].pop(key, None)
        self._maybe_flush()

    def mark_failed(self, key, reason):
        self.data["failed"][key] = reason
        self._maybe_flush()

    def _maybe_flush(self):
        self._n += 1
        if self._n % 25 == 0:
            self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=1))


def session_with_retries(email):
    s = requests.Session()
    s.headers.update(
        {
            # Publishers block obvious bots. Identify yourself honestly instead.
            "User-Agent": f"fetch_papers.py/1.0 (mailto:{email})",
            "Accept": "application/pdf,application/xml,text/html,*/*",
        }
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=3, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def looks_like_pdf(content):
    return content[:5] == b"%PDF-"


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:100]


# ---------------------------------------------------------------- stage 1: PMC


def pmc_s3_pdf_key(sess, pmcid):
    """List this article's folder in the OA bucket and return its .pdf object key.

    Returns (pdf_key, all_keys). all_keys empty -> article isn't in the OA subset.
    all_keys non-empty but pdf_key None -> OA, but only txt/xml is posted (common
    for author manuscripts).
    """
    r = sess.get(PMC_S3, params={"list-type": "2", "prefix": pmcid + "."}, timeout=TIMEOUT)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    keys = [k.text for k in root.findall(".//s:Key", S3_NS)]
    pdfs = [k for k in keys if k.lower().endswith(".pdf")]
    return (pdfs[0] if pdfs else None), keys


def stage_pmc(df, sess, outdir, state, limit):
    rows = df[df["PMCID"] != ""]
    log(f"\n=== Stage 1: PMC Open Access — {len(rows)} candidates ===")
    got = skipped = failed = 0

    for i, row in enumerate(rows.itertuples(), 1):
        if limit and got >= limit:
            break
        pmcid = row.PMCID
        if state.is_done(pmcid):
            skipped += 1
            continue

        dest = outdir / f"{pmcid}.pdf"
        try:
            pdf_key, keys = pmc_s3_pdf_key(sess, pmcid)
            time.sleep(PMC_DELAY)

            if not keys:
                # Free to read on the PMC site, but not in the OA reuse subset.
                state.mark_failed(pmcid, "not in OA subset")
                failed += 1
                continue
            if not pdf_key:
                # OA, but only txt/xml posted (Stage 2 / Unpaywall may still find a PDF).
                state.mark_failed(pmcid, "OA but no PDF (txt/xml only)")
                failed += 1
                continue

            r = sess.get(PMC_S3 + pdf_key, timeout=TIMEOUT)
            r.raise_for_status()
            time.sleep(DOWNLOAD_DELAY)

            data = r.content
            if not looks_like_pdf(data):
                state.mark_failed(pmcid, "S3 object was not a PDF")
                failed += 1
                continue

            dest.write_bytes(data)
            state.mark_done(pmcid, dest, "pmc-oa")
            got += 1

        except KeyboardInterrupt:
            raise
        except Exception as e:
            state.mark_failed(pmcid, f"{type(e).__name__}: {e}")
            failed += 1

        if i % 50 == 0:
            log(f"  [{i}/{len(rows)}] got={got} skipped={skipped} failed={failed}")

    state.save()
    log(f"Stage 1 done: {got} new, {skipped} already had, {failed} unavailable")
    return got


# --------------------------------------------------------- stage 2: Unpaywall


def stage_unpaywall(df, sess, outdir, state, email, limit):
    rows = df[(df["DOI"] != "")]
    rows = rows[~rows["PMCID"].map(state.is_done)]
    log(f"\n=== Stage 2: Unpaywall — {len(rows)} candidates ===")
    got = skipped = closed = failed = 0

    for i, row in enumerate(rows.itertuples(), 1):
        if limit and got >= limit:
            break
        doi = row.DOI
        key = doi
        if state.is_done(key) or state.is_done(row.PMCID or "\0"):
            skipped += 1
            continue

        try:
            r = sess.get(UNPAYWALL.format(doi=doi), params={"email": email}, timeout=TIMEOUT)
            time.sleep(UNPAYWALL_DELAY)
            if r.status_code == 404:
                state.mark_failed(key, "DOI unknown to Unpaywall")
                failed += 1
                continue
            r.raise_for_status()
            meta = r.json()

            loc = meta.get("best_oa_location") or {}
            url = loc.get("url_for_pdf") or loc.get("url")
            if not url:
                state.mark_failed(key, "no OA copy")
                closed += 1
                continue

            p = sess.get(url, timeout=TIMEOUT, allow_redirects=True)
            time.sleep(DOWNLOAD_DELAY)
            if p.status_code != 200 or not looks_like_pdf(p.content):
                # Landing page, paywall interstitial, or Cloudflare. Record the
                # URL so you can grab it by hand later.
                state.mark_failed(key, f"not a PDF: {url}")
                failed += 1
                continue

            dest = outdir / f"{safe_name(doi)}.pdf"
            dest.write_bytes(p.content)
            state.mark_done(key, dest, "unpaywall")
            got += 1

        except KeyboardInterrupt:
            raise
        except Exception as e:
            state.mark_failed(key, f"{type(e).__name__}: {e}")
            failed += 1

        if i % 100 == 0:
            log(f"  [{i}/{len(rows)}] got={got} paywalled={closed} failed={failed}")

    state.save()
    log(f"Stage 2 done: {got} new, {closed} paywalled, {failed} errors")
    return got


# ---------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--email", required=True, help="Required by Unpaywall and polite to NCBI")
    ap.add_argument("--out", default="pdfs")
    ap.add_argument("--stage", type=int, choices=[1, 2], help="Run only one stage")
    ap.add_argument("--limit", type=int, help="Stop after N new PDFs per stage (for testing)")
    ap.add_argument("--zip", action="store_true", help="Zip the PDFs when finished")
    args = ap.parse_args()

    if "@" not in args.email:
        sys.exit("--email must be a real address; Unpaywall rejects placeholders")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype=str).fillna("")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    for need in ("PMID", "PMCID", "DOI"):
        if need not in df.columns:
            sys.exit(f"CSV is missing a {need} column — is this a PubMed CSV export?")
    log(f"Loaded {len(df)} records from {args.csv}")

    state = State(outdir / "_state.json")
    sess = session_with_retries(args.email)

    try:
        if args.stage in (None, 1):
            stage_pmc(df, sess, outdir, state, args.limit)
        if args.stage in (None, 2):
            stage_unpaywall(df, sess, outdir, state, args.email, args.limit)
    except KeyboardInterrupt:
        log("\nInterrupted — progress saved, rerun to resume.")
    finally:
        state.save()

    # Leftovers -> interlibrary loan
    done = state.data["done"]
    missing = df[~(df["PMCID"].isin(done) | df["DOI"].isin(done))]
    missing.to_csv(outdir / "still_missing.csv", index=False)

    pdfs = sorted(outdir.glob("*.pdf"))
    log(f"\n{'='*52}")
    log(f"PDFs on disk:        {len(pdfs)}")
    log(f"Still missing:       {len(missing)}  -> {outdir/'still_missing.csv'}")
    log(f"Failure reasons:     {outdir/'_state.json'}")

    if args.zip and pdfs:
        import zipfile

        zpath = Path(f"{outdir.name}.zip")
        log(f"\nZipping {len(pdfs)} PDFs -> {zpath} (this takes a few minutes)")
        # PDFs are already compressed; ZIP_STORED is much faster and barely bigger.
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for p in pdfs:
                z.write(p, p.name)
            z.write(outdir / "still_missing.csv", "still_missing.csv")
        log(f"Done: {zpath}  ({zpath.stat().st_size / 1e9:.2f} GB)")
        log("Drag that into Google Drive when it finishes.")


if __name__ == "__main__":
    main()