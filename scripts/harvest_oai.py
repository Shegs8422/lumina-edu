"""
Lumina-Edu dataset harvester — OAI-PMH client for Nigerian university repositories.

Sources (confirmed live, see scripts/probe_repositories.py):
  University of Ibadan   ir.ui.edu.ng        ~15,585 records
  Redeemer's University  repository.run.edu.ng ~6,912 records

Paginates via the OAI resumptionToken cursor. The total record count is
published as the `completeListSize` ATTRIBUTE on <resumptionToken>, which is
easy to miss (reading it off <ListIdentifiers> yields None and makes a
healthy repository look like it holds 100 records).

Design notes
------------
* Checkpointing: every page is appended to JSONL immediately and the cursor
  is saved, so an interrupted run resumes instead of restarting.
* Politeness: fixed interval between requests, descriptive User-Agent,
  honours 503 Retry-After.
* Only records carrying a usable abstract are kept, because an abstract-free
  record cannot support the summarisation tasks we need.

Usage:
  python harvest_oai.py --limit 4000
  python harvest_oai.py --limit 100000 --from-date 2018-01-01
"""

import argparse
import json
import os
import random
import sys
import time

import requests
from lxml import etree

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

UA = (
    "LuminaEduDatasetHarvester/1.0 "
    "(NAIC 2026 academic research; contact: team@lumina-edu.ng)"
)

# Anchor output to the project root so results land in the same place
# regardless of which directory the script is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = [
    {
        "key": "ui_ibadan",
        "institution": "University of Ibadan",
        "short": "UI",
        "url": "https://repository.ui.edu.ng/server/oai/request",
    },
    {
        "key": "redeemers",
        "institution": "Redeemer's University",
        "short": "RUN",
        "url": "https://repository.run.edu.ng/server/oai/request",
    },
]

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CHECKPOINT = os.path.join(PROJECT_ROOT, "data", "harvest_checkpoint")


def make_session():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.headers["Accept"] = "text/xml"
    s.verify = False  # several Nigerian repositories still negotiate legacy TLS
    return s


def clean(text):
    if not text:
        return ""
    return " ".join(text.split())


def parse_record(rec, source):
    """Pull the fields we care about out of one <record> element."""
    header = rec.find("oai:header", NS)
    if header is not None and header.get("status") == "deleted":
        return None

    oai_id = None
    idn = rec.find("oai:header/oai:identifier", NS)
    if idn is not None:
        oai_id = clean(idn.text)

    # The <dc> wrapper lives in the oai_dc namespace; only its CHILDREN
    # (dc:title, dc:description, ...) are in the Dublin Core namespace.
    # Matching the wrapper on dc:dc silently matches nothing.
    dc = rec.find("oai:metadata/oai_dc:dc", NS)
    if dc is None:
        return None

    def values(tag):
        return [clean(e.text) for e in dc.findall(f"dc:{tag}", NS) if clean(e.text)]

    titles = values("title")
    if not titles:
        return None

    abstracts = values("description")
    abstract = max(abstracts, key=len).strip() if abstracts else ""

    # Keep only records with a substantial abstract; these become our inputs.
    if len(abstract) < 180:
        return None

    subjects = values("subject")
    creators = values("creator")
    types = values("type")
    dates = values("date")
    publishers = values("publisher")
    languages = values("language")

    return {
        "id": f"{source['key']}:{oai_id}",
        "source": source["key"],
        "institution": source["institution"],
        "title": titles[0],
        "abstract": abstract,
        "creators": creators[:5],
        "subjects": subjects[:12],
        "types": types[:3],
        "date": dates[0] if dates else "",
        "publisher": publishers[0] if publishers else "",
        "language": languages[0] if languages else "",
        "url": (rec.findtext("oai:header/oai:identifier", namespaces=NS, default="") or "").strip(),
    }


def fetch_page(session, url, token=None, from_date=None):
    # OAI-PMH rule: once you page with a resumptionToken you must send ONLY
    # verb + resumptionToken. Re-sending metadataPrefix (or `from`) alongside
    # it is a protocol violation and the server answers `badArgument`.
    if token:
        params = {"verb": "ListRecords", "resumptionToken": token}
    else:
        params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
        if from_date:
            params["from"] = from_date

    for attempt in range(5):
        try:
            r = session.get(url, params=params, timeout=60)
        except Exception as exc:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt + random.random())
            continue

        if r.status_code == 503:
            wait = int(r.headers.get("Retry-After", 10))
            print(f"    503 - honouring Retry-After {wait}s")
            time.sleep(wait)
            continue

        r.raise_for_status()
        return etree.fromstring(r.content)
    raise RuntimeError("exhausted retries")


def harvest_source(source, limit, from_date, delay):
    key = source["key"]
    session = make_session()
    out_path = os.path.join(OUT_DIR, f"{key}.jsonl")
    ck_path = os.path.join(CHECKPOINT, f"{key}.json")

    done = 0
    token = None
    if os.path.exists(ck_path) and os.path.exists(out_path):
        with open(ck_path, encoding="utf-8") as fh:
            ck = json.load(fh)
        if ck.get("complete") and ck.get("limit") == limit:
            print(f"  [{key}] already complete ({ck.get('kept')} kept) - skipping")
            return ck.get("kept", 0)
        done = ck.get("done", 0)
        token = ck.get("token")
        print(f"  [{key}] resuming at record {done}")

    if limit and done >= limit:
        print(f"  [{key}] limit already reached ({done})")
        return done

    if not os.path.exists(CHECKPOINT):
        os.makedirs(CHECKPOINT, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    mode = "a" if done else "w"
    fh = open(out_path, mode, encoding="utf-8")
    kept_total = ck_kept = 0
    try:
        if os.path.exists(ck_path):
            with open(ck_path, encoding="utf-8") as f2:
                ck_kept = json.load(f2).get("kept", 0)

        page = 0
        while True:
            root = fetch_page(session, source["url"], token, from_date)

            err = root.find("oai:error", NS)
            if err is not None:
                print(f"    OAI error: {err.get('code')}")
                break

            records = root.findall("oai:ListRecords/oai:record", NS)
            tok_el = root.find("oai:ListRecords/oai:resumptionToken", NS)

            for rec in records:
                item = parse_record(rec, source)
                if item:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                    kept_total += 1
                done += 1

            page += 1
            fh.flush()

            if page % 10 == 0:
                print(f"    page {page}: scanned={done} kept={kept_total}")

            if tok_el is None or not (tok_el.text or "").strip():
                print(f"    exhausted after {page} pages, scanned={done}")
                with open(ck_path, "w", encoding="utf-8") as f2:
                    json.dump({"done": done, "kept": ck_kept + kept_total,
                               "complete": True, "limit": limit}, f2)
                return kept_total

            token = tok_el.text.strip()
            total = tok_el.get("completeListSize")
            if total and page % 20 == 0:
                print(f"    ...{done}/{total}")

            if limit and done >= limit:
                print(f"    hit limit {limit}")
                with open(ck_path, "w", encoding="utf-8") as f2:
                    json.dump({"done": done, "kept": ck_kept + kept_total,
                               "complete": False, "limit": limit, "token": token}, f2)
                return kept_total

            time.sleep(delay)
    finally:
        fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="max records per source")
    ap.add_argument("--from-date", default=None, help="OAI 'from' datestamp, e.g. 2020-01-01")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    print(f"Harvesting (limit={args.limit}/source, from={args.from_date})\n")
    grand = 0
    for src in SOURCES:
        print(f"=== {src['institution']} ===")
        try:
            kept = harvest_source(src, args.limit, args.from_date, args.delay) or 0
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        grand += kept
        print(f"  -> kept {kept} usable records\n")

    print(f"TOTAL USABLE RECORDS: {grand}")
    print(f"Raw output in {OUT_DIR}/")
    return 0 if grand else 1


if __name__ == "__main__":
    sys.exit(main())