"""
Count harvestable records across every confirmed-live OAI endpoint.

IMPORTANT FIX
-------------
OAI-PMH `ListMetadataFormats` returns the metadata *schema URI* for each
format, NOT the short prefix. The wire protocol requires the PREFIX
(`oai_dc`, `xoai`, `dim`, ...) when calling ListRecords/ListIdentifiers.
Passing the schema URL back verbatim yields `cannotDisseminateFormat` on
every repository, which is what the first two attempts did.

So we advertise many schemas, and must map schema URI -> prefix locally.
Prefixes we can actually use:
  oai_dc  : Dublin Core, carries title/abstract/creator  <-- we want this
  xoai    : DSpace's own crosswalk, also carries abstract + collection
  dim     : DSpace Internal Metadata
  etd_ms  : ETD Metadata Search, thesis-specific
"""

import sys

import requests
from lxml import etree

NS = {"oai": "http://www.openarchives.org/OAI/2.0/"}

UA = "LuminaEduDatasetHarvester/1.0 (NAIC 2026 academic research; contact: team@lumina-edu.ng)"
requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

# schema URI -> OAI metadataPrefix
SCHEMA_TO_PREFIX = {
    "http://www.openarchives.org/OAI/2.0/oai_dc.xsd": "oai_dc",
    "http://www.lyncode.com/schemas/xoai.xsd": "xoai",
    "http://www.dspace.org/schema/dim.xsd": "dim",
    "http://www.ndltd.org/standards/metadata/etdms/1.0/etdms.xsd": "etd_ms",
    "http://www.rioxx.net/schema/v3.0/rioxx/": "rioxx",
    "http://www.loc.gov/standards/marcxml/schema/MARC21slim.xsd": "marc21",
}

LIVE = [
    ("University of Ibadan",  "https://repository.ui.edu.ng/server/oai/request"),
    ("FUNAAB Abeokuta",       "https://repository.funaab.edu.ng/server/oai/request"),
    ("Redeemer's University", "https://repository.run.edu.ng/server/oai/request"),
    ("Babcock University",    "https://repository.babcock.edu.ng/server/oai/request"),
]


def sess():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.verify = False
    return s


def advertised_prefixes(s, url):
    r = s.get(url, params={"verb": "ListMetadataFormats"}, timeout=30)
    root = etree.fromstring(r.content)
    found = []
    for f in root.findall("oai:ListMetadataFormats/oai:metadataFormat", NS):
        schema = (f.findtext("oai:schema", namespaces=NS) or "").strip()
        for known, prefix in SCHEMA_TO_PREFIX.items():
            if schema.startswith(known):
                found.append((prefix, schema))
                break
    return found


def count(s, url, prefix):
    r = s.get(url, params={"verb": "ListIdentifiers", "metadataPrefix": prefix}, timeout=30)
    root = etree.fromstring(r.content)
    err = root.find("oai:error", NS)
    if err is not None:
        return None, err.get("code")
    box = root.find("oai:ListIdentifiers", NS)
    total = box.get("completeListSize") if box is not None else None
    got = len(root.findall("oai:ListIdentifiers/oai:header", NS))
    tok = (root.findtext("oai:resumptionToken", namespaces=NS, default="") or "").strip()
    return (total, got, bool(tok)), None


def main():
    grand, plan = 0, []
    for label, url in LIVE:
        s = sess()
        print("=" * 74)
        print(label)
        print("=" * 74)
        try:
            fmts = advertised_prefixes(s, url)
        except Exception as exc:
            print(f"  ListMetadataFormats failed: {type(exc).__name__}")
            continue
        if not fmts:
            print("  no recognised prefixes advertised")
            continue

        results = []
        for prefix, schema in fmts:
            try:
                info, err = count(s, url, prefix)
            except Exception as exc:
                err = type(exc).__name__
                info = None
            if err:
                print(f"  {prefix:<8} -> {err}")
                continue
            total, got, more = info
            n = int(total) if total and total.isdigit() else got
            print(f"  {prefix:<8} -> size={str(total or f'>={got}'):<10} page={got:<5} more={more}")
            results.append((prefix, n, more))

        if results:
            # Prefer oai_dc; fall back to whatever holds the most records.
            pick = next((r for r in results if r[0] == "oai_dc"), max(results, key=lambda r: r[1]))
            print(f"  --> using '{pick[0]}' (~{pick[1]} records)")
            grand += pick[1]
            plan.append((label, url, pick[0], pick[1]))
        print()

    print("=" * 74)
    print(f"COMBINED HARVEST CEILING: ~{grand} records")
    print("=" * 74)
    for label, url, pfx, n in plan:
        print(f"  {label:<28} {pfx:<8} ~{n:<8} {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())