"""
Expanded probe of major Nigerian university repositories for OAI-PMH.

Targets the NUC-recognised federal and premier state universities (the
institutions whose theses carry real methodological weight for a dataset),
rather than whatever happens to answer on a first guess.

Path patterns covered:
  DSpace 5/6  : /oai/request
  DSpace 7    : /server/oai/request
  legacy      : /xmlui/oai/request , /jspui/oai/request , /oai2

For each host we also probe the bare base URL so we can distinguish
"repository is down / does not exist" from "repository is up but OAI
is mounted somewhere we did not guess".

Run:  python probe_repositories.py
"""

import concurrent.futures
import json
import ssl
import sys
from urllib.parse import urljoin

import requests
from lxml import etree

NS = {"oai": "http://www.openarchives.org/OAI/2.0/"}

TIMEOUT = 10
UA = "LuminaEduDatasetHarvester/1.0 (NAIC 2026 academic research; contact: team@lumina-edu.ng)"

requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

OAI_PATHS = [
    "/server/oai/request?verb=Identify",
    "/oai/request?verb=Identify",
    "/xmlui/oai/request?verb=Identify",
    "/jspui/oai/request?verb=Identify",
    "/oai2/request?verb=Identify",
    "/dspace-oai/request?verb=Identify",
]

# (label, host) - NUC federal + premier state universities
CANDIDATES = [
    # --- Federal Universities (NUC) ---
    ("University of Lagos",            "https://repository.unilag.edu.ng"),
    ("University of Ibadan",           "https://repository.ui.edu.ng"),
    ("Ahmadu Bello University",        "https://repository.abu.edu.ng"),
    ("Obafemi Awolowo University",     "https://repository.oauife.edu.ng"),
    ("University of Nigeria, Nsukka",  "https://repository.unn.edu.ng"),
    ("University of Ilorin",           "https://repository.unilorin.edu.ng"),
    ("University of Calabar",          "https://repository.unical.edu.ng"),
    ("Nnamdi Azikiwe University",      "https://repository.unizik.edu.ng"),
    ("University of Port Harcourt",    "https://repository.uniport.edu.ng"),
    ("University of Maiduguri",        "https://repository.unimaid.edu.ng"),
    ("University of Benin",            "https://repository.uniben.edu.ng"),
    ("University of Jos",              "https://repository.unijos.edu.ng"),
    ("Bayero University Kano",         "https://repository.buk.edu.ng"),
    ("Usmanu Danfodiyo University",    "https://repository.udus.edu.ng"),
    ("University of Agriculture, Abeokuta", "https://repository.funaab.edu.ng"),
    ("FUT Minna",                      "https://repository.futminna.edu.ng"),
    ("FUT Owerri",                     "https://repository.futo.edu.ng"),
    ("University of Agriculture Makurdi", "https://repository.uam.edu.ng"),
    ("Federal University of Technology, Akure", "https://repository.futa.edu.ng"),
    ("Niger Delta University",         "https://repository.ndu.edu.ng"),
    ("University of Abuja",            "https://repository.uab.edu.ng"),
    ("University of Uyo",              "https://repository.uu.edu.ng"),
    ("Federal University of Tech. Owerri", "https://repository.futo.edu.ng"),
    # --- Premier State Universities ---
    ("Lagos State University",         "https://repository.lasu.edu.ng"),
    ("Njala? (Sierra Leone)",          "https://repository.njala.edu.sl"),
    ("Kaduna State University",        "https://repository.kasu.edu.ng"),
    ("Ahmadu Bello (alt)",             "https://repository.abu.edu.ng"),
    ("Afe Babalola University",        "https://repository.abu.edu.ng"),
    ("University of Cape Coast",       "https://repository.ucc.edu.gh"),
    ("University of Ghana",            "https://repository.ug.edu.gh"),
    ("Kwame Nkrumah Univ. of Sci&Tech", "https://repository.knust.edu.gh"),
    ("University of Nairobi",          "https://repository.uonbi.ac.ke"),
    ("University of Lagos (alt)",      "https://repository.unilag.ac.ng"),
    ("Bowen University",               "https://repository.bowen.edu.ng"),
    ("Babcock University",             "https://repository.babcock.edu.ng"),
    ("Bells University",               "https://repository.bellsuniversity.edu.ng"),
    ("Covenant University",            "https://repository.covenantuniversity.edu.ng"),
    ("Calabar University",             "https://repository.unical.edu.ng"),
    ("Landmark University",            "https://repository.lmu.edu.ng"),
    ("Redeemer's University",          "https://repository.run.edu.ng"),
    ("Ajayi Crowther University",      "https://repository.acu.edu.ng"),
    ("University of Africa Toru-Orua", "https://repository.uat.edu.ng"),
    ("Michael Okpara University",      "https://repository.umue.edu.ng"),
    ("Imo State University",           "https://repository.imsu.edu.ng"),
    ("Ekiti State University",         "https://repository.eksu.edu.ng"),
    ("Delta State University",         "https://repository.delsu.edu.ng"),
    ("Ambrose Alli University",        "https://repository.aau.edu.ng"),
    ("Benson Idahosa University",      "https://repository.biue.edu.ng"),
    ("Madonna University",             "https://repository.mun.edu.ng"),
    ("African University of Science",  "https://repository.aust.edu.ng"),
    ("Bingham University",             "https://repository.binghamedu.edu.ng"),
    ("Veritas University",             "https://repository.veritasuniversity.edu.ng"),
    ("Pan-Atlantic University",        "https://repository.pauc.edu.ng"),
]


def probe(entry):
    label, base = entry
    out = {
        "label": label, "host": base, "oai_url": None, "repo_name": None,
        "earliest_datestamp": None, "live": False, "base_status": None, "error": None,
    }
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.verify = False

    try:
        br = s.get(base, timeout=TIMEOUT, allow_redirects=True)
        out["base_status"] = br.status_code
    except Exception as exc:
        out["error"] = f"base unreachable ({type(exc).__name__})"
        return out

    for path in OAI_PATHS:
        url = urljoin(base, path)
        try:
            r = s.get(url, timeout=TIMEOUT, allow_redirects=True)
        except Exception:
            continue
        if r.status_code != 200 or "OAI-PMH" not in r.text[:2000]:
            continue
        try:
            root = etree.fromstring(r.content)

            def txt(xp):
                n = root.find(xp, NS)
                return n.text.strip() if n is not None and n.text else None

            out.update({
                "oai_url": url,
                "repo_name": txt("oai:Identify/oai:repositoryName"),
                "earliest_datestamp": txt("oai:Identify/oai:earliestDatestamp"),
                "live": True,
            })
            return out
        except Exception:
            continue

    out["error"] = "up but no OAI-PMH at known paths"
    return out


def main():
    print(f"Probing {len(CANDIDATES)} major-university hosts...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(probe, CANDIDATES))

    live = [r for r in results if r["live"]]
    up_no_oai = [r for r in results if not r["live"] and r["base_status"] and r["base_status"] < 400]
    down = [r for r in results if not r["live"] and (not r["base_status"] or r["base_status"] >= 400)]

    def show(title, rows, fmt):
        print("=" * 78)
        print(f"{title} ({len(rows)})")
        print("=" * 78)
        for r in rows:
            fmt(r)
            print()

    def live_fmt(r):
        print(f"  {r['label']}")
        print(f"    {r['oai_url']}")
        print(f"    name    : {r['repo_name']}")
        print(f"    earliest: {r['earliest_datestamp']}")

    def up_fmt(r):
        print(f"  {r['label']:<42} base=HTTP {r['base_status']}  {r['error']}")

    def down_fmt(r):
        print(f"  {r['label']:<42} {r['error']}")

    show("LIVE OAI-PMH", live, live_fmt)
    show("UP BUT NO OAI ON KNOWN PATHS", up_no_oai, up_fmt)
    show("UNREACHABLE", down, down_fmt)

    with open("../data/repository_probe.json", "w", encoding="utf-8") as fh:
        json.dump({"live": live, "up_no_oai": up_no_oai, "down": down}, fh, indent=2, ensure_ascii=False)
    print(f"Wrote data/repository_probe.json - {len(live)} live")
    return 0 if live else 1


if __name__ == "__main__":
    sys.exit(main())