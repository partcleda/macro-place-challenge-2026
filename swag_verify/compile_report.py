#!/usr/bin/env python3
"""Compile the swag-eligibility report from worklist.tsv + results.tsv."""
import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WL = ROOT / "swag_verify" / "worklist.tsv"
RES = ROOT / "swag_verify" / "results.tsv"
REPORT = ROOT / "swag_verify" / "REPORT.md"

def slug(team):
    s = re.sub(r'[^a-z0-9]+', '_', team.lower()).strip('_')
    return s or "team"

wl = list(csv.DictReader(WL.open(), delimiter="\t"))
# unique slug per run-team (mirror orchestrate.py)
seen = {}
for r in wl:
    if r["decision"] != "run":
        continue
    s = slug(r["team"])
    while s in seen: s += "_x"
    seen[s] = r; r["_slug"] = s

TRIAGE = ROOT / "swag_verify" / "triage_results.tsv"
results = {}
# load batch results first, then let triage re-runs override by slug
for src in [RES, TRIAGE]:
    if src.exists():
        for line in src.read_text().splitlines():
            if not line.strip(): continue
            p = line.split("\t")
            results[p[0]] = {"verdict": p[1] if len(p)>1 else "?",
                             "detail": p[2] if len(p)>2 else "",
                             "entry": p[3] if len(p)>3 else "",
                             "conf": p[4] if len(p)>4 else "",
                             "time": p[5] if len(p)>5 else ""}

PASS_V = {"PASS"}
rows = []
for r in wl:
    team, rank, dec = r["team"], r["rank"], r["decision"]
    if dec == "skip":
        rows.append((rank, team, "SKIPPED", r["reason"], "", ""))
        continue
    res = results.get(r.get("_slug"))
    if not res:
        rows.append((rank, team, "PENDING", "not yet run", "", ""))
        continue
    v = res["verdict"]
    status = "ELIGIBLE" if v in PASS_V else "FAIL"
    rows.append((rank, team, status, f"{v}: {res['detail']}", res["entry"], res["time"]))

# counts
from collections import Counter
status_ct = Counter(x[2] for x in rows)
verdict_ct = Counter(results[r.get("_slug")]["verdict"]
                     for r in wl if r["decision"]=="run" and r.get("_slug") in results)

def rk(x):
    try: return int(x[0])
    except: return 999

rows.sort(key=lambda x: (x[2] != "ELIGIBLE", rk(x)))

with REPORT.open("w") as f:
    f.write("# Swag-eligibility verification — minimum functionality (ibm01 → valid placement)\n\n")
    f.write("**Bar:** submission runs end-to-end on ibm01 and produces a 0-overlap (VALID) placement.\n\n")
    f.write("**Method:** air-gapped Docker, each submission limited to 1 GPU / 16 cores / ≤40 GB "
            "(mirrors the eval machine per-submission). Challenge `macro_place` is authoritative "
            "(scorer integrity); a team's *extra* `macro_place.*` modules are overlaid so their "
            "imports resolve; bundled decoy placers (`will_seed`/`examples`) excluded by content hash. "
            "Main pass capped at 30 min; fixed-budget placers re-run at a 55–70 min cap (the contest's "
            "1 h/bench rule). DREAMPlace-bundled teams attempted in a DREAMPlace-compiled image.\n\n")
    f.write("## Summary\n\n")
    for k in ["ELIGIBLE","FAIL","SKIPPED","PENDING"]:
        if status_ct.get(k): f.write(f"- **{k}**: {status_ct[k]}\n")
    f.write(f"\nVerdict breakdown (run set): {dict(verdict_ct)}\n\n")
    f.write("## Per-team\n\n")
    f.write("| Rank | Team | Status | Detail | Entry | Time |\n|---|---|---|---|---|---|\n")
    for rank, team, status, detail, entry, t in rows:
        emoji = {"ELIGIBLE":"✅","FAIL":"❌","SKIPPED":"⏭️","PENDING":"⏳"}.get(status,"")
        detail = (detail or "").replace("|","\\|")[:90]
        f.write(f"| {rank} | {team} | {emoji} {status} | {detail} | `{entry}` | {t} |\n")

    f.write("\n## Caveats (eligible-but-borderline)\n\n")
    f.write("- **Seed-legalizer cluster** (proxy 1.0385 / 1.2253 across several teams): the provided "
            "ibm01 seed scores 1.0385 but is INVALID (69 overlaps); these teams legalize it to 0 "
            "overlaps with minimal displacement (valid + functional, but low-novelty — they pass).\n")
    f.write("- **Binghamton**: repo ships pre-computed placements + an `export.py` (team's note: source "
            "not provided). Passes by emitting stored placements, not a from-scratch placer.\n")
    f.write("- **Degenerate-but-valid**: KKPlace (proxy≈907), Team Rocket (≈53.6), SnoobQuants (≈7.1) "
            "produce 0-overlap placements of very poor quality — they clear the functionality bar only.\n")
    f.write("- A few entry-points were auto-detected at low confidence; spot-check if a team disputes.\n")

print(f"status: {dict(status_ct)}")
print(f"verdicts: {dict(verdict_ct)}")
print(f"-> {REPORT}")
