#!/usr/bin/env python3
"""Consolidate the FULL swag-eligibility picture across ALL leaderboard teams.

Merges:
  - the latest leaderboard (default branch arg) -> all ranked teams + Verified flag
  - swag_verify/REPORT.md (this branch)          -> verdicts for the unverified cohort
  - DQ rows                                       -> disqualified entries

The leaderboard is read from a git ref (argv[2], default 'add-shoom-rescore') so the
report reflects the CURRENT standings (e.g. Archgen re-instated at #1) rather than a
stale snapshot on the PR branch. Emits swag_verify/FULL_REPORT.md. argv[1] = date.
"""
import re, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATE = sys.argv[1] if len(sys.argv) > 1 else ""
LB_REF = sys.argv[2] if len(sys.argv) > 2 else "origin/main"
readme = subprocess.run(["git", "show", f"{LB_REF}:README.md"], capture_output=True, text=True).stdout
report = (ROOT / "swag_verify" / "REPORT.md").read_text()

# Teams still pending a faithful Dockerfile / API re-run (currently ❌ but may recover).
RERUN = {"ICAS_placer", "RuslanPlace", "Hoop Dreams", "ilovekiro", "Combobulating"}

# Verdicts established THIS verification pass — override the leaderboard/report categorization.
OVERRIDES = {
    "Dragonfly":        ("✅ ELIGIBLE",   "RECOVERED — built team Dockerfile; ibm01 VALID (2757s)"),
    "BakaBobo":         ("✅ ELIGIBLE",   "RECOVERED — false DQ; ships macro_place.fast_proxy (overlay resolves it); ibm01 VALID (1.0109)"),
    "Macropolis":       ("✅ ELIGIBLE",   "NEW form-only entry; ibm01 VALID (1.0276, self-contained)"),
    "Nikunj Bhatt":     ("✅ ELIGIBLE",   "NEW form-only entry; ibm01 VALID (1.0385, legalizes seed)"),
    "AxeCap":           ("✅ ELIGIBLE",   "NEW form-only entry; ibm01 VALID (1.2391)"),
    "A-cat-suki":       ("✅ ELIGIBLE",   "NEW form-only entry; ibm01 VALID (1.3446)"),
    "macrobossesiitp":  ("⏭️ SKIPPED",    "NEW form-only entry; repo unreachable even with judge token — cannot verify"),
    "MLforEDA":         ("❌ INELIGIBLE", "no place() entry (GNN/RL on ISPD-2005)"),
    "Wire We Even Here":("❌ INELIGIBLE", "py3.12-only f-string syntax in a py3.11 Dockerfile -> SyntaxError"),
}

# ── verdicts for the previously-unverified cohort (from the swag report) ──────
report_rows = {}
for line in report.splitlines():
    m = re.match(r'\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(✅|❌|⏭️)\s*(ELIGIBLE|FAIL|SKIPPED)\s*\|\s*([^|]*)\|', line)
    if m:
        report_rows[m.group(2).strip()] = (m.group(4), m.group(5).strip())

# ── all leaderboard rows (verified flag = the judge checkmark anywhere on the row) ─
teams = []
for line in readme.splitlines():
    if not re.match(r'\|\s*(\d+|DQ)\s*\|\s*"', line):
        continue
    cells = [c.strip() for c in line.split('|')[1:-1]]
    if len(cells) < 8:
        continue
    rank, name, proxy = cells[0], cells[1].strip('"').strip(), cells[2]
    verified = ':white_check_mark:' in line
    notes = cells[-1]
    teams.append((rank, name, proxy, verified, notes))

rows = []; kinds = []
for rank, name, proxy, verified, notes in teams:
    if rank == "DQ":
        cat, basis, kind = "❌ INELIGIBLE", "disqualified — " + (notes[:80] or "DQ"), "dq"
    elif name in report_rows:
        word, detail = report_rows[name]
        if word == "ELIGIBLE":  cat, basis, kind = "✅ ELIGIBLE", "swag-checked (ibm01): " + detail, "swag"
        elif word == "SKIPPED": cat, basis, kind = "⏭️ SKIPPED", detail, "skip"
        else:                   cat, basis, kind = "❌ INELIGIBLE", "swag-checked (ibm01): " + detail, "fail"
    elif verified:
        cat, basis, kind = "✅ ELIGIBLE", f"verified — full judge run (avg proxy {proxy.strip('*')})", "verified"
    else:
        cat, basis, kind = "❓ UNCLASSIFIED", "no verified flag and not in swag report", "other"
    if name in OVERRIDES:
        cat, basis = OVERRIDES[name]
        kind = "recovered" if cat.startswith("✅") else ("skip" if cat.startswith("⏭️") else "fail")
    if name in RERUN:
        basis += "  ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover)"
    rows.append((rank, name, cat, basis)); kinds.append(kind)
from collections import Counter
n = Counter(kinds)

def rk(r):
    try: return (0, int(r[0]))
    except: return (1, 0)
rows.sort(key=rk)

elig = n["verified"] + n["swag"] + n["recovered"]
total = len(teams)
o = []
def W(s=""): o.append(s)
W("# Macro-Placement Challenge — Full Swag-Eligibility Report\n")
if DATE: W(f"_Generated {DATE}. Leaderboard standings as of ref `{LB_REF}` (Archgen re-instated at #1)._\n")
W("**Swag bar:** the submission demonstrates minimum functionality — it runs end-to-end and "
  "produces a 0-overlap (VALID) placement, judged by the TILOS MacroPlacement scorer. Teams "
  "already confirmed by a full judge run qualify automatically.\n")
W("## Summary\n")
W(f"- **Total leaderboard teams:** {total}")
W(f"- **Swag-eligible: {elig}**  =  {n['verified']} verified (auto-qualify)  +  {n['swag']} swag-verified (cohort)  +  {n['recovered']} recovered / newly-verified this pass")
W(f"- **Ineligible: {n['fail'] + n['dq']}**  =  {n['fail']} failed the swag check  +  {n['dq']} disqualified")
W(f"- **Skipped (could not be run): {n['skip']}**  (Modal-only / private / unreachable repo)")
if n["other"]:
    W(f"- **Unclassified: {n['other']}** (needs a manual look)")
W()
W("### How eligibility was determined\n")
W("1. **Already-verified teams** cleared a full multi-benchmark judge run and qualify "
  "automatically — not re-run by the swag harness.")
W("2. **Previously-unverified teams** were each run **air-gapped** on `ibm01`, clamped to the "
  "eval envelope (1 GPU / 16 cores / ≤40 GB), and scored by the authoritative scorer. Detail: "
  "`swag_verify/REPORT.md` and `swag_verify/TRIAGE_NOTES.md`.")
W("3. **Disqualified** entries (failed judge run / superseded / won't run) are excluded.\n")
W("### Caveats\n")
W("- **Recovered / newly-verified this pass.** Re-running teams in their declared environment "
  "recovered **Dragonfly** & **BakaBobo** (built their Dockerfile / overlaid the module they were "
  "wrongly failed for), plus 4 form-only submissions that were never previously ranked "
  "(**Macropolis, Nikunj Bhatt, AxeCap, A-cat-suki**) — all ibm01 VALID.")
W("- **Still pending.** Teams that ship a Dockerfile or hit challenge-API drift but aren't yet "
  "cleanly reproduced are flagged **⟳ PENDING RE-RUN** — a faithful re-run may move some to ✅: "
  + ", ".join(sorted(RERUN)) + ". **macrobossesiitp** is unverifiable (repo unreachable).")
W("- **A PASS-gate bug was fixed.** The gate matched `VALID` inside `INVALID`; three submissions "
  "the scorer marked INVALID (ilovekiro, Binghamton, SnoobQuants) had been mis-scored eligible "
  "and are now correctly ❌.")
W("- **`ibm01` only.** Single-benchmark minimum-functionality check, not the 17-benchmark ranking.\n")
W("## All teams\n")
W("| Rank | Team | Status | Basis |")
W("|---|---|---|---|")
for rank, name, cat, basis in rows:
    W(f"| {rank} | {name} | {cat} | {basis.replace('|', '/')} |")
W()

(ROOT / "swag_verify" / "FULL_REPORT.md").write_text("\n".join(o))
print(f"eligible={elig} (verified {n['verified']} + swag {n['swag']} + recovered {n['recovered']}) | "
      f"ineligible={n['fail']+n['dq']} (fail {n['fail']} + dq {n['dq']}) | "
      f"skip={n['skip']} | other={n['other']} | total={total}")
