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
LB_REF = sys.argv[2] if len(sys.argv) > 2 else "add-shoom-rescore"
readme = subprocess.run(["git", "show", f"{LB_REF}:README.md"], capture_output=True, text=True).stdout
report = (ROOT / "swag_verify" / "REPORT.md").read_text()

# Teams pending a faithful Dockerfile / API re-run (currently ❌ but may recover).
RERUN = {"Dragonfly", "ICAS_placer", "RuslanPlace", "Hoop Dreams", "ilovekiro", "Combobulating"}

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

rows = []
n = {"verified": 0, "swag_pass": 0, "fail": 0, "skip": 0, "dq": 0, "other": 0}
for rank, name, proxy, verified, notes in teams:
    if rank == "DQ":
        cat, basis = "❌ INELIGIBLE", "disqualified — " + (notes[:80] or "DQ")
        n["dq"] += 1
    elif name in report_rows:
        word, detail = report_rows[name]
        if word == "ELIGIBLE":
            cat, basis = "✅ ELIGIBLE", "swag-checked (ibm01): " + detail; n["swag_pass"] += 1
        elif word == "SKIPPED":
            cat, basis = "⏭️ SKIPPED", detail; n["skip"] += 1
        else:
            cat, basis = "❌ INELIGIBLE", "swag-checked (ibm01): " + detail; n["fail"] += 1
    elif verified:
        cat, basis = "✅ ELIGIBLE", f"verified — full judge run (avg proxy {proxy.strip('*')})"; n["verified"] += 1
    else:
        cat, basis = "❓ UNCLASSIFIED", "no verified flag and not in swag report"; n["other"] += 1
    if name in RERUN:
        basis += "  ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover)"
    rows.append((rank, name, cat, basis))

def rk(r):
    try: return (0, int(r[0]))
    except: return (1, 0)
rows.sort(key=rk)

elig = n["verified"] + n["swag_pass"]
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
W(f"- **Swag-eligible: {elig}**  =  {n['verified']} already-verified (auto-qualify)  +  {n['swag_pass']} newly swag-verified")
W(f"- **Ineligible: {n['fail'] + n['dq']}**  =  {n['fail']} failed the swag check  +  {n['dq']} disqualified")
W(f"- **Skipped (could not be run): {n['skip']}**  (Modal-only / private / no resolvable repo)")
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
W("- **Conservative by construction.** A team is ✅ only if it *demonstrated* a valid placement. "
  "Teams whose declared environment (a shipped `Dockerfile`) was not built, or that hit a "
  "challenge-API change, are currently ❌ but flagged **⟳ PENDING RE-RUN** — a faithful re-run "
  "may move some back to ✅: " + ", ".join(sorted(RERUN)) + ".")
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
print(f"eligible={elig} (verified {n['verified']} + swag {n['swag_pass']}) | "
      f"ineligible={n['fail']+n['dq']} (fail {n['fail']} + dq {n['dq']}) | "
      f"skip={n['skip']} | other={n['other']} | total={total}")
