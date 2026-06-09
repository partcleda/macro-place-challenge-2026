#!/usr/bin/env python3
"""Insert a 'Swag' column into the README leaderboard table.

Swag-eligible = verified teams (already proven functional) OR teams my
minimum-functionality verification passed (results.tsv / triage_results.tsv).
Ranked-but-not-eligible -> ❌. Baselines / DQ / pending rows -> blank.
Operates in place on README.md (run on the PR branch).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

def slug(team):
    t = team.strip().strip('*').strip('"').strip()
    s = re.sub(r'[^a-z0-9]+', '_', t.lower()).strip('_')
    return s or "team"

PASS = set()
for f in [ROOT/"swag_verify"/"results.tsv", ROOT/"swag_verify"/"triage_results.tsv"]:
    if f.exists():
        for line in f.read_text().splitlines():
            p = line.split("\t")
            if len(p) > 1 and p[1] == "PASS":
                PASS.add(p[0])

lines = README.read_text().splitlines()
VERIFIED_IDX = 7   # 0-based cell index of "Verified" in the leaderboard row
out, in_lb, changed = [], False, 0

def cells_of(line):
    # split a markdown table row into inner cells (drop leading/trailing empties)
    parts = line.split("|")
    return parts[1:-1] if len(parts) >= 2 else parts

for line in lines:
    if line.startswith("| Rank | Team | Avg Proxy Cost |"):
        in_lb = True
        c = cells_of(line); c.insert(VERIFIED_IDX+1, " Swag ")
        out.append("|" + "|".join(c) + "|"); continue
    if in_lb and re.match(r'^\|[-\s|]+\|$', line):  # separator row
        c = cells_of(line); c.insert(VERIFIED_IDX+1, "------")
        out.append("|" + "|".join(c) + "|"); continue
    if in_lb and line.startswith("|"):
        c = cells_of(line)
        if len(c) <= VERIFIED_IDX:
            out.append(line); continue
        rank = c[0].strip()
        team = c[1].strip()
        verified = ":white_check_mark:" in c[VERIFIED_IDX]
        if re.match(r'^\d+$', rank):                       # a ranked team
            mark = " ✅ " if (verified or slug(team) in PASS) else " ❌ "
        else:                                              # baseline / DQ / pending
            mark = "  "
        c.insert(VERIFIED_IDX+1, mark)
        out.append("|" + "|".join(c) + "|"); changed += 1
        continue
    if in_lb and not line.startswith("|"):
        in_lb = False
    out.append(line)

README.write_text("\n".join(out) + "\n")
n_yes = sum(1 for l in out if l.startswith("|") and "✅" in l and "Rank" not in l)
n_no  = sum(1 for l in out if l.startswith("|") and "❌" in l)
print(f"rows updated: {changed}  swag ✅: {n_yes}  ❌: {n_no}  (PASS slugs loaded: {len(PASS)})")
