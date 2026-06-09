#!/usr/bin/env python3
"""Build the swag-verification work-list.

Joins the Google-form export (repo URLs + per-submission status) against the
README leaderboard (canonical team list + verified flag) to produce the set of
*unverified* teams to run, each with its latest active repo URL.
"""
import re, csv, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORM = ROOT / "swag_verify" / "form_export.txt"
README = ROOT / "README.md"

def norm(name: str) -> str:
    """Normalize a team name for joining across sources."""
    n = name.strip().strip('"').strip().lower()
    n = re.sub(r"\s+", " ", n)
    n = n.replace("’", "'")
    return n

# Manual repo URLs for leaderboard teams whose form team-name collides with
# another entry (the leaderboard disambiguated several "UT Austin" / per-person
# rows with suffixes that don't exist in the raw form).
OVERRIDES = {
    "makercode": "https://github.com/Weiyet/macro-place-challenge-2026/tree/main/v4_solution",
    "ut austin - as": "https://github.com/A14N77/macro-place-challenge-2026",
    "ut austin - ct": "https://github.com/themoddedcube/autoresearch-macro-place-challenge-2026",
    # UT Austin - RH (Richard Huang) gave only a profile link, no concrete repo.
    "ut austin - rh": "",
}

def decide(team_norm, url, status, notes):
    """Return (decision, reason). decision in {run, skip}."""
    low_notes = (notes or "").lower()
    if not url:
        return "skip", "no resolvable repo URL (profile link only / private)"
    if "private repo" in low_notes or status == "private":
        return "skip", "private repo — no judge access"
    if "modal" in low_notes or "air-gap" in low_notes:
        return "skip", "Modal cloud dispatch — cannot run air-gapped"
    if "docker-in-docker" in low_notes:
        return "skip", "Docker-in-Docker — won't run inside eval container"
    return "run", ""

# ── parse form export ────────────────────────────────────────────────────────
raw = FORM.read_text(encoding="utf-8", errors="replace")
records = [r for r in re.split(r'(?m)^(?=\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2})', raw) if r.strip()]

URL_RE = re.compile(r'(?:https?://)?github\.com/[^\s,"\'<>)]+', re.I)
STATUS_MARKERS = [
    ("Verified - Approved", "verified"),
    ("verification in progress", "in_progress"),
    ("needs to resend", "resend"),
    ("nèeds to resend", "resend"),
    ("reported worse score", "new_worse"),
    ("old version", "old"),
    ("Not verified", "not_verified"),
    ("Private repo", "private"),
    ("Pending", "pending"),
]

def parse_ts(s):
    try:
        return datetime.strptime(s.strip(), "%m/%d/%Y %H:%M:%S")
    except Exception:
        return datetime.min

def detect_status(rec: str) -> str:
    low = rec.lower()
    for marker, tag in STATUS_MARKERS:
        if marker.lower() in low:
            return tag
    # DQ appears as a standalone trailing field
    if re.search(r'(?:^|\s)DQ(?:\s|$)', rec):
        return "dq"
    return "blank"

def clean_url(u: str) -> str:
    u = u.rstrip('.,;)')
    if not u.startswith("http"):
        u = "https://" + u
    return u

subs = []  # (team_display, team_norm, ts, status, url, score, email)
for rec in records:
    first = rec.split("\n", 1)[0]
    fields = re.split(r'\s{3,}', first)
    if len(fields) < 3:
        continue
    ts = parse_ts(fields[0])
    email = fields[1].strip() if len(fields) > 1 else ""
    team = fields[2].strip() if len(fields) > 2 else ""
    if not team or "@" in team:
        continue
    urls = URL_RE.findall(rec)
    url = clean_url(urls[0]) if urls else ""
    status = detect_status(rec)
    # self-score: first standalone float in 0.5..100 range appearing after the url line (best-effort)
    score = ""
    m = re.search(r'\b(0\.\d{2,}|1\.\d{2,}|2\.\d{2,}|3\.\d{2,})\b', rec)
    if m:
        score = m.group(1)
    subs.append((team, norm(team), ts, status, url, score, email))

# latest active submission per team (prefer non-old/non-blank; else latest)
by_team = {}
for s in subs:
    tn = s[1]
    by_team.setdefault(tn, []).append(s)

latest = {}
for tn, lst in by_team.items():
    lst.sort(key=lambda x: x[2])  # by ts asc
    # prefer the most recent that has a url
    chosen = None
    for s in reversed(lst):
        if s[4]:
            chosen = s; break
    chosen = chosen or lst[-1]
    latest[tn] = chosen

# ── parse README leaderboard ─────────────────────────────────────────────────
lb_lines = [l for l in README.read_text(encoding="utf-8").splitlines()
            if re.match(r'^\| [0-9]+ \|', l)]
lb = []  # (rank, team_display, team_norm, verified, notes)
for l in lb_lines:
    cells = [c.strip() for c in l.split("|")[1:-1]]
    rank = cells[0]
    team = cells[1].strip().strip('*').strip('"')
    verified = ":white_check_mark:" in l
    notes = cells[-1] if cells else ""
    lb.append((rank, team, norm(team), verified, notes))

unverified = [r for r in lb if not r[3]]

def split_url(u: str):
    """Return (clone_url, branch, subpath) from a github URL that may contain
    /tree/<branch>/<path> or /blob/<branch>/<path>."""
    if not u:
        return "", "", ""
    m = re.match(r'(https://github\.com/[^/]+/[^/]+?)(?:\.git)?(?:/(?:tree|blob)/([^/]+)(?:/(.*))?)?$', u)
    if not m:
        return u, "", ""
    return m.group(1), (m.group(2) or ""), (m.group(3) or "")

# ── join ──────────────────────────────────────────────────────────────────────
out = ROOT / "swag_verify" / "worklist.tsv"
matched = unmatched = run_n = skip_n = 0
with out.open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["rank","team","decision","reason","clone_url","branch","subpath",
                "status_form","self_score","email","notes_leaderboard"])
    for rank, team, tn, verified, notes in unverified:
        rec = latest.get(tn)
        url = OVERRIDES.get(tn, rec[4] if rec else "")
        status = rec[3] if rec else "NO_FORM_MATCH"
        score = rec[5] if rec else ""
        email = rec[6] if rec else ""
        if tn in OVERRIDES or rec:
            matched += 1
        else:
            unmatched += 1
        clone, branch, sub = split_url(url)
        decision, reason = decide(tn, url, status, notes)
        run_n += decision == "run"; skip_n += decision == "skip"
        w.writerow([rank, team, decision, reason, clone, branch, sub, status, score, email, notes])

print(f"form submissions parsed: {len(subs)}  unique teams: {len(latest)}")
print(f"leaderboard ranked: {len(lb)}  verified: {sum(1 for r in lb if r[3])}  unverified: {len(unverified)}")
print(f"worklist: matched={matched}  unmatched={unmatched}  run={run_n}  skip={skip_n}  -> {out}")
