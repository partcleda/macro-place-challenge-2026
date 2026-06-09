#!/usr/bin/env python3
"""Detect a submission's entry-point placer file inside a cloned repo.

Prints JSON: {"entry": <relpath|null>, "confidence": "...", "candidates": [...]}.
The entry is the file holding the team's own placer class (a class with a
`place(self, benchmark)` method) — NOT the challenge's bundled example placers.
"""
import hashlib, json, re, sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hint = sys.argv[2] if len(sys.argv) > 2 else ""

# Canonical challenge-bundled "decoy" placers that every fork ships. A clone
# file byte-identical to one of these is NOT the team's work, so exclude it.
# (Teams who *modified* will_seed as their entry, e.g. V5, differ in hash and
# are kept.)
CHALLENGE_ROOT = Path(__file__).resolve().parent.parent
def _sha1(p):
    try: return hashlib.sha1(p.read_bytes()).hexdigest()
    except Exception: return None
DECOY_HASHES = set()
for _d in ["macro_place", "scripts", "test", "submissions", "src"]:
    base = CHALLENGE_ROOT / _d
    if base.exists():
        for _p in base.rglob("*.py"):
            if "swag_verify" in _p.parts or "clones" in _p.parts:
                continue
            h = _sha1(_p)
            if h: DECOY_HASHES.add(h)
for _p in CHALLENGE_ROOT.glob("*.py"):
    h = _sha1(_p)
    if h: DECOY_HASHES.add(h)

EXCLUDE_DIR = re.compile(
    r'(^|/)(\.git|external|__pycache__|node_modules|\.venv|venv|env|'
    r'examples|tests?|Plc_client|build|dist|\.pytest_cache|site-packages|'
    r'third_party|thirdparty)(/|$)', re.I)
# helper modules that define place() but aren't the entry
HELPER = {"legalize.py","config.py","utils.py","data.py","state.py",
          "objective.py","benchmark.py","loader.py","evaluate.py",
          "global_placer.py","def_writer.py","_plc.py","batched.py","pbsa.py"}

PLACE_RE = re.compile(r'def\s+place\s*\(\s*self', re.M)
CLASS_RE = re.compile(r'^\s*class\s+\w', re.M)
CLASSPLACER_RE = re.compile(r'class\s+\w*[Pp]lac\w*', re.M)

def has_placer(p: Path) -> bool:
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return bool(PLACE_RE.search(t) and CLASS_RE.search(t))

cands = []
for p in repo.rglob("*.py"):
    rel = p.relative_to(repo).as_posix()
    if EXCLUDE_DIR.search("/" + rel):
        continue
    # skip the challenge's own scorer package at repo root
    if rel.startswith("macro_place/"):
        continue
    # skip byte-identical copies of challenge-bundled decoy placers
    if _sha1(p) in DECOY_HASHES:
        continue
    if has_placer(p):
        cands.append(rel)

hint = hint.strip().strip("/")
hint_is_py = hint.endswith(".py")
hint_dir = "" if hint_is_py else hint

def score(rel: str) -> int:
    s = 0
    base = rel.rsplit("/", 1)[-1]
    txt_classplacer = False
    try:
        txt_classplacer = bool(CLASSPLACER_RE.search((repo / rel).read_text(errors="ignore")))
    except Exception:
        pass
    if hint_is_py and rel == hint: s += 1000
    if hint_dir and (rel == hint_dir or rel.startswith(hint_dir + "/")): s += 60
    if base == "placer.py": s += 35
    elif "placer" in base.lower(): s += 20
    if rel.startswith("submissions/"): s += 12
    if txt_classplacer: s += 12
    if base in HELPER: s -= 40
    s -= rel.count("/") * 2  # prefer shallower
    return s

# direct hint file even if no place() detected (place may be dynamic)
if hint_is_py and (repo / hint).is_file() and hint not in cands:
    cands.append(hint)

result = {"entry": None, "confidence": "none", "candidates": sorted(cands)}
if cands:
    ranked = sorted(cands, key=score, reverse=True)
    best = ranked[0]
    result["entry"] = best
    if hint_is_py and best == hint:
        result["confidence"] = "high"
    elif len(ranked) == 1:
        result["confidence"] = "high"
    elif hint_dir and best.startswith(hint_dir):
        result["confidence"] = "med"
    elif score(best) - (score(ranked[1]) if len(ranked) > 1 else -999) >= 25:
        result["confidence"] = "med"
    else:
        result["confidence"] = "low"

print(json.dumps(result))
