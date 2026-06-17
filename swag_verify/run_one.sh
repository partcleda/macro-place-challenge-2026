#!/usr/bin/env bash
# Verify one submission for minimum functionality (ibm01 -> valid placement).
# Usage: run_one.sh <slug> <clone_url> <branch> <subpath> <gpu_id>
# Emits one TSV line on stdout: slug \t verdict \t detail \t entry \t confidence
# Verdicts: PASS | INVALID | NO_ENTRY | MISSING_DEP | CRASH | TIMEOUT | CLONE_FAIL
set -uo pipefail

SLUG="$1"; URL="$2"; BRANCH="${3:-}"; SUBPATH="${4:-}"; GPU="${5:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLONES="$ROOT/swag_verify/clones"
LOGS="$ROOT/swag_verify/logs"
DIR="$CLONES/$SLUG"
LOG="$LOGS/$SLUG.log"
TIMEOUT="${SWAG_TIMEOUT:-1800}"   # 30 min default
emit(){ printf '%s\t%s\t%s\t%s\t%s\n' "$SLUG" "$1" "$2" "${3:-}" "${4:-}"; }

# ── clone ─────────────────────────────────────────────────────────────────────
if [ ! -d "$DIR/.git" ]; then
  rm -rf "$DIR"
  if [ -n "$BRANCH" ]; then
    git clone --depth 1 --branch "$BRANCH" "$URL" "$DIR" >"$LOG.clone" 2>&1 \
      || git clone --depth 1 "$URL" "$DIR" >>"$LOG.clone" 2>&1
  else
    git clone --depth 1 "$URL" "$DIR" >"$LOG.clone" 2>&1
  fi
  if [ ! -d "$DIR/.git" ]; then
    emit CLONE_FAIL "$(tail -1 "$LOG.clone" 2>/dev/null | tr '\t' ' ' | cut -c1-120)"
    exit 0
  fi
fi

# ── detect entry point ──────────────────────────────────────────────────────
DET="$(python3 "$ROOT/swag_verify/detect_entry.py" "$DIR" "$SUBPATH" 2>/dev/null)"
ENTRY="$(printf '%s' "$DET" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("entry") or "")')"
CONF="$(printf '%s' "$DET" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("confidence") or "")')"
if [ -z "$ENTRY" ]; then
  emit NO_ENTRY "no class with place(self,benchmark) found" "" "$CONF"
  exit 0
fi

# ── run ibm01 in air-gapped, resource-clamped container ─────────────────────
# Merged macro_place: challenge scorer files are authoritative (teams can't
# override objective/validate), but extra modules a team added under the
# macro_place.* namespace (fast_proxy, routing_surrogate, ...) are overlaid in
# so the placer's imports resolve. submissions.* resolve from /submission.
CNAME="swag_${SLUG}_$$"
# Policy: the challenge's macro_place is authoritative (scorer integrity + its
# correct _plc/external wiring). CWD=/challenge (WORKDIR) puts /challenge/macro_place
# first on sys.path for `-m`. We overlay ONLY the team's EXTRA macro_place.* modules
# (helpers they added that don't exist in the challenge package, e.g. fast_proxy,
# routing_surrogate) into /challenge/macro_place so their placer's imports resolve,
# WITHOUT overwriting challenge scorer/_plc files. submissions.* resolve via
# PYTHONPATH=/submission. (Rare strict self-containment audits are handled in triage.)
if [ "${SWAG_SELFCONTAINED:-0}" = "1" ]; then
  # Self-contained mode: the team's bundled macro_place is authoritative (their
  # Benchmark/API + helpers). PYTHONSAFEPATH drops CWD so /submission wins; the
  # challenge Plc_client is on path so plc_client_os imports regardless.
  PYP="/submission:/challenge/external/MacroPlacement/CodeElements/Plc_client"
  RUNENV=(-e PYTHONSAFEPATH=1 -e PYTHONPATH="$PYP" -e ENTRY="$ENTRY")
  RUNCMD='exec python -m macro_place.evaluate "/submission/$ENTRY" -b ibm01'
else
  # Default (approach A): challenge macro_place authoritative; overlay team extras.
  RUNENV=(-e PYTHONPATH=/submission -e ENTRY="$ENTRY")
  RUNCMD='set -e
if [ -d /submission/macro_place ]; then
  for it in /submission/macro_place/* ; do b=$(basename "$it");
    [ -e "/challenge/macro_place/$b" ] || cp -a "$it" /challenge/macro_place/ ; done
fi
exec python -m macro_place.evaluate "/submission/$ENTRY" -b ibm01'
fi
timeout --signal=KILL "$TIMEOUT" docker run --rm --name "$CNAME" \
  --network none --gpus "\"device=$GPU\"" --cpus 16 --memory 28g \
  -e OMP_NUM_THREADS=16 -e MKL_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 \
  "${RUNENV[@]}" \
  -v "$DIR:/submission:rw" \
  --entrypoint bash macro-place-eval -c "$RUNCMD" >"$LOG" 2>&1
RC=$?
docker rm -f "$CNAME" >/dev/null 2>&1 || true

# ── parse verdict ───────────────────────────────────────────────────────────
if [ $RC -eq 137 ] || [ $RC -eq 124 ]; then
  emit TIMEOUT "exceeded ${TIMEOUT}s" "$ENTRY" "$CONF"; exit 0
fi
# NOTE: 'INVALID' contains the substring 'VALID', so the scorer's INVALID
# verdict MUST be matched first — otherwise 'proxy=N ... INVALID (k overlaps)'
# false-matches a 'VALID' grep and an overlapping placement is scored PASS.
# Match the scorer's literal verdict token only (not the word 'overlaps', which
# also appears in many valid runs' progress output, e.g. 'overlaps=0').
if grep -qE 'INVALID|DISQUALIFIED' "$LOG" 2>/dev/null; then
  OV="$(grep -oE 'INVALID \([0-9]+ overlaps\)' "$LOG" | head -1)"
  emit INVALID "${OV:-overlaps present}" "$ENTRY" "$CONF"; exit 0
fi
# Genuine VALID: a 'VALID' not preceded by 'N' (so it can't be inside INVALID).
if grep -qE '(^|[^N])VALID' "$LOG" 2>/dev/null; then
  PROXY="$(grep -oE 'proxy=[0-9.]+' "$LOG" | head -1 | cut -d= -f2)"
  emit PASS "ibm01 valid proxy=${PROXY}" "$ENTRY" "$CONF"; exit 0
fi
if grep -qE 'ModuleNotFoundError|ImportError|No module named' "$LOG" 2>/dev/null; then
  MOD="$(grep -oE "No module named '[^']+'" "$LOG" | head -1)"
  emit MISSING_DEP "${MOD:-import error}" "$ENTRY" "$CONF"; exit 0
fi
if grep -qE 'No placer class found' "$LOG" 2>/dev/null; then
  emit NO_ENTRY "evaluate: no placer class in $ENTRY" "$ENTRY" "$CONF"; exit 0
fi
# generic failure: capture last exception line
ERR="$(grep -E 'Error|Exception|Traceback|assert' "$LOG" 2>/dev/null | tail -1 | tr '\t' ' ' | cut -c1-120)"
emit CRASH "rc=$RC ${ERR}" "$ENTRY" "$CONF"
