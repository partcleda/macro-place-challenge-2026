#!/usr/bin/env bash
# Run a DREAMPlace-bundled submission in the macro-place-eval-dreamplace image:
# inject compiled .so into every bundled dreamplace/ dir (any nesting), expose a
# global compiled dreamplace fallback, then run ibm01. Emits a verdict line.
# Usage: dreamplace_run.sh <slug> <entry_relpath> <gpu_id>
set -uo pipefail
SLUG="$1"; ENTRY="$2"; GPU="${3:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/swag_verify/clones/$SLUG"
LOG="$ROOT/swag_verify/logs/${SLUG}_dp.log"
TIMEOUT="${SWAG_TIMEOUT:-3300}"

WRAP='set -e
for SUB_DP in $(find /submission -type d -name dreamplace 2>/dev/null); do
  (cd /opt/DREAMPlace/build/dreamplace && find . -name "*.so" | while read so; do
     tgt="$SUB_DP/$(dirname "$so")"; [ -d "$tgt" ] && cp -n "/opt/DREAMPlace/build/dreamplace/$so" "$tgt/" 2>/dev/null || true; done)
done
if [ -d /submission/macro_place ]; then
  for it in /submission/macro_place/* ; do b=$(basename "$it"); [ -e "/challenge/macro_place/$b" ] || cp -a "$it" /challenge/macro_place/ ; done
fi
export PYTHONPATH=/submission:/opt/DREAMPlace/build
exec python -m macro_place.evaluate "/submission/$ENTRY" -b ibm01'

CNAME="swag_${SLUG}_dp_$$"
timeout --signal=KILL "$TIMEOUT" docker run --rm --name "$CNAME" \
  --network none --gpus "\"device=$GPU\"" --cpus 16 --memory 40g \
  -e ENTRY="$ENTRY" -v "$DIR:/submission:rw" \
  --entrypoint bash macro-place-eval-dreamplace -c "$WRAP" >"$LOG" 2>&1
RC=$?
docker rm -f "$CNAME" >/dev/null 2>&1 || true

if grep -qE 'proxy=[0-9].*VALID' "$LOG" && ! grep -qE 'INVALID' "$LOG"; then
  P=$(grep -oE 'proxy=[0-9.]+' "$LOG" | head -1)
  printf '%s\tPASS\tibm01 valid %s (dreamplace img)\t%s\tdreamplace\n' "$SLUG" "$P" "$ENTRY"
elif [ $RC -eq 137 ] || [ $RC -eq 124 ]; then
  printf '%s\tTIMEOUT\tdreamplace img, exceeded %ss\t%s\tdreamplace\n' "$SLUG" "$TIMEOUT" "$ENTRY"
else
  ERR=$(grep -E 'Error|Exception|No module|INVALID' "$LOG" | tail -1 | tr '\t' ' ' | cut -c1-90)
  printf '%s\tFAIL_DP\trc=%s %s\t%s\tdreamplace\n' "$SLUG" "$RC" "$ERR" "$ENTRY"
fi
