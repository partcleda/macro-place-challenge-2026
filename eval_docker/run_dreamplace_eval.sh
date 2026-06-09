#!/bin/bash
# Wrapper for DREAMPlace submissions: copies compiled .so extensions into
# the submission's bundled dreamplace directory before running evaluation.
#
# Usage (inside container):
#   /challenge/run_dreamplace_eval.sh submissions/placer.py --all

set -e

PLACER_PATH="$1"
shift

# If the submission has a bundled dreamplace/ directory, copy compiled .so
# files from the installed DREAMPlace into it
SUBMISSION_DIR=$(dirname "$PLACER_PATH")
if [ -d "$SUBMISSION_DIR/dreamplace" ] && { [ -d "/opt/DREAMPlace/install/dreamplace" ] || [ -d "/opt/DREAMPlace/build/dreamplace" ]; }; then
    # Copy to /challenge/dp_submission/ (writable) so __file__/../external works
    echo "[dreamplace-eval] Copying submission to writable location..."
    cp -a "$SUBMISSION_DIR" /challenge/dp_submission

    echo "[dreamplace-eval] Injecting compiled .so extensions..."
    # Copy from both install/ and build/ directories
    for search_root in /opt/DREAMPlace/install/dreamplace /opt/DREAMPlace/build/dreamplace; do
        [ -d "$search_root" ] || continue
        find "$search_root" -name "*.so" | while read so_file; do
            rel_path="${so_file#${search_root}/}"
            target_dir="/challenge/dp_submission/dreamplace/$(dirname "$rel_path")"
            if [ -d "$target_dir" ]; then
                cp -n "$so_file" "$target_dir/" 2>/dev/null || true
            fi
        done
    done

    # Create passthrough stubs for any missing CUDA modules (CUB compat issue with CUDA 12.4)
    echo "[dreamplace-eval] Creating stubs for missing CUDA modules..."
    python3 -c "
import os, re, glob
root = '/challenge/dp_submission/dreamplace'
for pyfile in glob.glob(root + '/ops/**/*.py', recursive=True):
    with open(pyfile) as f:
        for line in f:
            m = re.match(r'\s*import (dreamplace\.ops\.\S+)', line)
            if not m: continue
            mod = m.group(1)
            rel = mod.replace('dreamplace.', '').replace('.', '/')
            so_glob = os.path.join(root, rel + '*.so')
            py_path = os.path.join(root, rel + '.py')
            if not glob.glob(so_glob) and not os.path.exists(py_path):
                os.makedirs(os.path.dirname(py_path), exist_ok=True)
                with open(py_path, 'w') as sf:
                    sf.write('# stub for missing CUDA module\n')
                    sf.write('def forward(*a, **kw): raise RuntimeError(\"stub: not compiled\")\n')
                    sf.write('def backward(*a, **kw): raise RuntimeError(\"stub: not compiled\")\n')
                print(f'  stub: {rel}')
"

    PLACER_FILE=$(basename "$PLACER_PATH")
    exec python -m macro_place.evaluate "dp_submission/$PLACER_FILE" "$@"
else
    exec python -m macro_place.evaluate "$PLACER_PATH" "$@"
fi
