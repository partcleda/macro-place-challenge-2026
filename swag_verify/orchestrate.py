#!/usr/bin/env python3
"""Run minimum-functionality verification across the work-list.

4 worker slots, each pinned to a distinct physical GPU (0-3) so no submission
ever shares a GPU — each sees exactly the eval envelope (1 GPU / 16 cores).
Resumable: teams already in results.tsv are skipped.
"""
import csv, os, re, subprocess, sys, threading, queue, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WL = ROOT / "swag_verify" / "worklist.tsv"
RES = ROOT / "swag_verify" / "results.tsv"
RUN = str(ROOT / "swag_verify" / "run_one.sh")
NGPU = 4            # physical GPUs (0..3)
CONCURRENCY = 12   # workers; each pinned to gpu = worker_index % NGPU (3 small ibm01 jobs/GPU)

def slug(team):
    s = re.sub(r'[^a-z0-9]+', '_', team.lower()).strip('_')
    return s or "team"

# parse args
only = None
timeout = "1800"
outpath = None
for i, a in enumerate(sys.argv[1:]):
    if a == "--only": only = set(sys.argv[i+2].split(","))
    if a == "--timeout": timeout = sys.argv[i+2]
    if a == "--out": outpath = sys.argv[i+2]
RESOUT = Path(outpath) if outpath else RES

rows = list(csv.DictReader(WL.open(), delimiter="\t"))
todo = [r for r in rows if r["decision"] == "run"]
# unique slug
seen = {}
for r in todo:
    s = slug(r["team"]);
    while s in seen: s += "_x"
    seen[s] = r; r["_slug"] = s
if only:
    todo = [r for r in todo if r["_slug"] in only or r["team"] in only]

done = set()
if RESOUT.exists():
    for line in RESOUT.read_text().splitlines():
        if line.strip(): done.add(line.split("\t")[0])
todo = [r for r in todo if r["_slug"] not in done]

print(f"to run: {len(todo)}  (already done: {len(done)})  concurrency: {CONCURRENCY} over {NGPU} GPUs")
lock = threading.Lock()
q = queue.Queue()
for r in todo: q.put(r)
counter = {"n": 0}; total = len(todo)

def worker(gpu):
    while True:
        try: r = q.get_nowait()
        except queue.Empty: return
        s = r["_slug"]
        t0 = time.time()
        try:
            out = subprocess.run(
                [RUN, s, r["clone_url"], r["branch"], r["subpath"], str(gpu)],
                capture_output=True, text=True, timeout=int(timeout)+300,
                env={**os.environ, "SWAG_TIMEOUT": timeout},
            )
            line = (out.stdout.strip().splitlines() or ["%s\tNO_OUTPUT\t " % s])[-1]
        except subprocess.TimeoutExpired:
            line = f"{s}\tTIMEOUT\torchestrator killed\t\t"
        except Exception as e:
            line = f"{s}\tHARNESS_ERR\t{e}\t\t"
        dt = time.time() - t0
        with lock:
            counter["n"] += 1
            with RESOUT.open("a") as f: f.write(line + f"\t{dt:.0f}s\n")
            verdict = line.split("\t")[1] if "\t" in line else "?"
            print(f"[{counter['n']}/{total}] gpu{gpu} {dt:4.0f}s  {verdict:12s} {s}", flush=True)
        q.task_done()

ts = [threading.Thread(target=worker, args=(i % NGPU,)) for i in range(CONCURRENCY)]
for t in ts: t.start()
for t in ts: t.join()
print("DONE")
