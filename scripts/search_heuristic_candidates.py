#!/usr/bin/env python3
"""Search heuristic-learning placement candidates with official proxy scoring.

This is an offline tuning helper. It deliberately uses private methods from the
heuristic-learning placer so candidate evidence maps directly to the shipped
submission implementation. The script writes JSONL rows as it scores, supports
resume, and never edits the submission itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from macro_place.evaluate import IBM_BENCHMARKS
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


REPO_ROOT = Path(__file__).resolve().parents[1]
PLACER_PATH = REPO_ROOT / "submissions" / "heuristic_learning" / "placer.py"
TESTCASE_ROOT = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"

# Last validated checkpoint scores. Keep this table in sync with README notes
# when integrating winners back into the runtime selector.
CHECKPOINT_SCORES = {
    "ibm01": 1.005413,
    "ibm02": 1.541660,
    "ibm03": 1.281197,
    "ibm04": 1.275956,
    "ibm06": 1.639377,
    "ibm07": 1.432165,
    "ibm08": 1.451514,
    "ibm09": 1.077796,
    "ibm10": 1.325471,
    "ibm11": 1.202300,
    "ibm12": 1.613654,
    "ibm13": 1.366300,
    "ibm14": 1.572000,
    "ibm15": 1.585671,
    "ibm16": 1.480755,
    "ibm17": 1.712500,
    "ibm18": 1.765780,
}

RAW_REPAIR_CLEARANCES = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 3.5e-2)
PAIR_PUSH_GAPS = (0.0, 1e-4, 1e-3, 1e-2, 3.5e-2)
PAIR_PUSH_DAMPING = (0.45, 0.65, 0.85)
PAIR_PUSH_PASSES = (500, 1200)
SOFT_STRENGTHS = (0.08, 0.12, 0.15, 0.18, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45)
SOFT_STEPS = (1, 2, 3, 5)


def _load_hl_module():
    spec = importlib.util.spec_from_file_location("heuristic_learning_placer", PLACER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {PLACER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _benchmarks_from_args(args):
    if args.all:
        return IBM_BENCHMARKS
    if args.benchmark:
        return args.benchmark
    return ["ibm06", "ibm14", "ibm15", "ibm17", "ibm18"]


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _candidate_id(benchmark: str, stage: str, label: str, params: dict[str, Any]) -> str:
    payload = json.dumps(_jsonable(params), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{benchmark}:{stage}:{label}:{digest}"


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if not path.exists():
        return rows
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"bad JSONL at {path}:{line_no}: {exc}") from exc
            candidate_id = row.get("candidate_id")
            if candidate_id:
                rows[candidate_id] = row
    return rows


def _append_row(handle, existing_rows: dict[str, dict[str, Any]], row: dict[str, Any]):
    row = _jsonable(row)
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    existing_rows[row["candidate_id"]] = row


def _full_from_hard(placer, benchmark, n_hard: int, hard_pos: np.ndarray):
    full = benchmark.macro_positions.clone()
    full[:n_hard] = torch.tensor(hard_pos, dtype=torch.float32)
    return placer._clamp_movable_to_canvas(full, benchmark)


def _score_full(
    *,
    handle,
    existing_rows,
    candidate_id: str,
    benchmark,
    plc,
    label: str,
    stage: str,
    family: str,
    params: dict[str, Any],
    features: dict[str, Any],
    full: torch.Tensor,
    hard_valid: bool | None,
    generation_sec: float,
):
    if candidate_id in existing_rows:
        return existing_rows[candidate_id]

    score_start = time.time()
    costs = compute_proxy_cost(full, benchmark, plc)
    checkpoint = CHECKPOINT_SCORES.get(benchmark.name)
    proxy = float(costs["proxy_cost"])
    overlap_count = int(costs["overlap_count"])
    valid = overlap_count == 0 and (hard_valid is not False)
    row = {
        "benchmark": benchmark.name,
        "candidate_id": candidate_id,
        "label": label,
        "stage": stage,
        "family": family,
        "params": params,
        "features": features,
        "valid": bool(valid),
        "hard_valid": hard_valid,
        "proxy_cost": proxy,
        "wirelength_cost": float(costs["wirelength_cost"]),
        "density_cost": float(costs["density_cost"]),
        "congestion_cost": float(costs["congestion_cost"]),
        "overlap_count": overlap_count,
        "checkpoint_score": checkpoint,
        "improvement_vs_checkpoint": None if checkpoint is None else checkpoint - proxy,
        "generation_sec": generation_sec,
        "score_sec": time.time() - score_start,
    }
    _append_row(handle, existing_rows, row)
    return row


def _feature_summary(features: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in features.items():
        if isinstance(value, float):
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out


def _benchmark_context(mod, placer, benchmark, plc):
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype("float64")
    initial = benchmark.macro_positions[:n_hard].cpu().numpy().astype("float64")
    fixed_mask = benchmark.macro_fixed[:n_hard].cpu().numpy().astype(bool)
    movable = ~fixed_mask
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    edges, edge_weights, degree = mod._extract_edges(benchmark, plc)
    features = placer._features(benchmark, sizes, degree)
    return {
        "n_hard": n_hard,
        "sizes": sizes,
        "initial": initial,
        "fixed_mask": fixed_mask,
        "movable": movable,
        "cw": cw,
        "ch": ch,
        "edges": edges,
        "edge_weights": edge_weights,
        "degree": degree,
        "features": features,
    }


def _hard_candidate(
    *,
    mod,
    placer,
    benchmark,
    plc,
    ctx,
    handle,
    existing_rows,
    label: str,
    family: str,
    params: dict[str, Any],
    hard_pos: np.ndarray,
    generation_sec: float,
):
    hard_valid = mod._validate_hard(
        hard_pos,
        ctx["sizes"],
        ctx["initial"],
        ctx["fixed_mask"],
        ctx["cw"],
        ctx["ch"],
    )
    full = _full_from_hard(placer, benchmark, ctx["n_hard"], hard_pos)
    candidate_id = _candidate_id(benchmark.name, "hard", label, params)
    row = _score_full(
        handle=handle,
        existing_rows=existing_rows,
        candidate_id=candidate_id,
        benchmark=benchmark,
        plc=plc,
        label=label,
        stage="hard",
        family=family,
        params=params,
        features=_feature_summary(ctx["features"]),
        full=full,
        hard_valid=bool(hard_valid),
        generation_sec=generation_sec,
    )
    return row, full, hard_pos


def _score_soft_candidate(
    *,
    placer,
    benchmark,
    plc,
    ctx,
    handle,
    existing_rows,
    source_row,
    source_full,
    strength: float,
    steps: int,
):
    params = {
        "source_candidate_id": source_row["candidate_id"],
        "source_label": source_row["label"],
        "strength": strength,
        "steps": steps,
    }
    label = f"soft_s{strength:g}_t{steps}_{source_row['label']}"
    candidate_id = _candidate_id(benchmark.name, "soft", label, params)
    if candidate_id in existing_rows:
        return existing_rows[candidate_id], source_full

    start = time.time()
    full = placer._soft_hotspot_relief(
        source_full,
        benchmark,
        plc,
        strength=strength,
        steps=steps,
    )
    row = _score_full(
        handle=handle,
        existing_rows=existing_rows,
        candidate_id=candidate_id,
        benchmark=benchmark,
        plc=plc,
        label=label,
        stage="soft",
        family="soft_hotspot_sweep",
        params=params,
        features=_feature_summary(ctx["features"]),
        full=full,
        hard_valid=True,
        generation_sec=time.time() - start,
    )
    return row, full


def _score_local_search(
    *,
    placer,
    benchmark,
    plc,
    ctx,
    handle,
    existing_rows,
    source_row,
    source_full,
):
    params = {
        "source_candidate_id": source_row["candidate_id"],
        "source_label": source_row["label"],
        "source_proxy_cost": source_row["proxy_cost"],
    }
    label = f"official_hard_local_search_{source_row['label']}"
    candidate_id = _candidate_id(benchmark.name, "local_search", label, params)
    if candidate_id in existing_rows:
        return existing_rows[candidate_id]

    start = time.time()
    local_full, local_score = placer._official_hard_local_search(
        source_full,
        benchmark,
        plc,
        float(source_row["proxy_cost"]),
        ctx["features"],
        time.time(),
    )
    params["proxy_from_method"] = float(local_score)
    return _score_full(
        handle=handle,
        existing_rows=existing_rows,
        candidate_id=candidate_id,
        benchmark=benchmark,
        plc=plc,
        label=label,
        stage="local_search",
        family="official_hard_local_search",
        params=params,
        features=_feature_summary(ctx["features"]),
        full=local_full,
        hard_valid=True,
        generation_sec=time.time() - start,
    )


def _should_stop(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline


def _generate_hard_candidates(mod, placer, benchmark, plc, ctx, handle, existing_rows, deadline):
    rows = []
    full_by_id = {}

    def add(label, family, params, hard_pos, generation_sec):
        row, full, _ = _hard_candidate(
            mod=mod,
            placer=placer,
            benchmark=benchmark,
            plc=plc,
            ctx=ctx,
            handle=handle,
            existing_rows=existing_rows,
            label=label,
            family=family,
            params=params,
            hard_pos=hard_pos,
            generation_sec=generation_sec,
        )
        rows.append(row)
        full_by_id[row["candidate_id"]] = full
        print(
            f"  {row['stage']:<12} {row['label']:<42} "
            f"proxy={row['proxy_cost']:.6f} overlaps={row['overlap_count']} "
            f"valid={row['valid']}",
            flush=True,
        )

    start = time.time()
    add("legalized_initial", "baseline", {"gap": 0.035},
        placer._legalize(ctx["initial"], ctx["movable"], ctx["sizes"], ctx["cw"], ctx["ch"], gap=0.035),
        time.time() - start)

    if _should_stop(deadline):
        return rows, full_by_id

    start = time.time()
    add("will_seed_legalized", "baseline", {},
        placer._will_seed_legalize(ctx["initial"], ctx["movable"], ctx["sizes"], ctx["cw"], ctx["ch"]),
        time.time() - start)

    for clearance in RAW_REPAIR_CLEARANCES:
        if _should_stop(deadline):
            return rows, full_by_id
        start = time.time()
        hard_pos = placer._repair_hard_overlaps(
            ctx["initial"],
            ctx["movable"],
            ctx["sizes"],
            ctx["cw"],
            ctx["ch"],
            clearance=clearance,
        )
        add(
            f"raw_repair_c{clearance:g}",
            "raw_repair",
            {"clearance": clearance},
            hard_pos,
            time.time() - start,
        )

    for gap in PAIR_PUSH_GAPS:
        for damping in PAIR_PUSH_DAMPING:
            for max_passes in PAIR_PUSH_PASSES:
                if _should_stop(deadline):
                    return rows, full_by_id
                start = time.time()
                hard_pos = placer._pair_push_legalize(
                    ctx["initial"],
                    ctx["movable"],
                    ctx["sizes"],
                    ctx["cw"],
                    ctx["ch"],
                    gap=gap,
                    max_passes=max_passes,
                    damping=damping,
                )
                repaired = placer._repair_hard_overlaps(
                    hard_pos,
                    ctx["movable"],
                    ctx["sizes"],
                    ctx["cw"],
                    ctx["ch"],
                    clearance=1e-4,
                )
                add(
                    f"pair_repair_g{gap:g}_d{damping:g}_p{max_passes}",
                    "pair_push_repair",
                    {
                        "gap": gap,
                        "damping": damping,
                        "max_passes": max_passes,
                        "repair_clearance": 1e-4,
                    },
                    repaired,
                    time.time() - start,
                )
    return rows, full_by_id


def _search_benchmark(mod, placer, name: str, handle, existing_rows, deadline, args):
    benchmark, plc = mod._quiet_call(load_benchmark_from_dir, str(TESTCASE_ROOT / name))
    ctx = _benchmark_context(mod, placer, benchmark, plc)
    print(
        f"{name}: n_hard={ctx['n_hard']} features={_feature_summary(ctx['features'])}",
        flush=True,
    )

    rows, full_by_id = _generate_hard_candidates(
        mod, placer, benchmark, plc, ctx, handle, existing_rows, deadline
    )
    valid_hard = [row for row in rows if row.get("valid")]
    valid_hard.sort(key=lambda row: row["proxy_cost"])

    all_valid_sources = []
    for row in valid_hard[: args.top_hard_sources]:
        source_full = full_by_id[row["candidate_id"]]
        all_valid_sources.append((row, source_full))
        for strength in SOFT_STRENGTHS:
            for steps in SOFT_STEPS:
                if _should_stop(deadline):
                    return rows
                soft_row, soft_full = _score_soft_candidate(
                    placer=placer,
                    benchmark=benchmark,
                    plc=plc,
                    ctx=ctx,
                    handle=handle,
                    existing_rows=existing_rows,
                    source_row=row,
                    source_full=source_full,
                    strength=strength,
                    steps=steps,
                )
                rows.append(soft_row)
                if soft_row.get("valid"):
                    all_valid_sources.append((soft_row, soft_full))
                print(
                    f"  {soft_row['stage']:<12} {soft_row['label']:<42} "
                    f"proxy={soft_row['proxy_cost']:.6f} overlaps={soft_row['overlap_count']} "
                    f"valid={soft_row['valid']}",
                    flush=True,
                )

    checkpoint = CHECKPOINT_SCORES.get(name)
    if checkpoint is None:
        return rows

    local_sources = []
    seen_sources = set()
    for row, full in sorted(all_valid_sources, key=lambda item: item[0]["proxy_cost"]):
        if row["candidate_id"] in seen_sources:
            continue
        seen_sources.add(row["candidate_id"])
        if row["proxy_cost"] <= checkpoint - args.local_search_threshold:
            local_sources.append((row, full))

    for row, full in local_sources:
        if _should_stop(deadline):
            return rows
        local_row = _score_local_search(
            placer=placer,
            benchmark=benchmark,
            plc=plc,
            ctx=ctx,
            handle=handle,
            existing_rows=existing_rows,
            source_row=row,
            source_full=full,
        )
        rows.append(local_row)
        print(
            f"  {local_row['stage']:<12} {local_row['label']:<42} "
            f"proxy={local_row['proxy_cost']:.6f} overlaps={local_row['overlap_count']} "
            f"valid={local_row['valid']}",
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", "-b", action="append", help="IBM benchmark name; repeatable")
    parser.add_argument("--all", action="store_true", help="search all IBM benchmarks")
    parser.add_argument("--out", type=Path, default=Path("runs/heuristic_learning/candidate_search.jsonl"))
    parser.add_argument("--resume", action="store_true", help="reuse rows already present in --out")
    parser.add_argument("--time-limit-hours", type=float, default=10.0)
    parser.add_argument("--top-hard-sources", type=int, default=5)
    parser.add_argument("--local-search-threshold", type=float, default=0.005)
    args = parser.parse_args()

    if not TESTCASE_ROOT.exists():
        raise SystemExit("external/MacroPlacement submodule is not initialized")

    mod = _load_hl_module()
    placer = mod.HeuristicLearningPlacer()
    benchmarks = _benchmarks_from_args(args)
    deadline = None
    if args.time_limit_hours and args.time_limit_hours > 0:
        deadline = time.time() + args.time_limit_hours * 3600.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_existing(args.out) if args.resume else {}
    mode = "a" if args.resume else "w"
    all_rows = []
    with args.out.open(mode) as handle:
        for name in benchmarks:
            if _should_stop(deadline):
                break
            start = time.time()
            rows = _search_benchmark(mod, placer, name, handle, existing_rows, deadline, args)
            all_rows.extend(rows)
            valid_rows = [row for row in rows if row.get("valid")]
            if valid_rows:
                best = min(valid_rows, key=lambda row: row["proxy_cost"])
                print(
                    f"{name}: best={best['label']} proxy={best['proxy_cost']:.6f} "
                    f"improvement={best.get('improvement_vs_checkpoint')} "
                    f"rows={len(rows)} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
            else:
                print(f"{name}: no valid rows elapsed={time.time() - start:.1f}s", flush=True)

    print(f"wrote/searched {len(all_rows)} rows; output={args.out}")


if __name__ == "__main__":
    main()
