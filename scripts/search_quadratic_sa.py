#!/usr/bin/env python3
"""Search quadratic-seed placements with real proxy simulated annealing.

This is an offline experiment driver. It does not modify the submitted placer;
it imports the heuristic-learning placer's private helpers so legality and
scoring evidence can be compared against existing checkpoint rows.
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

try:
    from scipy import sparse
    from scipy.sparse import linalg as spla
except ModuleNotFoundError:  # uv dev env may omit scipy; Docker includes it.
    sparse = None
    spla = None

from macro_place.evaluate import IBM_BENCHMARKS
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


REPO_ROOT = Path(__file__).resolve().parents[1]
PLACER_PATH = REPO_ROOT / "submissions" / "heuristic_learning" / "placer.py"
TESTCASE_ROOT = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"

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

DEFAULT_PAIR_PARAMS = {
    "gap": 1e-4,
    "damping": 0.45,
    "max_passes": 500,
    "repair_clearance": 1e-4,
}


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


def _feature_summary(features: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in features.items():
        out[key] = round(value, 6) if isinstance(value, float) else value
    return out


def _load_reference(path: Path | None):
    rows_by_id = {}
    rows_by_bench = {}
    if path is None or not path.exists():
        return rows_by_id, rows_by_bench
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            benchmark = row.get("benchmark")
            if candidate_id:
                rows_by_id[candidate_id] = row
            if row.get("valid") and benchmark:
                cur = rows_by_bench.get(benchmark)
                if cur is None or row["proxy_cost"] < cur["proxy_cost"]:
                    rows_by_bench[benchmark] = row
    return rows_by_id, rows_by_bench


def _best_pair_params(reference_rows: dict[str, dict[str, Any]], benchmark: str):
    best = None
    for row in reference_rows.values():
        if row.get("benchmark") != benchmark or not row.get("valid"):
            continue
        if row.get("family") != "pair_push_repair" or row.get("stage") != "hard":
            continue
        if best is None or row["proxy_cost"] < best["proxy_cost"]:
            best = row
    if best is None:
        return dict(DEFAULT_PAIR_PARAMS)
    params = dict(DEFAULT_PAIR_PARAMS)
    params.update(best.get("params", {}))
    return params


def _best_soft_params(reference_rows: dict[str, dict[str, Any]], benchmark: str):
    best = None
    for row in reference_rows.values():
        if row.get("benchmark") != benchmark or not row.get("valid"):
            continue
        if row.get("family") != "soft_hotspot_sweep":
            continue
        if best is None or row["proxy_cost"] < best["proxy_cost"]:
            best = row
    if best is None:
        return {"strength": 0.25, "steps": 1, "source": "default"}
    params = best.get("params", {})
    return {
        "strength": float(params.get("strength", 0.25)),
        "steps": int(params.get("steps", 1)),
        "source": best.get("candidate_id"),
    }


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
    sa_trace: list[dict[str, Any]] | None = None,
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
    if sa_trace is not None:
        row["sa_trace"] = sa_trace
    _append_row(handle, existing_rows, row)
    return row


def _solve_quadratic(ctx, weights, ridge=1e-3):
    n = ctx["n_hard"]
    initial = ctx["initial"]
    movable = ctx["movable"]
    fixed_mask = ctx["fixed_mask"]
    move_idx = np.where(movable)[0]
    index = {int(node): pos for pos, node in enumerate(move_idx)}
    if len(move_idx) == 0:
        return initial.copy()

    rows = []
    cols = []
    data = []
    bx = np.zeros(len(move_idx), dtype=np.float64)
    by = np.zeros(len(move_idx), dtype=np.float64)

    def add_diag(node, value):
        row = index[int(node)]
        rows.append(row)
        cols.append(row)
        data.append(value)

    for (a, b), weight in zip(ctx["edges"], weights):
        a = int(a)
        b = int(b)
        weight = float(max(weight, 1e-9))
        a_mov = movable[a]
        b_mov = movable[b]
        if a_mov and b_mov:
            ia = index[a]
            ib = index[b]
            rows.extend([ia, ib, ia, ib])
            cols.extend([ia, ib, ib, ia])
            data.extend([weight, weight, -weight, -weight])
        elif a_mov and fixed_mask[b]:
            add_diag(a, weight)
            bx[index[a]] += weight * initial[b, 0]
            by[index[a]] += weight * initial[b, 1]
        elif b_mov and fixed_mask[a]:
            add_diag(b, weight)
            bx[index[b]] += weight * initial[a, 0]
            by[index[b]] += weight * initial[a, 1]

    # Stabilize disconnected movable components without pinning them strongly.
    for node in move_idx:
        row = index[int(node)]
        rows.append(row)
        cols.append(row)
        data.append(ridge)
        bx[row] += ridge * initial[node, 0]
        by[row] += ridge * initial[node, 1]

    shape = (len(move_idx), len(move_idx))
    if sparse is not None and spla is not None:
        matrix = sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
        x = spla.spsolve(matrix, bx)
        y = spla.spsolve(matrix, by)
    else:
        matrix = np.zeros(shape, dtype=np.float64)
        np.add.at(matrix, (rows, cols), data)
        x = np.linalg.solve(matrix, bx)
        y = np.linalg.solve(matrix, by)
    out = initial.copy()
    out[move_idx, 0] = x
    out[move_idx, 1] = y
    half_w = ctx["sizes"][:, 0] / 2.0
    half_h = ctx["sizes"][:, 1] / 2.0
    out[:, 0] = np.clip(out[:, 0], half_w, ctx["cw"] - half_w)
    out[:, 1] = np.clip(out[:, 1], half_h, ctx["ch"] - half_h)
    return out


def _quadratic_seed(ctx):
    if len(ctx["edges"]) == 0:
        return ctx["initial"].copy()
    weights = ctx["edge_weights"].astype(np.float64).copy()
    first = _solve_quadratic(ctx, weights)
    a = ctx["edges"][:, 0]
    b = ctx["edges"][:, 1]
    dist = np.linalg.norm(first[a] - first[b], axis=1)
    weights = weights / np.maximum(dist, 1.0)
    return _solve_quadratic(ctx, weights)


def _legalize_seed(placer, ctx, hard_pos, params, max_pair_passes, max_repair_iters):
    max_passes = min(
        int(params.get("max_passes", DEFAULT_PAIR_PARAMS["max_passes"])),
        int(max_pair_passes),
    )
    pushed = placer._pair_push_legalize(
        hard_pos,
        ctx["movable"],
        ctx["sizes"],
        ctx["cw"],
        ctx["ch"],
        gap=float(params.get("gap", DEFAULT_PAIR_PARAMS["gap"])),
        max_passes=max_passes,
        damping=float(params.get("damping", DEFAULT_PAIR_PARAMS["damping"])),
    )
    repaired = placer._repair_hard_overlaps(
        pushed,
        ctx["movable"],
        ctx["sizes"],
        ctx["cw"],
        ctx["ch"],
        clearance=float(params.get("repair_clearance", DEFAULT_PAIR_PARAMS["repair_clearance"])),
        max_iters=int(max_repair_iters),
    )
    return repaired


def _quadratic_pull_seed(placer, ctx, target):
    base = placer._legalize(
        ctx["initial"], ctx["movable"], ctx["sizes"], ctx["cw"], ctx["ch"], gap=0.035
    )
    out = base.copy()
    movable_idx = np.where(ctx["movable"])[0]
    distances = np.linalg.norm(target[movable_idx] - base[movable_idx], axis=1)
    order = movable_idx[np.argsort(-distances)]
    for idx in order:
        direction = target[idx] - out[idx]
        if float(np.linalg.norm(direction)) < 1e-9:
            continue
        for alpha in (1.0, 0.75, 0.5, 0.25, 0.10):
            candidate = out.copy()
            candidate[idx] = out[idx] + alpha * direction
            half_w = ctx["sizes"][idx, 0] / 2.0
            half_h = ctx["sizes"][idx, 1] / 2.0
            candidate[idx, 0] = np.clip(candidate[idx, 0], half_w, ctx["cw"] - half_w)
            candidate[idx, 1] = np.clip(candidate[idx, 1], half_h, ctx["ch"] - half_h)
            if _legal_single(ctx, candidate, int(idx)):
                out = candidate
                break
    return out

def _validate_hard(mod, ctx, hard_pos):
    return bool(
        mod._validate_hard(
            hard_pos,
            ctx["sizes"],
            ctx["initial"],
            ctx["fixed_mask"],
            ctx["cw"],
            ctx["ch"],
        )
    )


def _legal_single(ctx, pos, idx, gap=0.0):
    sizes = ctx["sizes"]
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    if pos[idx, 0] < half_w[idx] - 1e-9 or pos[idx, 0] > ctx["cw"] - half_w[idx] + 1e-9:
        return False
    if pos[idx, 1] < half_h[idx] - 1e-9 or pos[idx, 1] > ctx["ch"] - half_h[idx] + 1e-9:
        return False
    dx = np.abs(pos[idx, 0] - pos[:, 0])
    dy = np.abs(pos[idx, 1] - pos[:, 1])
    sep_x = (sizes[idx, 0] + sizes[:, 0]) / 2.0 + gap
    sep_y = (sizes[idx, 1] + sizes[:, 1]) / 2.0 + gap
    overlaps = (dx < sep_x) & (dy < sep_y)
    overlaps[idx] = False
    return not bool(overlaps.any())


def _apply_hard_to_full(placer, full, benchmark, n_hard, hard_pos):
    candidate = full.clone()
    candidate[:n_hard] = torch.tensor(hard_pos, dtype=torch.float32)
    return placer._clamp_movable_to_canvas(candidate, benchmark)


def _score_proxy(full, benchmark, plc):
    costs = compute_proxy_cost(full, benchmark, plc)
    if int(costs["overlap_count"]) != 0:
        return float("inf"), costs
    return float(costs["proxy_cost"]), costs


def _cold_cells_from_current(full, benchmark, plc):
    compute_proxy_cost(full, benchmark, plc)
    rows = int(benchmark.grid_rows)
    cols = int(benchmark.grid_cols)
    density = np.asarray(getattr(plc, "grid_cells", [0.0] * (rows * cols)), dtype=np.float64)
    h_cong = np.asarray(getattr(plc, "H_routing_cong", [0.0] * (rows * cols)), dtype=np.float64)
    v_cong = np.asarray(getattr(plc, "V_routing_cong", [0.0] * (rows * cols)), dtype=np.float64)
    if density.size != rows * cols or h_cong.size != rows * cols or v_cong.size != rows * cols:
        return None
    hot = density.reshape(rows, cols) + 0.35 * np.maximum(
        h_cong.reshape(rows, cols), v_cong.reshape(rows, cols)
    )
    flat = hot.ravel()
    cutoff = np.percentile(flat, 35)
    cold = np.argwhere(hot <= cutoff)
    return cold if len(cold) else None


def _proposal_shift(ctx, pos, rng, cold_cells, cell_w, cell_h):
    movable_idx = np.where(ctx["movable"])[0]
    sizes = ctx["sizes"]
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    for _ in range(32):
        idx = int(rng.choice(movable_idx))
        if cold_cells is not None:
            row, col = cold_cells[int(rng.integers(0, len(cold_cells)))]
            tx = (col + rng.uniform(0.25, 0.75)) * cell_w
            ty = (row + rng.uniform(0.25, 0.75)) * cell_h
        else:
            tx = rng.uniform(half_w[idx], ctx["cw"] - half_w[idx])
            ty = rng.uniform(half_h[idx], ctx["ch"] - half_h[idx])
        candidate = pos.copy()
        candidate[idx, 0] = np.clip(tx, half_w[idx], ctx["cw"] - half_w[idx])
        candidate[idx, 1] = np.clip(ty, half_h[idx], ctx["ch"] - half_h[idx])
        if _legal_single(ctx, candidate, idx):
            return candidate
    return None


def _proposal_swap(ctx, pos, rng):
    movable_idx = np.where(ctx["movable"])[0]
    sizes = ctx["sizes"]
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    if len(movable_idx) < 2:
        return None
    for _ in range(32):
        i, j = rng.choice(movable_idx, size=2, replace=False)
        i = int(i)
        j = int(j)
        candidate = pos.copy()
        candidate[i] = pos[j]
        candidate[j] = pos[i]
        candidate[i, 0] = np.clip(candidate[i, 0], half_w[i], ctx["cw"] - half_w[i])
        candidate[i, 1] = np.clip(candidate[i, 1], half_h[i], ctx["ch"] - half_h[i])
        candidate[j, 0] = np.clip(candidate[j, 0], half_w[j], ctx["cw"] - half_w[j])
        candidate[j, 1] = np.clip(candidate[j, 1], half_h[j], ctx["ch"] - half_h[j])
        if _legal_single(ctx, candidate, i) and _legal_single(ctx, candidate, j):
            return candidate
    return None


def _proxy_sa(
    *,
    mod,
    placer,
    benchmark,
    plc,
    ctx,
    seed_full,
    seed_label,
    rng,
    proposals,
    t0_frac,
    tf_frac,
    per_benchmark_deadline,
):
    hard_pos = seed_full[: ctx["n_hard"]].cpu().numpy().astype(np.float64)
    if not _validate_hard(mod, ctx, hard_pos):
        return seed_full, float("inf"), []
    cur_full = seed_full.clone()
    cur_score, _ = _score_proxy(cur_full, benchmark, plc)
    best_full = cur_full.clone()
    best_score = cur_score
    if not math.isfinite(cur_score):
        return best_full, best_score, []
    t0 = max(cur_score * t0_frac, 1e-9)
    tf = max(cur_score * tf_frac, 1e-12)
    cell_w = ctx["cw"] / max(int(benchmark.grid_cols), 1)
    cell_h = ctx["ch"] / max(int(benchmark.grid_rows), 1)
    cold_cells = _cold_cells_from_current(cur_full, benchmark, plc)
    rejected = 0
    accepted_total = 0
    trace = []

    for step in range(max(0, proposals)):
        if time.time() >= per_benchmark_deadline:
            break
        frac = step / max(1, proposals - 1)
        temp = t0 * (tf / t0) ** frac
        if rng.random() < 0.5:
            candidate_pos = _proposal_shift(ctx, hard_pos, rng, cold_cells, cell_w, cell_h)
            move_kind = "shift"
        else:
            candidate_pos = _proposal_swap(ctx, hard_pos, rng)
            move_kind = "swap"
        if candidate_pos is None:
            rejected += 1
            if rejected >= 150:
                break
            continue

        candidate_full = _apply_hard_to_full(
            placer, cur_full, benchmark, ctx["n_hard"], candidate_pos
        )
        candidate_score, _ = _score_proxy(candidate_full, benchmark, plc)
        delta = candidate_score - cur_score
        accepted = math.isfinite(candidate_score) and (
            delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-12))
        )
        if accepted:
            hard_pos = candidate_pos
            cur_full = candidate_full
            cur_score = candidate_score
            rejected = 0
            accepted_total += 1
            if cur_score < best_score:
                best_score = cur_score
                best_full = cur_full.clone()
                cold_cells = _cold_cells_from_current(cur_full, benchmark, plc)
        else:
            rejected += 1
            if rejected >= 150:
                break
        if step % 25 == 0 or accepted:
            trace.append(
                {
                    "proposal_idx": step,
                    "move": move_kind,
                    "accepted": accepted,
                    "proxy": cur_score,
                    "best_proxy": best_score,
                    "temp": temp,
                    "accepted_total": accepted_total,
                    "source": seed_label,
                }
            )
    return best_full, best_score, trace


def _row_for_full(
    *,
    mod,
    placer,
    benchmark,
    plc,
    ctx,
    handle,
    existing_rows,
    label,
    stage,
    family,
    params,
    full,
    generation_sec,
    sa_trace=None,
):
    hard_pos = full[: ctx["n_hard"]].cpu().numpy().astype(np.float64)
    hard_valid = _validate_hard(mod, ctx, hard_pos)
    candidate_id = _candidate_id(benchmark.name, stage, label, params)
    return _score_full(
        handle=handle,
        existing_rows=existing_rows,
        candidate_id=candidate_id,
        benchmark=benchmark,
        plc=plc,
        label=label,
        stage=stage,
        family=family,
        params=params,
        features=_feature_summary(ctx["features"]),
        full=full,
        hard_valid=hard_valid,
        generation_sec=generation_sec,
        sa_trace=sa_trace,
    )


def _search_benchmark(
    *,
    mod,
    placer,
    benchmark_name,
    reference_rows,
    handle,
    existing_rows,
    args,
    global_deadline,
    bench_index,
):
    benchmark, plc = mod._quiet_call(
        load_benchmark_from_dir, str(TESTCASE_ROOT / benchmark_name)
    )
    ctx = _benchmark_context(mod, placer, benchmark, plc)
    per_benchmark_deadline = min(
        global_deadline,
        time.time() + args.per_benchmark_minutes * 60.0,
    )
    print(
        f"{benchmark_name}: n_hard={ctx['n_hard']} features={_feature_summary(ctx['features'])}",
        flush=True,
    )

    pair_params = _best_pair_params(reference_rows, benchmark_name)
    soft_params = _best_soft_params(reference_rows, benchmark_name)
    rng = np.random.default_rng(args.seed + bench_index)
    rows = []

    start = time.time()
    hard_target = _quadratic_seed(ctx)
    if args.seed_legalizer == "pull":
        hard_seed = _quadratic_pull_seed(placer, ctx, hard_target)
    else:
        hard_seed = _legalize_seed(
            placer,
            ctx,
            hard_target,
            pair_params,
            max_pair_passes=args.max_pair_passes,
            max_repair_iters=args.max_repair_iters,
        )
    seed_full = _full_from_hard(placer, benchmark, ctx["n_hard"], hard_seed)
    quad_params = {
        "pair_params": pair_params,
        "soft_params": None,
        "seed": args.seed + bench_index,
    }
    quad_row = _row_for_full(
        mod=mod,
        placer=placer,
        benchmark=benchmark,
        plc=plc,
        ctx=ctx,
        handle=handle,
        existing_rows=existing_rows,
        label="quadratic_seed",
        stage="quadratic_seed",
        family="quadratic_seed",
        params=quad_params,
        full=seed_full,
        generation_sec=time.time() - start,
    )
    rows.append(quad_row)
    print(
        f"  {quad_row['stage']:<14} {quad_row['label']:<36} "
        f"proxy={quad_row['proxy_cost']:.6f} overlaps={quad_row['overlap_count']} "
        f"valid={quad_row['valid']}",
        flush=True,
    )

    soft_full = seed_full
    if benchmark.num_soft_macros > 0 and time.time() < per_benchmark_deadline:
        start = time.time()
        soft_full = placer._soft_hotspot_relief(
            seed_full,
            benchmark,
            plc,
            strength=float(soft_params["strength"]),
            steps=int(soft_params["steps"]),
        )
        soft_row = _row_for_full(
            mod=mod,
            placer=placer,
            benchmark=benchmark,
            plc=plc,
            ctx=ctx,
            handle=handle,
            existing_rows=existing_rows,
            label="quadratic_seed_soft",
            stage="quadratic_seed",
            family="quadratic_seed_soft",
            params={
                "pair_params": pair_params,
                "soft_params": soft_params,
                "seed": args.seed + bench_index,
            },
            full=soft_full,
            generation_sec=time.time() - start,
        )
        rows.append(soft_row)
        print(
            f"  {soft_row['stage']:<14} {soft_row['label']:<36} "
            f"proxy={soft_row['proxy_cost']:.6f} overlaps={soft_row['overlap_count']} "
            f"valid={soft_row['valid']}",
            flush=True,
        )

    sa_sources = [("quadratic_seed", seed_full), ("quadratic_seed_soft", soft_full)]
    seen = set()
    for source_label, source_full in sa_sources:
        if source_label in seen or time.time() >= per_benchmark_deadline:
            continue
        seen.add(source_label)
        start = time.time()
        best_full, best_score, trace = _proxy_sa(
            mod=mod,
            placer=placer,
            benchmark=benchmark,
            plc=plc,
            ctx=ctx,
            seed_full=source_full,
            seed_label=source_label,
            rng=rng,
            proposals=args.sa_proposals,
            t0_frac=args.t0_frac,
            tf_frac=args.tf_frac,
            per_benchmark_deadline=per_benchmark_deadline,
        )
        params = {
            "source": source_label,
            "sa_proposals": args.sa_proposals,
            "t0_frac": args.t0_frac,
            "tf_frac": args.tf_frac,
            "seed": args.seed + bench_index,
            "best_proxy_from_loop": best_score,
        }
        row = _row_for_full(
            mod=mod,
            placer=placer,
            benchmark=benchmark,
            plc=plc,
            ctx=ctx,
            handle=handle,
            existing_rows=existing_rows,
            label=f"proxy_sa_{source_label}",
            stage="sa",
            family="proxy_sa",
            params=params,
            full=best_full,
            generation_sec=time.time() - start,
            sa_trace=trace,
        )
        rows.append(row)
        print(
            f"  {row['stage']:<14} {row['label']:<36} "
            f"proxy={row['proxy_cost']:.6f} overlaps={row['overlap_count']} "
            f"valid={row['valid']}",
            flush=True,
        )
    return rows


def _should_stop(deadline):
    return time.time() >= deadline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", "-b", action="append", help="IBM benchmark name; repeatable")
    parser.add_argument("--all", action="store_true", help="search all IBM benchmarks")
    parser.add_argument("--out", type=Path, default=Path("runs/heuristic_learning/quad_sa.jsonl"))
    parser.add_argument("--resume", action="store_true", help="reuse rows already present in --out")
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--time-limit-hours", type=float, default=4.0)
    parser.add_argument("--per-benchmark-minutes", type=float, default=12.0)
    parser.add_argument("--sa-proposals", type=int, default=400)
    parser.add_argument("--max-pair-passes", type=int, default=250)
    parser.add_argument("--max-repair-iters", type=int, default=12)
    parser.add_argument("--seed-legalizer", choices=("pull", "pair"), default="pull")
    parser.add_argument("--t0-frac", type=float, default=0.02)
    parser.add_argument("--tf-frac", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()

    if not TESTCASE_ROOT.exists():
        raise SystemExit("external/MacroPlacement submodule is not initialized")

    mod = _load_hl_module()
    placer = mod.HeuristicLearningPlacer()
    benchmarks = _benchmarks_from_args(args)
    _, reference_by_bench = _load_reference(args.reference_jsonl)
    reference_rows, _ = _load_reference(args.reference_jsonl)
    existing_rows = _load_existing(args.out) if args.resume else {}
    deadline = time.time() + max(args.time_limit_hours, 0.001) * 3600.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    all_rows = []
    with args.out.open(mode) as handle:
        for bench_index, benchmark_name in enumerate(benchmarks):
            if _should_stop(deadline):
                break
            start = time.time()
            rows = _search_benchmark(
                mod=mod,
                placer=placer,
                benchmark_name=benchmark_name,
                reference_rows=reference_rows,
                handle=handle,
                existing_rows=existing_rows,
                args=args,
                global_deadline=deadline,
                bench_index=bench_index,
            )
            all_rows.extend(rows)
            valid_rows = [row for row in rows if row.get("valid")]
            if valid_rows:
                best = min(valid_rows, key=lambda row: row["proxy_cost"])
                ref = reference_by_bench.get(benchmark_name)
                ref_score = None if ref is None else ref.get("proxy_cost")
                print(
                    f"{benchmark_name}: best={best['label']} proxy={best['proxy_cost']:.6f} "
                    f"improvement={best.get('improvement_vs_checkpoint')} "
                    f"reference_best={ref_score} rows={len(rows)} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"{benchmark_name}: no valid rows elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
    print(f"wrote/searched {len(all_rows)} rows; output={args.out}", flush=True)


if __name__ == "__main__":
    main()
