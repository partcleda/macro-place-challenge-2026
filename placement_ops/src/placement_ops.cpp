// placement_ops.cpp
// Fast C++ kernels for macro placement SA.
// Provides: init_nets, rebuild_cache, compute_hpwl,
//           compute_delta_hpwl, commit_move, count_overlap
//
// Build: cd placement_ops && python setup.py build_ext --inplace
// Usage in Python: import placement_ops

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;
using f32 = float;

// ─────────────────────────────────────────────────────────────────────────────
// DATA STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

struct Pin {
    int macro_idx;   // -1 = fixed IO port (absolute position in ox, oy)
    f32 ox, oy;      // offset from macro center; absolute pos if port
};

struct Net {
    std::vector<Pin> pins;
};

// Module-level global state — one set per Python process.
// Each worker process initialises its own copy via init_nets().
static std::vector<Net>              g_nets;
static std::vector<std::vector<int>> g_macro_nets;   // macro → list of net indices
static int                           g_num_macros = 0;

// Incremental HPWL cache
static std::vector<f32> g_net_hpwl;   // per-net cached HPWL
static f32              g_total_hpwl = 0.0f;

// Scratch buffer for pending (not yet committed) delta computation
static std::vector<f32> g_pending;   // new HPWL per affected net after delta call
static std::vector<int> g_pending_nets;  // which nets are in g_pending

// ─────────────────────────────────────────────────────────────────────────────
// INTERNAL: compute HPWL of one net from raw float* positions array
// ─────────────────────────────────────────────────────────────────────────────

static inline f32 _net_hpwl(int net_idx, const f32* pos) {
    const Net& net = g_nets[net_idx];
    if (net.pins.size() < 2) return 0.0f;

    f32 xmin =  std::numeric_limits<f32>::max();
    f32 xmax = -std::numeric_limits<f32>::max();
    f32 ymin =  std::numeric_limits<f32>::max();
    f32 ymax = -std::numeric_limits<f32>::max();

    for (const Pin& p : net.pins) {
        f32 px, py;
        if (p.macro_idx < 0) {
            px = p.ox; py = p.oy;
        } else {
            px = pos[p.macro_idx * 2 + 0] + p.ox;
            py = pos[p.macro_idx * 2 + 1] + p.oy;
        }
        if (px < xmin) xmin = px;
        if (px > xmax) xmax = px;
        if (py < ymin) ymin = py;
        if (py > ymax) ymax = py;
    }
    return (xmax - xmin) + (ymax - ymin);
}

// ─────────────────────────────────────────────────────────────────────────────
// init_nets — initialise all state for one benchmark. Call once per process.
//
// nets_py format: list of tuples
//   (drv_idx: int, drv_ox: float, drv_oy: float,
//    sinks: list of (s_idx: int, s_ox: float, s_oy: float))
//
// drv_idx / s_idx == -1  means the pin is a fixed IO port at (drv_ox, drv_oy).
// ─────────────────────────────────────────────────────────────────────────────

void init_nets(const py::list& nets_py, int num_macros) {
    g_nets.clear();
    g_macro_nets.clear();
    g_net_hpwl.clear();
    g_pending.clear();
    g_pending_nets.clear();

    g_num_macros = num_macros;
    g_macro_nets.resize(num_macros);

    int net_idx = 0;
    for (const py::handle& h : nets_py) {
        py::tuple t = h.cast<py::tuple>();

        int   drv_idx = t[0].cast<int>();
        f32   drv_ox  = t[1].cast<f32>();
        f32   drv_oy  = t[2].cast<f32>();
        py::list sinks = t[3].cast<py::list>();

        Net net;
        net.pins.push_back({drv_idx, drv_ox, drv_oy});
        if (drv_idx >= 0 && drv_idx < num_macros)
            g_macro_nets[drv_idx].push_back(net_idx);

        for (const py::handle& sh : sinks) {
            py::tuple st = sh.cast<py::tuple>();
            int s_idx = st[0].cast<int>();
            f32 s_ox  = st[1].cast<f32>();
            f32 s_oy  = st[2].cast<f32>();
            net.pins.push_back({s_idx, s_ox, s_oy});
            if (s_idx >= 0 && s_idx < num_macros)
                g_macro_nets[s_idx].push_back(net_idx);
        }

        g_nets.push_back(std::move(net));
        g_net_hpwl.push_back(0.0f);
        ++net_idx;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// rebuild_cache — recompute ALL per-net HPWLs and total.
// Call after: init_nets, GWTW sync, or any large position change.
// positions: numpy float32 array, shape [N, 2], C-contiguous.
// ─────────────────────────────────────────────────────────────────────────────

void rebuild_cache(
    py::array_t<f32, py::array::c_style | py::array::forcecast> positions)
{
    auto buf = positions.unchecked<2>();
    if (buf.shape(1) != 2)
        throw std::runtime_error("positions must be shape [N, 2]");

    const f32* raw = positions.data();
    f32 total = 0.0f;
    int n = (int)g_nets.size();
    for (int i = 0; i < n; ++i) {
        f32 h = _net_hpwl(i, raw);
        g_net_hpwl[i] = h;
        total += h;
    }
    g_total_hpwl = total;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_hpwl — full HPWL. Also rebuilds the cache.
// Returns total HPWL as a float.
// ─────────────────────────────────────────────────────────────────────────────

f32 compute_hpwl(
    py::array_t<f32, py::array::c_style | py::array::forcecast> positions)
{
    rebuild_cache(positions);
    return g_total_hpwl;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_delta_hpwl — INCREMENTAL update for a single-macro shift move.
//
// Temporarily patches pos[moved_idx] = (new_x, new_y), recomputes only the
// nets that touch moved_idx, restores the position, and returns:
//   (delta_hpwl, new_total_hpwl)
//
// The caller may then decide to accept or reject. If accepted, call
// commit_move(moved_idx) to update the cache. If rejected, do nothing.
//
// The positions array is patched in-place during computation and
// immediately restored — this is safe because Python holds the GIL.
// ─────────────────────────────────────────────────────────────────────────────

std::pair<f32, f32> compute_delta_hpwl(
    py::array_t<f32, py::array::c_style | py::array::forcecast> positions,
    int moved_idx,
    f32 new_x,
    f32 new_y)
{
    if (moved_idx < 0 || moved_idx >= g_num_macros)
        throw std::runtime_error("moved_idx out of range");

    f32* raw = const_cast<f32*>(positions.data());

    // Save old position
    f32 old_x = raw[moved_idx * 2 + 0];
    f32 old_y = raw[moved_idx * 2 + 1];

    // Patch to new position
    raw[moved_idx * 2 + 0] = new_x;
    raw[moved_idx * 2 + 1] = new_y;

    const std::vector<int>& affected = g_macro_nets[moved_idx];
    int n_aff = (int)affected.size();

    g_pending.resize(n_aff);
    g_pending_nets.resize(n_aff);

    f32 delta = 0.0f;
    for (int k = 0; k < n_aff; ++k) {
        int ni = affected[k];
        f32 new_h = _net_hpwl(ni, raw);
        g_pending[k]      = new_h;
        g_pending_nets[k] = ni;
        delta += new_h - g_net_hpwl[ni];
    }

    // Restore position
    raw[moved_idx * 2 + 0] = old_x;
    raw[moved_idx * 2 + 1] = old_y;

    return {delta, g_total_hpwl + delta};
}

// ─────────────────────────────────────────────────────────────────────────────
// commit_move — apply the pending delta to the cache.
// Must be called immediately after compute_delta_hpwl if the move is accepted.
// The caller must ALSO update pos[moved_idx] in the numpy array.
// ─────────────────────────────────────────────────────────────────────────────

void commit_move() {
    int n = (int)g_pending_nets.size();
    for (int k = 0; k < n; ++k) {
        int ni = g_pending_nets[k];
        g_total_hpwl  += g_pending[k] - g_net_hpwl[ni];
        g_net_hpwl[ni] = g_pending[k];
    }
    // Clear pending so stale data can't be committed twice
    g_pending.clear();
    g_pending_nets.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// count_overlap — O(n²) pairwise overlap check for hard macros only.
// Returns (overlap_pair_count, total_overlap_area).
//
// positions: float32 [N, 2]
// sizes:     float32 [N, 2]   (width, height per macro)
// hard_idx:  int32   [n_hard] (indices into positions/sizes)
// ─────────────────────────────────────────────────────────────────────────────

std::pair<int, f32> count_overlap(
    py::array_t<f32, py::array::c_style | py::array::forcecast> positions,
    py::array_t<f32, py::array::c_style | py::array::forcecast> sizes,
    py::array_t<int, py::array::c_style | py::array::forcecast> hard_idx,
    f32 threshold = 0.0040f)   // ← add threshold parameter matching evaluator
{
    auto pos = positions.unchecked<2>();
    auto sz  = sizes.unchecked<2>();
    auto idx = hard_idx.unchecked<1>();
    int n = (int)idx.shape(0);

    if (n < 2) return {0, 0.0f};

    // Precompute bounding boxes (avoids recomputation in inner loop)
    std::vector<f32> xmin(n), xmax(n), ymin(n), ymax(n);
    for (int i = 0; i < n; ++i) {
        int mi  = idx(i);
        f32 hw  = sz(mi, 0) * 0.5f;
        f32 hh  = sz(mi, 1) * 0.5f;
        xmin[i] = pos(mi, 0) - hw;
        xmax[i] = pos(mi, 0) + hw;
        ymin[i] = pos(mi, 1) - hh;
        ymax[i] = pos(mi, 1) + hh;
    }

    const f32 eps = 1e-3f;
    int   count = 0;
    f32   area  = 0.0f;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            f32 ox = std::min(xmax[i], xmax[j]) - std::max(xmin[i], xmin[j]);
            f32 oy = std::min(ymax[i], ymax[j]) - std::max(ymin[i], ymin[j]);
            if (ox > 0 && oy > 0) {
                f32 ov_area  = ox * oy;
                // Area of each macro
                int mi = idx(i); int mj = idx(j);
                f32 area_i = sz(mi, 0) * sz(mi, 1);
                f32 area_j = sz(mj, 0) * sz(mj, 1);
                f32 min_area = std::min(area_i, area_j);
                // Use same criterion as evaluator
                if (min_area > 0 && ov_area / min_area > threshold) {
                    ++count;
                    area += ov_area;
                }
            }
        }
    }
    return {count, area};
}

// ─────────────────────────────────────────────────────────────────────────────
// PYBIND11 BINDINGS
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(placement_ops, m) {
    m.doc() = "Fast C++ placement kernels: HPWL (full + incremental) and overlap";

    m.def("init_nets", &init_nets,
        py::arg("nets_py"), py::arg("num_macros"),
        R"doc(
        Initialise net index. Call once per benchmark, per process.

        nets_py: list of (drv_idx, drv_ox, drv_oy, [(s_idx, s_ox, s_oy), ...])
            drv_idx / s_idx == -1 means fixed IO port at (drv_ox, drv_oy).
        num_macros: total number of macros (hard + soft).
        )doc");

    m.def("rebuild_cache", &rebuild_cache,
        py::arg("positions"),
        R"doc(
        Recompute ALL per-net HPWLs and update the running total.
        Call after GWTW sync or any bulk position change.
        positions: float32 numpy array shape [N, 2].
        )doc");

    m.def("compute_hpwl", &compute_hpwl,
        py::arg("positions"),
        R"doc(
        Full HPWL recompute (calls rebuild_cache internally).
        Returns total HPWL as float.
        )doc");

    m.def("compute_delta_hpwl", &compute_delta_hpwl,
        py::arg("positions"), py::arg("moved_idx"),
        py::arg("new_x"), py::arg("new_y"),
        R"doc(
        Incremental HPWL for a single-macro move.
        Patches pos[moved_idx] temporarily, recomputes only affected nets,
        restores position. Returns (delta_hpwl, new_total_hpwl).
        Does NOT update the cache — call commit_move() if you accept.
        )doc");

    m.def("commit_move", &commit_move,
        R"doc(
        Commit the pending delta to the HPWL cache.
        Call immediately after compute_delta_hpwl() if the move is accepted.
        You must also update pos[moved_idx] in your numpy array.
        )doc");

    m.def("count_overlap", &count_overlap,
        py::arg("positions"), py::arg("sizes"), py::arg("hard_idx"),
        py::arg("threshold") = 0.0040f,
        "Count overlapping hard macro pairs using relative-area threshold. "
        "Returns (count, total_area).");
}