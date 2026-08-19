"""
IK solver benchmark — measures time-to-convergence across realistic scenarios.

Reports per (solver, scenario):
  - convergence latency: p50 / p95 / p99 / max (ms)
  - iterations to converge: p50 / p95
  - convergence rate (% of trials that hit tolerance)
  - final pos_err / rot_err distribution

Scenarios:
  A. servo_steady:  target moves 1mm/tick, warm start (typical teleop cost)
  B. movel_near:    fresh target 5-10 cm from current pose, warm start
  C. movel_cold:    fresh target 30-50 cm, q_init = Q_COLD_INIT
                    (non-singular alternate pose, no warm-start info)
  D. near_singular: elbow nearly straight (sigma_min ≈ 0.009, below
                    the 0.02 damping threshold) — tests DLS damping ramp

All initial configurations are non-singular. The URDF neutral (zero joints)
is deliberately avoided — it sits at a hard singularity for the A1 arm
(sigma_min ≈ 5e-8) and would produce misleading "IK failed" results.

All measurements use unified tolerance: pos_tol=1e-4 m, rot_tol=1e-3 rad.

Usage:
    python -m benchmarks.bench_ik           # run full benchmark
    python -m benchmarks.bench_ik --quick   # 100 samples/scenario (fast)
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onero_sdk.ik_solver import IKSolver, PinkIKSolver
from onero_sdk.model_registry import _find_urdf

POS_TOL = 1e-4
ROT_TOL = 1e-3

# Non-singular home pose (sigma_min ≈ 0.5, well-conditioned).
Q_HOME = np.array([0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0])

# Alternate non-singular pose used for "cold start" scenarios — represents
# starting IK when the arm is not at the home config (e.g. after a jog or
# manual reposition). NOT the URDF neutral (zero joints), which sits at a
# hard shoulder-elbow singularity for the A1 arm (sigma_min ≈ 5e-8).
Q_COLD_INIT = np.array([0.3, -0.6, 0.2, -1.2, -0.2, 0.7, 0.1])


@dataclass
class Trial:
    latency_ms: float
    iters: int
    pos_err: float
    rot_err: float
    converged: bool


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
    return xs[k]


def summarize(name: str, trials: list[Trial]) -> dict:
    ok = [t for t in trials if t.converged]
    fail = [t for t in trials if not t.converged]

    lat_ok = [t.latency_ms for t in ok]
    iters_ok = [t.iters for t in ok]
    pos_err_ok = [t.pos_err for t in ok]
    rot_err_ok = [t.rot_err for t in ok]

    fail_pos = [t.pos_err for t in fail]
    fail_rot = [t.rot_err for t in fail]

    return {
        "name": name,
        "n": len(trials),
        "n_ok": len(ok),
        "converge_rate": len(ok) / max(1, len(trials)),
        # Timing on converged trials only (real IK cost).
        "lat_p50": pct(lat_ok, 0.50),
        "lat_p95": pct(lat_ok, 0.95),
        "lat_p99": pct(lat_ok, 0.99),
        "lat_max": max(lat_ok) if lat_ok else float("nan"),
        "iters_p50": pct(iters_ok, 0.50),
        "iters_p95": pct(iters_ok, 0.95),
        "pos_err_ok_p95": pct(pos_err_ok, 0.95),
        "rot_err_ok_p95": pct(rot_err_ok, 0.95),
        # Non-converged: residual error distribution (how bad was the failure).
        "fail_pos_p50": pct(fail_pos, 0.50) if fail else 0.0,
        "fail_pos_max": max(fail_pos) if fail else 0.0,
        "fail_rot_p50": pct(fail_rot, 0.50) if fail else 0.0,
    }


def print_report(rows: list[dict]) -> None:
    print("Table 1 — Converged trials only (IK timing you can trust)")
    hdr = ["scenario / solver", "conv %", "n_ok",
           "p50 ms", "p95 ms", "p99 ms", "max ms",
           "iters p50", "iters p95",
           "pos_err p95 (mm)", "rot_err p95 (mrad)"]
    widths = [30, 7, 5, 8, 8, 8, 8, 10, 10, 18, 20]
    line = "  ".join(f"{h:<{w}}" for h, w in zip(hdr, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        if r["n_ok"] == 0:
            cells = [r["name"], f"{r['converge_rate']*100:.1f}",
                     r["n_ok"], "-", "-", "-", "-", "-", "-", "-", "-"]
        else:
            cells = [
                r["name"],
                f"{r['converge_rate']*100:.1f}",
                r["n_ok"],
                f"{r['lat_p50']:.3f}",
                f"{r['lat_p95']:.3f}",
                f"{r['lat_p99']:.3f}",
                f"{r['lat_max']:.3f}",
                f"{r['iters_p50']:.0f}",
                f"{r['iters_p95']:.0f}",
                f"{r['pos_err_ok_p95']*1000:.4f}",
                f"{r['rot_err_ok_p95']*1000:.4f}",
            ]
        print("  ".join(f"{str(c):<{w}}" for c, w in zip(cells, widths)))

    print()
    print("Table 2 — Non-converged trials (residual error, IK output NOT trustworthy)")
    hdr2 = ["scenario / solver", "n_fail",
            "fail_pos p50 (mm)", "fail_pos max (mm)", "fail_rot p50 (mrad)"]
    widths2 = [30, 7, 20, 20, 22]
    line2 = "  ".join(f"{h:<{w}}" for h, w in zip(hdr2, widths2))
    print(line2)
    print("-" * len(line2))
    for r in rows:
        n_fail = r["n"] - r["n_ok"]
        if n_fail == 0:
            continue
        cells = [
            r["name"],
            n_fail,
            f"{r['fail_pos_p50']*1000:.3f}",
            f"{r['fail_pos_max']*1000:.3f}",
            f"{r['fail_rot_p50']*1000:.3f}",
        ]
        print("  ".join(f"{str(c):<{w}}" for c, w in zip(cells, widths2)))


def fk_pose(solver: IKSolver, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = solver.forward_kinematics(q)
    return T[:3, 3].copy(), T[:3, :3].copy()


def run_scenario_servo(solver, solver_kind: str, n_trials: int,
                       ref_solver: IKSolver, rng: np.random.Generator) -> list[Trial]:
    """Warm-start streaming: target moves 1mm/tick from prev pose."""
    pos, rot = fk_pose(ref_solver, Q_HOME)
    if solver_kind == "pink":
        solver.reset(Q_HOME.copy())
    else:
        solver.q = Q_HOME.copy()
    trials = []
    q_prev = Q_HOME.copy()
    for i in range(n_trials):
        step = rng.normal(0, 0.001, 3)
        target_pos = pos + step
        pos = target_pos
        # Ensure warm start actually reused
        t = time.perf_counter()
        if solver_kind == "pink":
            _, info = solver.solve(
                target_pos, rot, dt=0.005,
                max_iter=10, pos_tol=POS_TOL, rot_tol=ROT_TOL,
                return_info=True,
            )
        else:
            _, info = solver.solve_ik(
                target_pos, rot, q_init=q_prev,
                max_iter=200, pos_tol=POS_TOL, rot_tol=ROT_TOL,
                return_info=True,
            )
        dt_ms = (time.perf_counter() - t) * 1000
        trials.append(Trial(dt_ms, info["iters"], info["pos_err"],
                            info["rot_err"], info["converged"]))
        q_prev = solver.q if solver_kind != "pink" else solver._configuration.q.copy()
    return trials


def run_scenario_movel(solver, solver_kind: str, n_trials: int,
                       ref_solver: IKSolver, rng: np.random.Generator,
                       jump_range: tuple[float, float], cold: bool) -> list[Trial]:
    """Fresh targets. warm: q_init at home. cold: q_init at Q_COLD_INIT
    (non-singular but different from home — simulates jogged/manual repose).
    """
    home_pos, home_rot = fk_pose(ref_solver, Q_HOME)
    q_init_pose = Q_COLD_INIT if cold else Q_HOME
    trials = []
    for i in range(n_trials):
        direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction) + 1e-9
        dist = rng.uniform(*jump_range)
        target_pos = home_pos + direction * dist
        axis = rng.normal(0, 1, 3); axis /= np.linalg.norm(axis) + 1e-9
        angle = rng.uniform(-0.5, 0.5)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        target_rot = R_delta @ home_rot

        if solver_kind == "pink":
            solver.reset(q_init_pose.copy())
        else:
            solver.q = q_init_pose.copy()

        t = time.perf_counter()
        if solver_kind == "pink":
            _, info = solver.solve(
                target_pos, target_rot, dt=0.005,
                max_iter=500, pos_tol=POS_TOL, rot_tol=ROT_TOL,
                return_info=True,
            )
        else:
            _, info = solver.solve_ik(
                target_pos, target_rot, q_init=q_init_pose,
                max_iter=500, pos_tol=POS_TOL, rot_tol=ROT_TOL,
                return_info=True,
            )
        dt_ms = (time.perf_counter() - t) * 1000
        trials.append(Trial(dt_ms, info["iters"], info["pos_err"],
                            info["rot_err"], info["converged"]))
    return trials


def run_scenario_singular(solver, solver_kind: str, n_trials: int,
                          ref_solver: IKSolver, rng: np.random.Generator) -> list[Trial]:
    """Start with elbow near-straight (sigma_min ≈ 0.009, well below the
    0.02 damping threshold). Tests DLS damping ramp and Pink QP behaviour
    when the Jacobian is rank-deficient."""
    # J3 near zero → arm nearly fully extended → elbow singularity.
    q_near_sing = np.array([0.0, -0.5, 0.0, -0.1, 0.0, 0.5, 0.0])
    pos_sing, rot_sing = fk_pose(ref_solver, q_near_sing)

    trials = []
    for i in range(n_trials):
        step = rng.normal(0, 0.003, 3)
        target_pos = pos_sing + step
        if solver_kind == "pink":
            solver.reset(q_near_sing.copy())
        else:
            solver.q = q_near_sing.copy()
        t = time.perf_counter()
        if solver_kind == "pink":
            _, info = solver.solve(
                target_pos, rot_sing, dt=0.005,
                max_iter=100, pos_tol=POS_TOL, rot_tol=ROT_TOL,
                return_info=True,
            )
        else:
            _, info = solver.solve_ik(
                target_pos, rot_sing, q_init=q_near_sing,
                max_iter=500, pos_tol=POS_TOL, rot_tol=ROT_TOL,
                return_info=True,
            )
        dt_ms = (time.perf_counter() - t) * 1000
        trials.append(Trial(dt_ms, info["iters"], info["pos_err"],
                            info["rot_err"], info["converged"]))
    return trials


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="100 samples/scenario")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n = 100 if args.quick else 1000

    urdf = str(_find_urdf("A1", "r"))
    pink_ik = PinkIKSolver(urdf)
    dls_ik = IKSolver(urdf)
    ref = IKSolver(urdf)

    rng = np.random.default_rng(args.seed)

    print(f"# Onero A1 IK benchmark")
    print(f"platform    : {platform.system()} {platform.machine()}, {platform.python_version()}")
    print(f"URDF        : {urdf}")
    print(f"trials/scen : {n}")
    print(f"tolerances  : pos={POS_TOL} m, rot={ROT_TOL} rad")
    print()

    scenarios = [
        ("A. servo_steady",  lambda s, k: run_scenario_servo(s, k, n, ref, rng)),
        ("B. movel_near",    lambda s, k: run_scenario_movel(s, k, n, ref, rng, (0.05, 0.10), cold=False)),
        ("C. movel_cold",    lambda s, k: run_scenario_movel(s, k, n, ref, rng, (0.30, 0.50), cold=True)),
        ("D. near_singular", lambda s, k: run_scenario_singular(s, k, n, ref, rng)),
    ]

    rows = []
    for scen_name, run_fn in scenarios:
        # Reset RNG state so both solvers see the same targets
        rng.bit_generator.state = np.random.default_rng(args.seed).bit_generator.state
        # Warm-up
        _ = run_fn(pink_ik, "pink")[:3]
        rng.bit_generator.state = np.random.default_rng(args.seed).bit_generator.state
        pink_trials = run_fn(pink_ik, "pink")

        rng.bit_generator.state = np.random.default_rng(args.seed).bit_generator.state
        _ = run_fn(dls_ik, "dls")[:3]
        rng.bit_generator.state = np.random.default_rng(args.seed).bit_generator.state
        dls_trials = run_fn(dls_ik, "dls")

        rows.append(summarize(f"{scen_name}  · Pink QP", pink_trials))
        rows.append(summarize(f"{scen_name}  · DLS",     dls_trials))

    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
