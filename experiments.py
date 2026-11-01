#!/usr/bin/env python3
"""Reproducible CPU-only synthetic experiments for the verification-control draft.

These experiments test the stated mathematical model; they are not evaluations of
an LLM framework. Outputs are written next to this file under results/.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEED = 20261101


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def chain_fixed_point(g: float, rho: float) -> float:
    return g / (g + rho - g * rho)


def e1_chain(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    rows = []
    grid_g = np.linspace(0.02, 0.30, 8)
    grid_rho = np.linspace(0.0, 0.95, 9)
    n = 200_000
    for g in grid_g:
        for rho in grid_rho:
            m = chain_fixed_point(float(g), float(rho))
            # Start at stationarity, then apply one transition. This estimates
            # the invariant marginal without a long burn-in near the slow edge.
            x = rng.random(n) < m
            own_failure = rng.random(n) < g
            missed = x & (rng.random(n) >= rho)
            y = own_failure | missed
            empirical = float(y.mean())
            se = math.sqrt(max(m * (1.0 - m), 1e-15) / n)
            rows.append(
                {
                    "g": float(g),
                    "rho": float(rho),
                    "n": n,
                    "theory": m,
                    "empirical": empirical,
                    "absolute_error": abs(empirical - m),
                    "mc_se": se,
                }
            )

    ablations = []
    for length in (1, 5, 25, 100, 500):
        for rho in (0.0, 0.2, 0.6):
            g = 0.1
            n_path = 50_000
            state = np.zeros(n_path, dtype=bool)
            for _ in range(length):
                state = (rng.random(n_path) < g) | (
                    state & (rng.random(n_path) >= rho)
                )
            expected = 0.0
            for _ in range(length):
                expected = g + (1 - g) * (1 - rho) * expected
            ablations.append(
                {
                    "chain_length": length,
                    "g": g,
                    "rho": rho,
                    "empirical_final_corruption": float(state.mean()),
                    "finite_horizon_expectation": expected,
                }
            )

    return rows, ablations


def topology_matrix(k: int, name: str, rng: np.random.Generator) -> np.ndarray:
    p = np.zeros((k, k), dtype=float)
    if name == "chain":
        for i in range(k):
            p[i, max(0, i - 1)] = 1.0
    elif name == "random":
        p = rng.random((k, k))
        p /= p.sum(axis=1, keepdims=True)
    elif name == "hub":
        hub = k // 2
        for i in range(k):
            p[i, hub if i != hub else (0 if k > 1 else hub)] = 1.0
    elif name == "cycle":
        for i in range(k):
            p[i, (i - 1) % k] = 1.0
    else:
        raise ValueError(name)
    return p


def simulate_roles(
    g: np.ndarray,
    catch: np.ndarray,
    p: np.ndarray,
    sizes: np.ndarray,
    rng: np.random.Generator,
    steps: int = 450,
    burn: int = 250,
) -> np.ndarray:
    k = len(g)
    counts = np.zeros(k, dtype=float)
    collected = np.zeros(k, dtype=float)
    n_collected = 0
    for t in range(steps):
        m = counts / sizes
        inherited = p @ m
        probs = g + (1.0 - g) * (1.0 - catch) * inherited
        counts = rng.binomial(sizes.astype(int), np.clip(probs, 0.0, 1.0))
        if t >= burn:
            collected += counts / sizes
            n_collected += 1
    return collected / n_collected


def e2_roles(rng: np.random.Generator) -> list[dict]:
    rows = []
    for k in (2, 3, 5):
        g = np.linspace(0.04, 0.18, k)
        catch = np.linspace(0.15, 0.65, k)
        for topology in ("chain", "random", "hub"):
            p = topology_matrix(k, topology, rng)
            a = np.diag((1 - g) * (1 - catch)) @ p
            theory = np.linalg.solve(np.eye(k) - a, g)
            for size_name, sizes in (
                ("balanced", np.full(k, 4_000, dtype=int)),
                ("imbalanced", np.array([1_000 * (i + 1) for i in range(k)])),
            ):
                empirical = simulate_roles(g, catch, p, sizes, rng)
                for role in range(k):
                    rows.append(
                        {
                            "k": k,
                            "topology": topology,
                            "population": size_name,
                            "role": role,
                            "size": int(sizes[role]),
                            "theory": float(theory[role]),
                            "empirical": float(empirical[role]),
                            "absolute_error": abs(
                                float(empirical[role] - theory[role])
                            ),
                            "contraction_u": float(
                                np.max((1 - g) * (1 - catch))
                            ),
                        }
                    )
    return rows


def e3_concentration(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    rows = []
    initial_rows = []
    ns = (250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000)
    replicas, steps = 100, 35
    for speed, gval, rhoval in (("fast", 0.08, 0.60), ("slow", 0.02, 0.02)):
        g = np.full(3, gval)
        catch = np.full(3, rhoval)
        p = topology_matrix(3, "cycle", rng)
        a = np.diag((1 - g) * (1 - catch)) @ p
        u = float(np.max((1 - g) * (1 - catch)))
        for n in ns:
            counts = np.zeros((replicas, 3), dtype=np.int64)
            meanfield = np.zeros(3)
            deviations = []
            for t in range(steps):
                means = counts / n
                probs = g + (1 - g) * (1 - catch) * (means @ p.T)
                counts = rng.binomial(n, probs)
                meanfield = g + a @ meanfield
                if t == steps - 1:
                    deviations = np.abs(counts / n - meanfield)
            per_role_mean = deviations.mean(axis=0)
            rows.append(
                {
                    "mixing": speed,
                    "n_per_role": n,
                    "replicas": replicas,
                    "steps": steps,
                    "u": u,
                    "mean_max_abs_deviation": float(np.max(per_role_mean)),
                    "max_role_sd": float(
                        np.max(np.std(deviations, axis=0, ddof=1))
                    ),
                    "theorem_bound": 1.0 / (2 * (1 - u) * math.sqrt(n)),
                }
            )
            # Initial-condition mismatch is deliberately not covered by the
            # same-initial-condition bound; show its deterministic contraction.
            start = np.ones(3)
            mf = np.zeros(3)
            fin = start.copy()
            for _ in range(steps):
                mf = g + a @ mf
                fin = g + a @ fin
            mismatch = float(np.max(np.abs(fin - mf)))
            initial_rows.append(
                {
                    "mixing": speed,
                    "n_per_role": n,
                    "initial_condition_gap_after_35": mismatch,
                    "u_power_35": u**steps,
                    "finite_initial_max": float(start.max()),
                }
            )
    # Least-squares slope is reported separately for each mixing speed.
    for speed in ("fast", "slow"):
        subset = [r for r in rows if r["mixing"] == speed]
        slope = float(
            np.polyfit(
                np.log([r["n_per_role"] for r in subset]),
                np.log([r["mean_max_abs_deviation"] for r in subset]),
                1,
            )[0]
        )
        for row in subset:
            row["loglog_slope_for_mixing"] = slope
    return rows, initial_rows


def level_cost(
    b: float,
    level: int,
    *,
    lam: float = 10.0,
    mu: float = 0.35,
    g_n: float = 0.08,
    g_d: float = 0.08,
    delta: float = 0.25,
) -> float:
    costs = (0.0, 0.7, 0.35, 1.5)
    catches = (0.0, 0.55, 0.75 * (1 - (1 - delta) * b), 0.9)
    g = (1 - b) * g_n + b * g_d
    return costs[level] + lam * (g + (1 - g) * (1 - catches[level]) * mu)


def action_policy(
    beliefs: np.ndarray,
    **kwargs: float,
) -> np.ndarray:
    costs = np.stack(
        [np.array([level_cost(float(b), v, **kwargs) for b in beliefs]) for v in range(4)],
        axis=1,
    )
    return np.argmin(costs, axis=1)


def belief_update(
    b: float, signal: int, alpha: float, beta: float, p: float, q: float
) -> float:
    predicted = b * (1 - beta) + (1 - b) * alpha
    ld = p if signal else (1 - p)
    ln = q if signal else (1 - q)
    den = ld * predicted + ln * (1 - predicted)
    return predicted if den <= 0 else ld * predicted / den


def e4_filter(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    rows = []
    policy_rows = []
    beliefs = np.linspace(0, 1, 1001)
    for p_signal in (0.30, 0.50, 0.70, 0.90, 0.99):
        actions = action_policy(beliefs, g_n=0.08, g_d=0.08, delta=0.20)
        tail = int(actions[-1])
        non_level2 = np.flatnonzero(actions != 2)
        first_tail = int(non_level2[0]) if len(non_level2) else len(actions)
        structure_ok = bool(
            np.all(actions[:first_tail] == 2)
            and np.all(actions[first_tail:] == tail)
        )
        policy_rows.append(
            {
                "signal_p": p_signal,
                "signal_q": 0.02,
                "tail_action": tail,
                "level2_initial_interval_end": float(
                    beliefs[max(0, first_tail - 1)]
                ),
                "grid_points": len(beliefs),
                "initial_interval_constant_tail": structure_ok,
            }
        )
        # Detection delay following an observed transition N -> D. The policy
        # uses filtered belief only; the true regime is used for evaluation.
        alpha, beta, q = 0.03, 0.03, 0.02
        delays = []
        for _ in range(200):
            b = alpha / (alpha + beta)
            delay = 0
            for _t in range(1_000):
                signal = int(rng.random() < p_signal)
                b = belief_update(b, signal, alpha, beta, p_signal, q)
                delay += 1
                if b >= 0.9:
                    break
            delays.append(delay)
        rows.append(
            {
                "p": p_signal,
                "q": q,
                "signal_gap": p_signal - q,
                "alpha": alpha,
                "beta": beta,
                "replicas": len(delays),
                "mean_detection_delay": float(np.mean(delays)),
                "median_detection_delay": float(np.median(delays)),
                "p90_detection_delay": float(np.quantile(delays, 0.90)),
            }
        )
    # Switching-speed and filter-miscalibration ablations.
    for alpha, beta in ((0.005, 0.005), (0.03, 0.03), (0.15, 0.15)):
        for model_p in (0.70, 0.90):
            delays = []
            for _ in range(200):
                b = alpha / (alpha + beta)
                for t in range(1_000):
                    signal = int(rng.random() < 0.90)
                    b = belief_update(b, signal, alpha, beta, model_p, 0.02)
                    if b >= 0.9:
                        break
                delays.append(t + 1)
            rows.append(
                {
                    "p": 0.90,
                    "q": 0.02,
                    "signal_gap": 0.88,
                    "alpha": alpha,
                    "beta": beta,
                    "filter_p": model_p,
                    "replicas": 200,
                    "mean_detection_delay": float(np.mean(delays)),
                    "median_detection_delay": float(np.median(delays)),
                    "p90_detection_delay": float(np.quantile(delays, 0.90)),
                }
            )
    return rows, policy_rows


def lower_hull_error_at_budget(points: list[tuple[float, float]], budget: float) -> float:
    """Best error from a randomized static mixture at the requested spend."""
    ordered = sorted(points)
    best = math.inf
    for cost, error in ordered:
        if abs(cost - budget) < 1e-12:
            best = min(best, error)
    for i, (c1, e1) in enumerate(ordered):
        for c2, e2 in ordered[i + 1 :]:
            if c1 <= budget <= c2 and c2 > c1:
                fraction = (budget - c1) / (c2 - c1)
                best = min(best, e1 + fraction * (e2 - e1))
    return best


def e5_budget(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    rows, ablations = [], []
    n = 300_000
    action_costs = np.array([0.0, 0.7, 0.35, 1.5])
    catches = np.array([0.0, 0.55, 0.75, 0.9])
    for p_signal, delta, mu, g_n, g_d in (
        (0.90, 0.20, 0.35, 0.08, 0.08),
        (0.02, 0.20, 0.35, 0.08, 0.08),
        (0.50, 0.20, 0.35, 0.08, 0.08),
        (0.90, 0.80, 0.35, 0.08, 0.08),
        (0.90, 0.20, 0.10, 0.08, 0.08),
        (0.90, 0.20, 0.35, 0.04, 0.20),
    ):
        q, alpha, beta = 0.02, 0.03, 0.03
        regime = np.zeros(n, dtype=np.int8)
        signals = np.zeros(n, dtype=np.int8)
        for t in range(1, n):
            prior = regime[t - 1]
            regime[t] = (
                1 - prior if rng.random() < (beta if prior else alpha) else prior
            )
            signals[t] = rng.random() < (p_signal if regime[t] else q)
        beliefs = np.empty(n, dtype=float)
        beliefs[0] = alpha / (alpha + beta)
        for t in range(1, n):
            beliefs[t] = belief_update(
                beliefs[t - 1], int(signals[t]), alpha, beta, p_signal, q
            )
        g_t = np.where(regime == 1, g_d, g_n)
        # Select actions from the posterior belief, not the hidden true state.
        belief_g = g_n + beliefs * (g_d - g_n)
        belief_catches = np.tile(catches, (n, 1))
        belief_catches[:, 2] *= 1 - (1 - delta) * beliefs
        belief_error = belief_g[:, None] + (1 - belief_g[:, None]) * (
            1 - belief_catches
        ) * mu
        chosen = np.argmin(belief_error * 10 + action_costs, axis=1)
        rho_t = np.tile(catches, (n, 1))
        rho_t[:, 2] *= np.where(regime == 1, delta, 1.0)
        err_t = g_t[:, None] + (1 - g_t[:, None]) * (1 - rho_t) * mu
        adaptive_spend = float(action_costs[chosen].mean())
        adaptive_error = float(err_t[np.arange(n), chosen].mean())
        static_points = [
            (float(action_costs[v]), float(err_t[:, v].mean())) for v in range(4)
        ]
        matched_error = lower_hull_error_at_budget(static_points, adaptive_spend)
        regime_catches = np.array(
            [
                [0.0, 0.55, 0.75, 0.9],
                [0.0, 0.55, 0.75 * delta, 0.9],
            ]
        )
        regime_generations = np.array([g_n, g_d])[:, None]
        regime_errors = regime_generations + (1 - regime_generations) * (
            1 - regime_catches
        ) * mu
        regime_optima = np.argmin(
            action_costs[None, :] + 10 * regime_errors, axis=1
        )
        rows.append(
            {
                "p": p_signal,
                "q": q,
                "delta": delta,
                "mu": mu,
                "g_n": g_n,
                "g_d": g_d,
                "informative": p_signal > q,
                "n_units": n,
                "adaptive_spend": adaptive_spend,
                "adaptive_error": adaptive_error,
                "matched_static_error": matched_error,
                "error_improvement": matched_error - adaptive_error,
                "normal_optimal_level": int(regime_optima[0]),
                "degraded_optimal_level": int(regime_optima[1]),
                "regime_optimal_levels_differ": int(
                    regime_optima[0] != regime_optima[1]
                ),
            }
        )
    # A single shock is shared by all agents in a cluster at each time step.
    # This creates genuine cross-agent dependence while preserving marginal g.
    for shock in (0.0, 0.02, 0.05):
        g, rho = 0.10, 0.55
        clusters, group_size, horizon = 2_000, 8, 300
        x = np.zeros((clusters, group_size), dtype=bool)
        observations = []
        generation_last = None
        for t in range(horizon):
            shared = rng.random(clusters) < shock
            individual = (g - shock) / (1 - shock)
            own_failure = shared[:, None] | (
                rng.random((clusters, group_size)) < individual
            )
            x = own_failure | (x & (rng.random((clusters, group_size)) >= rho))
            if t >= horizon // 2:
                observations.append(x.mean(axis=1))
            if t == horizon - 1:
                generation_last = own_failure
        corr = np.corrcoef(generation_last, rowvar=False)
        pair_corr = corr[np.triu_indices(group_size, 1)].mean()
        ablations.append(
            {
                "common_shock_probability": shock,
                "marginal_generation_failure": g,
                "catch_rate": rho,
                "clusters": clusters,
                "agents_per_cluster": group_size,
                "empirical_corruption": float(np.mean(observations)),
                "independent_fixed_point": chain_fixed_point(g, rho),
                "within_cluster_generation_failure_correlation": float(pair_corr),
                "scope": "shared-shock cluster; independence theorem does not apply",
            }
        )
    return rows, ablations


def has_four_level_shape(actions: np.ndarray, fragile: int = 2) -> bool:
    tail = int(actions[-1])
    starts = np.flatnonzero(actions != fragile)
    split = int(starts[0]) if len(starts) else len(actions)
    return bool(np.all(actions[:split] == fragile) and np.all(actions[split:] == tail))


def e6_structure(
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict]]:
    grid = np.linspace(0.0, 1.0, 2_001)
    two_level = []
    for _ in range(400):
        g_n = float(rng.uniform(0.01, 0.20))
        g_d = float(rng.uniform(0.01, 0.20))
        rho_f = float(rng.uniform(0.10, 0.60))
        delta = float(rng.uniform(0.05, 0.90))
        rho_r = float(rng.uniform(rho_f + 0.02, 0.98))
        lam = float(rng.uniform(1, 20))
        mu = float(rng.uniform(0.05, 0.90))
        kappa_f = float(rng.uniform(0.05, 1.0))
        kappa_r = float(kappa_f + rng.uniform(0.05, 2.0))
        condition = (1 - delta) * rho_f * (1 - g_d) >= (
            g_d - g_n
        ) * (rho_r - delta * rho_f)
        h = (1 - ((1 - grid) * g_n + grid * g_d)) * (
            rho_r - rho_f * (1 - (1 - delta) * grid)
        )
        brute = bool(np.all(np.diff(h) >= -1e-12))
        costs_f = kappa_f + lam * (
            ((1 - grid) * g_n + grid * g_d)
            + (1 - ((1 - grid) * g_n + grid * g_d))
            * (1 - rho_f * (1 - (1 - delta) * grid))
            * mu
        )
        costs_r = kappa_r + lam * (
            ((1 - grid) * g_n + grid * g_d)
            + (1 - ((1 - grid) * g_n + grid * g_d)) * (1 - rho_r) * mu
        )
        policy = np.where(costs_f <= costs_r, 1, 3)
        two_level.append(
            {
                "g_n": g_n,
                "g_d": g_d,
                "analytic_condition": condition,
                "brute_force_monotone": brute,
                "condition_matches_monotonicity": condition == brute,
                "policy_threshold": bool(
                    np.all(np.diff(policy.astype(float)) >= 0)
                ),
            }
        )

    four_level = []
    for idx in range(100):
        g = float(rng.uniform(0.02, 0.18))
        delta = float(rng.uniform(0.05, 0.75))
        costs = (0.0, float(rng.uniform(0.1, 1.0)), float(rng.uniform(0.05, 0.8)), float(rng.uniform(1.0, 3.0)))
        catches = (0.0, float(rng.uniform(0.25, 0.65)), float(rng.uniform(0.55, 0.85)), float(rng.uniform(0.80, 0.99)))
        lam, mu = float(rng.uniform(2, 20)), float(rng.uniform(0.05, 0.8))
        cols = []
        for b in grid:
            rho = [catches[0], catches[1], catches[2] * (1 - (1 - delta) * b), catches[3]]
            cols.append(
                [
                    costs[v] + lam * (g + (1 - g) * (1 - rho[v]) * mu)
                    for v in range(4)
                ]
            )
        actions = np.argmin(np.asarray(cols), axis=1)
        four_level.append(
            {
                "instance": idx,
                "delta": delta,
                "initial_interval_constant_tail": has_four_level_shape(actions),
                "tail_action": int(actions[-1]),
                "level2_grid_fraction": float(np.mean(actions == 2)),
            }
        )

    general_g = []
    failures = 0
    for idx in range(500):
        g_n = float(rng.uniform(0.01, 0.15))
        g_d = float(rng.uniform(g_n + 0.005, min(0.35, g_n + 0.15)))
        delta = float(rng.uniform(0.02, 0.90))
        level_costs = (
            0.0,
            float(rng.uniform(0.1, 1.0)),
            float(rng.uniform(0.05, 0.8)),
            float(rng.uniform(1.0, 3.0)),
        )
        level_catches = (
            0.0,
            float(rng.uniform(0.25, 0.65)),
            float(rng.uniform(0.55, 0.85)),
            float(rng.uniform(0.80, 0.99)),
        )
        lam, mu = float(rng.uniform(1, 20)), float(rng.uniform(0.05, 0.9))
        actions = []
        for b in grid:
            g_b = (1 - b) * g_n + b * g_d
            catches = (
                level_catches[0],
                level_catches[1],
                level_catches[2] * (1 - (1 - delta) * b),
                level_catches[3],
            )
            stage_cost = [
                level_costs[v]
                + lam * (g_b + (1 - g_b) * (1 - catches[v]) * mu)
                for v in range(4)
            ]
            actions.append(int(np.argmin(stage_cost)))
        # The equal-generation structure permits one initial level-2 interval
        # and a constant tail. Multiple switches/other initial actions refute it.
        switches = int(np.count_nonzero(np.diff(actions)))
        fails = not has_four_level_shape(np.asarray(actions))
        failures += int(fails)
        general_g.append(
            {
                "instance": idx,
                "g_n": g_n,
                "g_d": g_d,
                "four_level_structure_fails": fails,
                "number_of_switches": switches,
            }
        )
    return (
        two_level,
        four_level,
        general_g
        + [
            {
                "summary": "two_level_general_generation",
                "instances": len(general_g),
                "four_level_structure_failures": failures,
                "failure_rate": failures / len(general_g),
            }
        ],
    )


def make_plots(e1: list[dict], e3: list[dict], e4: list[dict], e5: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # E1 fixed-point agreement.
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for rho in (0.0, 0.2375, 0.475, 0.7125, 0.95):
        subset = [r for r in e1 if abs(r["rho"] - rho) < 1e-8]
        ax.plot(
            [r["theory"] for r in subset],
            [r["empirical"] for r in subset],
            "o-",
            ms=3,
            label=rf"$\rho={rho:.2f}$",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="identity")
    ax.set(xlabel="Analytic stationary corruption", ylabel="Monte Carlo estimate", title="E1: serial-chain fixed point")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure_e1_fixed_point.png", dpi=180)
    plt.close(fig)

    # E3 concentration rate.
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for speed, marker in (("fast", "o"), ("slow", "s")):
        subset = [r for r in e3 if r["mixing"] == speed]
        ax.loglog(
            [r["n_per_role"] for r in subset],
            [r["mean_max_abs_deviation"] for r in subset],
            marker + "-",
            label=f"{speed} mixing (slope {subset[0]['loglog_slope_for_mixing']:.3f})",
        )
        ax.loglog(
            [r["n_per_role"] for r in subset],
            [r["theorem_bound"] for r in subset],
            "--",
            alpha=0.7,
            label=f"{speed} bound",
        )
    ax.set(xlabel="Agents per role N (log scale)", ylabel="Mean max-role absolute deviation", title="E3: finite-team concentration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure_e3_concentration.png", dpi=180)
    plt.close(fig)

    # E4 detection delay and E5 matched-budget comparison.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    base = [r for r in e4 if "filter_p" not in r]
    axes[0].plot(
        [r["signal_gap"] for r in base],
        [r["mean_detection_delay"] for r in base],
        "o-",
    )
    axes[0].set(xlabel="Signal gap p-q", ylabel="Mean detection delay (steps)", title="E4: filtering")
    axes[1].bar(
        np.arange(len(e5)),
        [r["error_improvement"] for r in e5],
        color=["#2a9d8f" if r["informative"] else "#777777" for r in e5],
    )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(
        xlabel="Scenario",
        ylabel="Static error - adaptive error at matched spend",
        title="E5: budget-matched comparison",
        xticks=np.arange(len(e5)),
        xticklabels=[f"S{i+1}" for i in range(len(e5))],
    )
    fig.tight_layout()
    fig.savefig(OUT / "figure_e4_e5.png", dpi=180)
    plt.close(fig)


def main() -> None:
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    e1, e1_ab = e1_chain(rng)
    e2 = e2_roles(rng)
    e3, e3_ab = e3_concentration(rng)
    e4, e4_policy = e4_filter(rng)
    e5, e5_ab = e5_budget(rng)
    e6, e6_four_level, e6_ab = e6_structure(rng)
    write_csv("e1_chain.csv", e1)
    write_csv("e1_ablations.csv", e1_ab)
    write_csv("e2_roles.csv", e2)
    write_csv("e3_concentration.csv", e3)
    write_csv("e3_initial_condition.csv", e3_ab)
    write_csv("e4_detection.csv", e4)
    write_csv("e4_policy_structure.csv", e4_policy)
    write_csv("e5_budget.csv", e5)
    write_csv("e5_correlated_failure.csv", e5_ab)
    write_csv("e6_two_level.csv", e6)
    write_csv("e6_four_level_equal_generation.csv", e6_four_level)
    write_csv("e6_general_g.csv", e6_ab)
    make_plots(e1, e3, e4, e5)
    summary = {
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "elapsed_seconds": time.perf_counter() - start,
        "e1_max_absolute_error": max(r["absolute_error"] for r in e1),
        "e1_fraction_within_3_mc_se": float(
            np.mean(
                [
                    r["absolute_error"] <= 3 * r["mc_se"]
                    for r in e1
                    if r["rho"] > 0
                ]
            )
        ),
        "e2_max_absolute_error": max(r["absolute_error"] for r in e2),
        "e3_slopes": {
            speed: next(
                r["loglog_slope_for_mixing"] for r in e3 if r["mixing"] == speed
            )
            for speed in ("fast", "slow")
        },
        "e3_bound_violations": sum(
            r["mean_max_abs_deviation"] > r["theorem_bound"] for r in e3
        ),
        "e4_policy_shape_passes": sum(
            int(r["initial_interval_constant_tail"]) for r in e4_policy
        ),
        "e6_equal_generation_shape_passes": sum(
            int(r["initial_interval_constant_tail"]) for r in e6_four_level
        ),
        "e5_scenarios": e5,
        "e6_general_g_failures": e6_ab[-1]["four_level_structure_failures"],
        "e6_general_g_trials": e6_ab[-1]["instances"],
    }
    summary["elapsed_seconds"] = time.perf_counter() - start
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
