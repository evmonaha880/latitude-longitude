"""
Latitude — the COMPUTE reskin.

Dresses the generic single-SKU engine (latitude.py) in GPU supply-chain
clothes: one unit = one block of H100-equivalent capacity. You order GPU
capacity under uncertainty, it arrives after a fab/packaging lead time, it
serves inference demand (revenue), and you eat holding + stockout costs.

Calibration provenance: vault note "Compute Data — Sources & Provenance" §5
(canonical params, Doc3 base). The MAGNITUDES of the six shocks come from §5E
(cap hit / price spike / lead blowout / recovery). They are TIME-COMPRESSED to a
26-week playable episode — the real CoWoS lead blowout is +40-52 wks, which
won't fit a 26-week horizon, so the shape is faithful and the clock is scaled.
HONESTY FLAG: these are calibrated-directional, not slide-ready absolute figures.

Baseline unit economics are tuned to the §5B newsvendor critical ratio
(stockout:holding ≈ 4 for rented GPUs) so the agent's optimal safety stock
behaves like the real thing — over-provisioning bleeds cash, under-provisioning
loses high-margin inference revenue.

Run:  ~/latitude/.venv312/bin/python compute.py
"""

from __future__ import annotations

import numpy as np

from latitude import Scenario, Simulator, base_stock, do_nothing, run, brute_force_S


# --------------------------------------------------------------------------
# Baseline GPU economics — one "unit" = a block of H100e capacity.
# Calibrated to §5B: stockout:holding ≈ 4 (rented GPUs punish over-provisioning).
# --------------------------------------------------------------------------

def gpu_baseline(**overrides) -> Scenario:
    base = dict(
        horizon=26,
        demand_mean=40.0,      # inference demand, GPU-blocks/wk
        demand_std=12.0,       # bursty (BurstGPT CoV high; std bumped vs generic)
        lead_time=2,
        r=4000.0,              # revenue per GPU-block of served inference
        c=2500.0,              # acquisition cost per block
        h=120.0,               # holding $/wk  (~$1.70/hr all-in, §5A H100)
        g=900.0,               # goodwill / SLA-breach penalty per block short
        K=5000.0,              # fixed cost to place a procurement order
        seed=7,
    )
    base.update(overrides)
    return Scenario(**base)


# --------------------------------------------------------------------------
# The six disruption archetypes (§5E) — the taskset of "boss levels".
# Each is the baseline + a supply/demand shock window. Magnitudes directional
# from §5E; window timing compressed to fit the 26-week episode.
# --------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {
    "baseline": dict(
        title="Baseline — calm compute market",
        brief="Steady inference demand, nominal 2-week lead times, no disruption.",
        params=dict(shock=False),
    ),
    "cowos": dict(
        title="(i) CoWoS packaging crunch",
        brief=("Advanced-packaging capacity collapses — the 2023 bottleneck. "
               "Orders are heavily rationed, prices spike, lead times blow out. "
               "S-curve onset, slow recovery."),
        params=dict(shock=True, shock_start=8, shock_len=10,
                    shock_lead_time=6, shock_cap_hit=0.45, shock_price_mult=3.0),
    ),
    "hbm": dict(
        title="(ii) HBM memory sellout",
        brief=("High-bandwidth memory sells out — a step-function supply cut. "
               "Moderate rationing, sharp price step, multi-month lead inflation."),
        params=dict(shock=True, shock_start=9, shock_len=8,
                    shock_lead_time=5, shock_cap_hit=0.28, shock_price_mult=1.65),
    ),
    "export": dict(
        title="(iii) Export-control shock",
        brief=("New export controls strand a chunk of supply. Smaller cap hit, "
               "modest price spike, short lead bump, linear decay back to normal."),
        params=dict(shock=True, shock_start=10, shock_len=6,
                    shock_lead_time=3, shock_cap_hit=0.18, shock_price_mult=1.40),
    ),
    "power": dict(
        title="(iv) Power & grid bottleneck",
        brief=("The 2025-26 defining constraint: you can BUY the GPUs but you "
               "can't POWER them. Grid interconnect queues (IEA: 5-10 yr) cap "
               "ENERGIZED capacity for a long window — owned chips strand as idle "
               "capital. A DEPLOYMENT ceiling (serve-side), not an order shortage: "
               "you cannot pre-build your way out, so timing alone can't save you."),
        params=dict(shock=True, shock_start=8, shock_len=14,
                    shock_lead_time=4, shock_serve_cap=20, shock_price_mult=1.20),
    ),
    "viral": dict(
        title="(v) Viral demand spike",
        brief=("A consumer app goes viral — inference demand jumps ~3x almost "
               "instantly (Dirac-delta), prices spike, supply can't catch up. "
               "Demand-side shock: the shelf empties from the front."),
        params=dict(shock=True, shock_start=10, shock_len=6,
                    shock_lead_time=3, shock_demand_mult=3.0, shock_price_mult=2.0),
    ),
    "agentic": dict(
        title="(vi) Agentic cascade",
        brief=("Agentic workloads compound: token amplification 200x, KV-cache "
               "thrash, demand ~2x AND supply rationed as everyone scrambles. "
               "Geometric, recursive — the hero scenario."),
        params=dict(shock=True, shock_start=9, shock_len=8,
                    shock_lead_time=4, shock_demand_mult=2.0, shock_cap_hit=0.40,
                    shock_price_mult=2.25),
    ),
}


def make_scenario(name: str, seed: int | None = None) -> Scenario:
    """Build the Scenario for a named archetype."""
    spec = SCENARIOS[name]
    sc = gpu_baseline(**spec["params"])
    if seed is not None:
        sc = gpu_baseline(seed=seed, **spec["params"])
    return sc


# --------------------------------------------------------------------------
# Reward normalizer — maps realized profit to 0..1 between a weak anchor
# (do-nothing) and the strong anchor (brute-forced best base-stock).
# This is what HUD grades on; the spanning anchors give within-group spread.
# --------------------------------------------------------------------------

def base_stock_anticipatory(S_base: int, S_shock: int, prebuild: int):
    """Order-up-to that RAMPS the target to S_shock starting `prebuild` weeks
    before the shock and holds it through the window. This is the adaptive lever
    a smart agent uses (pre-build, then draw down) — so it's a far stronger
    ceiling than any fixed base-stock, which an adaptive policy trivially beats."""
    def policy(sim):
        sc, w = sim.s, sim.week
        if sc.shock and (sc.shock_start - prebuild) <= w < (sc.shock_start + sc.shock_len):
            S = S_shock
        else:
            S = S_base
        return max(0, S - sim.inventory_position)
    return policy


def brute_force_anticipatory(scenario) -> float:
    """Coarse sweep over (S_base, S_shock, prebuild) → best profit. The ceiling.
    For a no-shock scenario this collapses to the best fixed base-stock."""
    best = float("-inf")
    s_base_range = range(60, 230, 10)
    # wide ranges: a smart agent front-loads HARD before a shock to dodge the
    # in-window cap, so the ceiling must allow aggressive pre-build to stay above
    # what an adaptive agent can reach.
    s_shock_range = range(80, 620, 20) if scenario.shock else [0]
    prebuild_range = range(0, 16, 2) if scenario.shock else [0]
    for sb in s_base_range:
        for ss in s_shock_range:
            for pb in prebuild_range:
                p = run(scenario, base_stock_anticipatory(sb, max(ss, sb), pb)).total_profit()
                if p > best:
                    best = p
    return best


def anchors(name: str, seed: int | None = None) -> tuple[float, float]:
    """(floor, ceiling) profit for a scenario: do-nothing and the ANTICIPATORY
    optimum (pre-build-aware). The stronger ceiling pushes adaptive agents off
    the 1.0 rail and into a learnable reward band with within-group variance."""
    sc = make_scenario(name, seed)
    floor = run(sc, do_nothing).total_profit()
    # ceiling = the better of the anticipatory optimum and a FINE fixed sweep,
    # so the coarse anticipatory grid can never undercut the simple optimum.
    _, fixed_best = brute_force_S(sc, lo=40, hi=260)
    ceiling = max(brute_force_anticipatory(sc), fixed_best)
    return floor, ceiling


def normalized_reward(name: str, profit: float, seed: int | None = None) -> float:
    """clip((profit - floor) / (ceiling - floor), 0, 1)."""
    floor, ceiling = anchors(name, seed)
    if ceiling <= floor:
        return 0.0
    return float(np.clip((profit - floor) / (ceiling - floor), 0.0, 1.0))


# --------------------------------------------------------------------------
# Self-check — run all six, show that the shock bites and the anchors span.
# --------------------------------------------------------------------------

def main():
    print("=" * 92)
    print("LATITUDE — COMPUTE ENV self-check: six shocks, anchors span, shock bites")
    print("  reward = where a fixed S=130 heuristic lands between do-nothing(0) and optimal(1)")
    print("-" * 92)
    print(f"{'scenario':<10}{'title':<34}{'do-nothing':>12}{'heuristic':>12}"
          f"{'optimal':>12}{'reward':>9}")
    print("-" * 92)
    for name, spec in SCENARIOS.items():
        sc = make_scenario(name)
        floor, ceiling = anchors(name)
        heur = run(sc, base_stock(130)).total_profit()
        rew = normalized_reward(name, heur)
        short = spec["title"][:32]
        print(f"{name:<10}{short:<34}{floor:>12,.0f}{heur:>12,.0f}"
              f"{ceiling:>12,.0f}{rew:>9.2f}")
    print("=" * 92)
    print("Read: optimal > heuristic > do-nothing on every row (anchors span = trainable);")
    print("reward in (0,1) = the heuristic leaves room an RL agent can learn to close.")


if __name__ == "__main__":
    main()
