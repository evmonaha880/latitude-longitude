"""
emit_trace.py — run the REAL Latitude engine and emit a per-week episode trace
the Disruption Desk console replays. This is what turns the console from
"MOCK DATA — directional only" into a real, deterministic replay of the actual
simulator: every number on screen comes from latitude.py / compute.py.

Three real policies span the scoreboard:
  - floor      : do-nothing            (never orders — the weak anchor)
  - agent      : base-stock S=130      (the untrained heuristic the agent starts at)
  - optimal    : anticipatory optimum  (pre-build-aware ceiling — the best play)

The "agent" line is the honest current position BEFORE training. When a trained
model's per-week trace exists, drop it in under key "agent" with the same shape
and the green line moves toward optimal — no console change needed.

Run:  ~/latitude/.venv312/bin/python emit_trace.py [scenario]   (default: cowos)
Writes: ~/latitude-web/trace.json
"""
from __future__ import annotations

import json
import os
import sys

from latitude import Scenario, Simulator, base_stock, do_nothing, run
from compute import (
    SCENARIOS,
    make_scenario,
    anchors,
    normalized_reward,
    base_stock_anticipatory,
)

OUT = os.path.expanduser("~/latitude-web/trace.json")


def best_anticipatory(sc: Scenario):
    """Re-run compute's coarse search but RETURN the winning (S_base,S_shock,prebuild)
    so we can replay that exact optimal policy and capture its per-week log."""
    best, best_params = float("-inf"), (130, 130, 0)
    s_base_range = range(60, 230, 10)
    s_shock_range = range(80, 620, 20) if sc.shock else [0]
    prebuild_range = range(0, 16, 2) if sc.shock else [0]
    for sb in s_base_range:
        for ss in s_shock_range:
            for pb in prebuild_range:
                p = run(sc, base_stock_anticipatory(sb, max(ss, sb), pb)).total_profit()
                if p > best:
                    best, best_params = p, (sb, max(ss, sb), pb)
    return best_params


def trace_policy(sc: Scenario, policy):
    """Step the sim manually so we capture richer per-week state than WeekResult
    alone (in-transit pipeline, inventory position, lead time, cumulative profit)."""
    sim = Simulator(sc)
    weeks = []
    cum = 0.0
    cum_demand = 0
    cum_sold = 0
    while not sim.done():
        w = sim.week
        start_inv = sim.on_hand
        inv_pos_before = sim.inventory_position
        order = policy(sim)
        r = sim.step(order)
        in_transit = sim.inventory_position - sim.on_hand  # still-in-pipeline after this order
        cum += r.period_profit
        cum_demand += r.demand
        cum_sold += r.sold
        weeks.append({
            "w": w,
            "shock": sc.in_shock(w),
            "lead": sc.lead_time_at(w),
            "start_inv": start_inv,
            "received": r.received,
            "order": r.order_qty,
            "in_transit": int(round(in_transit)),
            "demand": r.demand,
            "sold": r.sold,
            "stockout": r.stockout,
            "end_inv": r.end_inv,
            "revenue": round(r.revenue, 1),
            "holding": round(r.holding_cost, 1),
            "goodwill": round(r.goodwill_cost, 1),
            "order_cost": round(r.order_cost, 1),
            "period_profit": round(r.period_profit, 1),
            "cum_profit": round(cum, 1),
            "fill": round(cum_sold / cum_demand, 4) if cum_demand else 1.0,
        })
    return weeks, cum


def build(scenario_name: str) -> dict:
    spec = SCENARIOS[scenario_name]
    sc = make_scenario(scenario_name)
    floor, ceiling = anchors(scenario_name)

    sb, ss, pb = best_anticipatory(sc)
    opt_policy = base_stock_anticipatory(sb, ss, pb)

    floor_weeks, floor_total = trace_policy(sc, do_nothing)
    agent_weeks, agent_total = trace_policy(sc, base_stock(130))
    opt_weeks, opt_total = trace_policy(sc, opt_policy)

    reward = normalized_reward(scenario_name, agent_total)

    return {
        "scenario": scenario_name,
        "title": spec["title"],
        "brief": spec["brief"],
        "horizon": sc.horizon,
        "shock": {
            "active": bool(sc.shock),
            "start": sc.shock_start,
            "len": sc.shock_len,
            "lead_nominal": sc.lead_time,
            "lead_shock": sc.shock_lead_time,
            "cap_hit": sc.shock_cap_hit,
            "price_mult": sc.shock_price_mult,
            "demand_mult": sc.shock_demand_mult,
        },
        "econ": {"r": sc.r, "c": sc.c, "h": sc.h, "g": sc.g, "K": sc.K, "seed": sc.seed},
        "anchors": {"floor": round(floor, 1), "ceiling": round(ceiling, 1)},
        "optimal_params": {"S_base": sb, "S_shock": ss, "prebuild": pb},
        "reward": round(reward, 3),
        "totals": {
            "floor": round(floor_total, 1),
            "agent": round(agent_total, 1),
            "optimal": round(opt_total, 1),
        },
        "lines": {
            "floor": floor_weeks,
            "agent": agent_weeks,
            "optimal": opt_weeks,
        },
        "labels": {
            "floor": "do-nothing",
            "agent": "agent (S=130, untrained)",
            "optimal": "optimal (anticipatory)",
        },
    }


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "cowos"
    data = build(name)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    a = data["anchors"]
    t = data["totals"]
    print(f"wrote {OUT}")
    print(f"scenario   : {data['title']}")
    print(f"horizon    : {data['horizon']} wks   shock w{data['shock']['start']}"
          f"–{data['shock']['start']+data['shock']['len']}  "
          f"lead {data['shock']['lead_nominal']}→{data['shock']['lead_shock']}")
    print(f"floor/ceil : {a['floor']:,.0f} … {a['ceiling']:,.0f}")
    print(f"totals     : floor {t['floor']:,.0f} | agent {t['agent']:,.0f} | "
          f"optimal {t['optimal']:,.0f}")
    print(f"reward     : {data['reward']}  (agent between floor=0 and optimal=1)")


if __name__ == "__main__":
    main()
