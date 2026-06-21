"""
Latitude — a supply-chain simulator that scores a decision-maker on PROFIT
under disruption.

This is the non-LLM core: an e-commerce single-SKU inventory scenario, driven
by a heuristic rule, scored by realized profit (revenue earned minus every cost
paid). Later the LLM replaces the heuristic and we compare its profit to the
heuristic and the optimal.

The name: in film, "latitude" is how much over/under-exposure the stock tolerates
before the image breaks — i.e. how much shock an operation can absorb before it
fails.

Why profit, not cost: see Decisions/2026-06-18-latitude-cost-model.md. Scoring
cost alone (and charging for goods but never crediting the sale) made starving
customers the "optimal" move. Profit fixes that at the root — a stockout costs
you the lost margin automatically.

Run:  ~/latitude/.venv/bin/python latitude.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------
# Scenario configuration  (locked 2026-06-18 — see the decision record)
# --------------------------------------------------------------------------

@dataclass
class Scenario:
    horizon: int = 26              # weeks to simulate

    # demand: Normal(mean, std), clipped >= 0
    demand_mean: float = 40.0
    demand_std: float = 10.0

    # lead time (weeks): nominal, plus an optional disruption window.
    lead_time: int = 2
    shock: bool = False            # if True, the disruption window is active
    shock_start: int = 10          # first week (0-indexed) the window applies
    shock_len: int = 6             # number of weeks it lasts
    shock_lead_time: int = 4       # inflated lead time during the window

    # supply-side disruption levers (the compute-shock dials — see compute.py).
    # All default to no-op so existing scenarios are unchanged.
    shock_cap_hit: float = 0.0     # fraction of an order RATIONED away during window
    shock_price_mult: float = 1.0  # acquisition-cost multiplier during window
    shock_demand_mult: float = 1.0 # demand multiplier during window (viral/agentic)
    shock_serve_cap: int = 0       # max units that can be ENERGIZED/served per week
                                   # during the window, regardless of how many you
                                   # hold (the power/grid deployment ceiling). Owned
                                   # units above the cap strand as idle capital —
                                   # still cost holding, can't serve. 0 = no cap.

    # money
    r: float = 50.0                # retail price per unit SOLD (revenue)
    c: float = 30.0                # purchase cost per unit ORDERED
    h: float = 0.14                # holding cost per unit per week (25%/yr of c)
    g: float = 10.0                # goodwill cost per unit short (churn beyond
                                   # lost margin; the lost margin itself is
                                   # captured automatically as missed revenue)
    K: float = 75.0                # fixed cost per order placed

    seed: int = 7                  # freezes the demand path so every policy faces
                                   # the identical scenario. Monte Carlo varies it.

    def in_shock(self, week: int) -> bool:
        """Is `week` inside the disruption window?"""
        return self.shock and self.shock_start <= week < self.shock_start + self.shock_len

    def lead_time_at(self, week: int) -> int:
        """Lead time experienced by an order *placed* in `week`."""
        if self.in_shock(week):
            return self.shock_lead_time
        return self.lead_time

    def demand_trace(self) -> np.ndarray:
        """The frozen demand path. Same seed -> same path -> fair comparison."""
        rng = np.random.default_rng(self.seed)
        d = rng.normal(self.demand_mean, self.demand_std, self.horizon)
        return np.clip(np.round(d), 0, None).astype(int)


# --------------------------------------------------------------------------
# Per-week accounting
# --------------------------------------------------------------------------

@dataclass
class WeekResult:
    week: int
    start_inv: int
    received: int
    demand: int
    sold: int
    order_qty: int
    end_inv: int
    stockout: int
    revenue: float
    holding_cost: float
    goodwill_cost: float
    order_cost: float
    period_profit: float


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

class Simulator:
    """
    Lost-sales single-SKU inventory model.

    Each week:
      1. receive arrivals scheduled for this week (from the pipeline)
      2. the decision-maker places an order (arrives lead_time weeks later)
      3. demand hits; you sell what you can, the rest is lost (not backordered)
      4. tally profit = revenue - holding - goodwill - order cost
    """

    def __init__(self, scenario: Scenario):
        self.s = scenario
        self.demand = scenario.demand_trace()
        self.reset()

    def reset(self):
        s = self.s
        self.week = 0
        # pipeline[w] = units scheduled to arrive at the start of week w
        self.pipeline: dict[int, int] = {}
        # start with a stocked shelf so week 0 isn't an automatic stockout:
        # roughly one lead-time + review period of mean demand.
        self.on_hand = int(round(s.demand_mean * (s.lead_time + 1)))
        self.log: list[WeekResult] = []

    @property
    def inventory_position(self) -> int:
        """On-hand plus everything already ordered and still in transit."""
        return self.on_hand + sum(self.pipeline.values())

    def step(self, order_qty: int) -> WeekResult:
        s = self.s
        w = self.week

        shock = s.in_shock(w)

        # 1. receive arrivals scheduled for this week
        received = self.pipeline.pop(w, 0)
        start_inv = self.on_hand
        avail = self.on_hand + received   # what's on the shelf to meet demand

        # 2. place the order -> arrives after this week's lead time.
        #    During a supply shock, capacity is RATIONED (you get less than you
        #    ordered) and the units you do get cost more.
        order_qty = max(0, int(order_qty))
        unit_cost = s.c * (s.shock_price_mult if shock else 1.0)
        filled = order_qty
        if shock and s.shock_cap_hit > 0:
            filled = int(round(order_qty * (1.0 - s.shock_cap_hit)))
        if filled > 0:
            arrive = w + s.lead_time_at(w)
            self.pipeline[arrive] = self.pipeline.get(arrive, 0) + filled

        # 3. demand (lost sales): sell what you can, lose the rest.
        #    Demand-side shocks (viral / agentic) inflate the draw in-window.
        demand = int(self.demand[w])
        if shock and s.shock_demand_mult != 1.0:
            demand = int(round(demand * s.shock_demand_mult))
        #    Power/grid shock: an ENERGIZATION ceiling caps how many units you can
        #    actually run this week, no matter how many you hold. Units above the
        #    cap strand (carry forward, keep costing holding) — you can't buy your
        #    way out, so pre-building is punished, not rewarded.
        serveable = avail
        if shock and s.shock_serve_cap > 0:
            serveable = min(avail, s.shock_serve_cap)
        sold = min(demand, serveable)
        stockout = demand - sold
        end_inv = avail - sold

        # 4. profit for the week (you pay for what you actually RECEIVE)
        revenue = s.r * sold
        holding_cost = s.h * end_inv
        goodwill_cost = s.g * stockout
        order_cost = unit_cost * filled + (s.K if order_qty > 0 else 0.0)
        period_profit = revenue - holding_cost - goodwill_cost - order_cost

        # carry forward
        self.on_hand = end_inv
        self.week += 1

        result = WeekResult(
            week=w, start_inv=start_inv, received=received, demand=demand,
            sold=sold, order_qty=order_qty, end_inv=end_inv, stockout=stockout,
            revenue=revenue, holding_cost=holding_cost,
            goodwill_cost=goodwill_cost, order_cost=order_cost,
            period_profit=period_profit,
        )
        self.log.append(result)
        return result

    def done(self) -> bool:
        return self.week >= self.s.horizon

    def total_profit(self) -> float:
        return sum(r.period_profit for r in self.log)

    def fill_rate(self) -> float:
        """Fraction of total demand actually served. The service-level readout."""
        d = sum(r.demand for r in self.log)
        s = sum(r.sold for r in self.log)
        return s / d if d else 1.0


# --------------------------------------------------------------------------
# Policies (the "agents")
# --------------------------------------------------------------------------

# A policy maps the current simulator state -> order quantity for this week.
Policy = Callable[["Simulator"], int]


def do_nothing(_: Simulator) -> int:
    """Baseline: never order. Sells the starting shelf, then stocks out forever."""
    return 0


def fixed_order(qty: int) -> Policy:
    """Dumb constant order — used for the sanity checks."""
    def policy(_: Simulator) -> int:
        return qty
    return policy


def base_stock(S: int) -> Policy:
    """
    Order-up-to-S. Each week top the inventory position back up to S.
    A heuristic, not a trained policy — that's the point.
    """
    def policy(sim: Simulator) -> int:
        return max(0, S - sim.inventory_position)
    return policy


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(scenario: Scenario, policy: Policy) -> Simulator:
    sim = Simulator(scenario)
    while not sim.done():
        sim.step(policy(sim))
    return sim


def breakdown(sim: Simulator) -> dict[str, float]:
    return {
        "profit": sim.total_profit(),
        "revenue": sum(r.revenue for r in sim.log),
        "holding": sum(r.holding_cost for r in sim.log),
        "goodwill": sum(r.goodwill_cost for r in sim.log),
        "order": sum(r.order_cost for r in sim.log),
        "fill_rate": sim.fill_rate(),
        "stockout_units": sum(r.stockout for r in sim.log),
    }


def report(label: str, sim: Simulator) -> None:
    b = breakdown(sim)
    print(
        f"{label:<28} profit=${b['profit']:>8,.0f}  "
        f"(rev ${b['revenue']:>8,.0f} | hold ${b['holding']:>5,.0f} | "
        f"good ${b['goodwill']:>6,.0f} | ord ${b['order']:>8,.0f})  "
        f"fill {b['fill_rate']*100:>5.1f}%"
    )


def brute_force_S(scenario: Scenario, lo: int = 80, hi: int = 320) -> tuple[int, float]:
    """Sweep base-stock S to find the most PROFITABLE. The reward normalizer."""
    best_S, best_profit = lo, float("-inf")
    for S in range(lo, hi + 1):
        p = run(scenario, base_stock(S)).total_profit()
        if p > best_profit:
            best_S, best_profit = S, p
    return best_S, best_profit


def monte_carlo(policy: Policy, shock: bool, n: int = 200, seed0: int = 1000):
    """
    Run the same policy across `n` different demand years (each frozen and
    identical across policies for a given seed). Returns the spread, not one
    lucky number. This is what makes a comparison airtight.
    """
    profits, fills = [], []
    for i in range(n):
        sc = Scenario(shock=shock, seed=seed0 + i)
        sim = run(sc, policy)
        profits.append(sim.total_profit())
        fills.append(sim.fill_rate())
    return np.array(profits), np.array(fills)


# --------------------------------------------------------------------------
# Main — the build sequence, each step with its sanity check
# --------------------------------------------------------------------------

def main():
    calm = Scenario(shock=False)
    storm = Scenario(shock=True)

    print("=" * 84)
    print("STEP 2 — engine sanity check (calm, dumb fixed orders)")
    print("  expect: matched order is most profitable; under-order leaves sales")
    print("  on the table; over-order bleeds holding + tied-up cash")
    print("-" * 84)
    report("fixed 20/wk (under-order)", run(calm, fixed_order(20)))
    report("fixed 40/wk (~matched)",   run(calm, fixed_order(40)))
    report("fixed 80/wk (over-order)", run(calm, fixed_order(80)))

    print()
    print("=" * 84)
    print("STEP 3 — heuristic vs baseline (calm)")
    print("  expect: base-stock makes real money; do-nothing dies after the shelf")
    print("-" * 84)
    report("do-nothing baseline", run(calm, do_nothing))
    report("base-stock S=160",    run(calm, base_stock(160)))

    print()
    print("=" * 84)
    print("STEP 4 — the shock (week 10, lead time 2 -> 4 for 6 weeks)")
    print("  expect: same heuristic earns LESS under disruption (stockouts bite)")
    print("-" * 84)
    report("base-stock S=160  (calm)",  run(calm, base_stock(160)))
    report("base-stock S=160  (storm)", run(storm, base_stock(160)))

    print()
    print("=" * 84)
    print("STEP 5 — brute-force the most profitable S (the reward normalizer)")
    print("-" * 84)
    S_calm, p_calm = brute_force_S(calm)
    S_storm, p_storm = brute_force_S(storm)
    print(f"calm : best S={S_calm:>3}  ->  ${p_calm:,.0f} profit")
    print(f"storm: best S={S_storm:>3}  ->  ${p_storm:,.0f} profit   "
          f"(buffers UP to ride out the longer lead time)")
    print(f"cost of the disruption (each at its own best S): "
          f"${p_calm - p_storm:,.0f}")

    print()
    print("=" * 84)
    print("STEP 6 — Monte Carlo: 200 different demand years, not one lucky run")
    print("  the airtight comparison — report the spread, not a single number")
    print("-" * 84)
    for label, shock in [("CALM ", False), ("STORM", True)]:
        bs = monte_carlo(base_stock(160), shock=shock)
        dn = monte_carlo(do_nothing, shock=shock)
        wins = (bs[0] > dn[0]).mean() * 100
        print(f"[{label}] base-stock S=160:  "
              f"profit ${bs[0].mean():>8,.0f} avg  "
              f"(worst ${bs[0].min():>8,.0f} / best ${bs[0].max():>8,.0f})  "
              f"fill {bs[1].mean()*100:>5.1f}%")
        print(f"         beats do-nothing in {wins:.0f}% of 200 scenarios")
    print("=" * 84)


if __name__ == "__main__":
    main()
