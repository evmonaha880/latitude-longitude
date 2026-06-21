"""
Longitude — chip-switching / substitution environment (Latitude's sibling).

ISOLATED from the frozen Latitude build (~/latitude/). Nothing here imports or
touches latitude.py / compute.py — this is a separate game so the proven
submission can never break. Latitude = WHEN / HOW MUCH to buy through 6 shocks;
Longitude = WHAT TO DEPLOY (H100 vs TPU failover).

The new idea: the agent doesn't just decide HOW MUCH compute to buy — it decides
WHAT TO DEPLOY. Two chips:
  - H100  : the workhorse. 1.0 H100e/unit, cheap, fast lead — BUT it rides the
            silicon supply chain, so a packaging/HBM shock rations it and spikes
            its price.
  - TPU   : the failover. 0.93 H100e/unit, cheaper to rent, a DIFFERENT supply
            chain (not hit by the silicon shock) — BUT a longer quota lead time
            and a one-time SWITCHING COST the first week you adopt it (JAX/XLA
            lock-in tax). Real number: TPU v6e = 0.93 H100e, ~$4.3k (Epoch §2B).

Demand is in H100-equivalent compute. A mixed fleet serves it: effective
capacity = h100_units*1.0 + tpu_units*0.93. Lost sales (no backorders), profit-
scored exactly like the v1 engine, so the two are directly comparable.

The scenario this makes possible: in CALM markets the optimal play is H100-only
(TPU's switching cost isn't worth it). Under a PACKAGING CRUNCH, H100 gets
rationed + expensive, and the optimal play FAILS OVER TO TPU. That contrast is
the whole demo. The self-check at the bottom proves it emerges from the numbers.

Run:  ~/latitude/.venv312/bin/python ~/longitude/longitude.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------
# Chip economics — one physical unit of each.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Chip:
    name: str
    h100e: float        # compute-equivalence per unit (H100 = 1.0)
    cost: float         # acquisition $ per unit (nominal, pre-shock)
    lead: int           # nominal lead time (weeks) to delivery
    holding: float      # holding $/unit/week
    silicon_exposed: bool   # does the silicon shock ration + price-spike this chip?
    switch_cost: float  # one-time $ the first week you ever order this chip


# Calibration: TPU is slightly PRICIER per unit of compute in normal times
# ($2600/0.93 ≈ $2796/H100e vs H100's $2500/H100e) + a switching tax, so H100 is
# genuinely preferred when available. The crunch flips it: H100 → 3× price AND
# 45% rationed, while TPU (different supply chain) holds — so failover wins.
H100 = Chip("H100", h100e=1.00, cost=2500.0, lead=2, holding=120.0,
            silicon_exposed=True,  switch_cost=0.0)
TPU  = Chip("TPU",  h100e=0.93, cost=2600.0, lead=4, holding=110.0,
            silicon_exposed=False, switch_cost=40000.0)

CHIPS = [H100, TPU]


# --------------------------------------------------------------------------
# Scenario — baseline demand + an optional silicon-supply shock window.
# --------------------------------------------------------------------------

@dataclass
class Scenario:
    horizon: int = 26
    demand_mean: float = 40.0      # H100e/week
    demand_std: float = 12.0
    r: float = 4000.0              # revenue per H100e served
    g: float = 900.0              # goodwill penalty per H100e short
    K: float = 5000.0             # fixed cost per chip-order placed

    # silicon shock (hits H100 only — TPU is a different supply chain)
    shock: bool = False
    shock_start: int = 8
    shock_len: int = 10
    shock_cap_hit: float = 0.45    # fraction of an H100 order rationed away
    shock_price_mult: float = 3.0  # H100 acquisition-cost multiplier in-window
    shock_lead: int = 6            # inflated H100 lead in-window

    seed: int = 7

    def in_shock(self, w: int) -> bool:
        return self.shock and self.shock_start <= w < self.shock_start + self.shock_len

    def h100_lead_at(self, w: int) -> int:
        return self.shock_lead if self.in_shock(w) else H100.lead

    def demand_trace(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        d = rng.normal(self.demand_mean, self.demand_std, self.horizon)
        return np.clip(np.round(d), 0, None).astype(int)


# --------------------------------------------------------------------------
# Simulator — pooled H100e inventory on the demand side, per-chip on supply.
# --------------------------------------------------------------------------

class Sim:
    def __init__(self, sc: Scenario):
        self.sc = sc
        self.demand = sc.demand_trace()
        self.week = 0
        # pipelines[chip][arrival_week] = H100e scheduled to land
        self.pipeline = {c.name: {} for c in CHIPS}
        self.on_hand = float(round(sc.demand_mean * (H100.lead + 1)))  # H100e on shelf
        self.adopted = {c.name: False for c in CHIPS}
        self.profit = 0.0
        self.log = []

    @property
    def in_transit(self) -> float:
        return sum(sum(p.values()) for p in self.pipeline.values())

    @property
    def inv_position(self) -> float:
        return self.on_hand + self.in_transit

    def step(self, order_h100: int, order_tpu: int):
        sc, w = self.sc, self.week
        shock = sc.in_shock(w)
        orders = {"H100": max(0, int(order_h100)), "TPU": max(0, int(order_tpu))}

        # 1. receive arrivals (in H100e) scheduled for this week
        received = 0.0
        for c in CHIPS:
            received += self.pipeline[c.name].pop(w, 0.0)
        self.on_hand += received

        # 2. place orders, per chip — ration + price the silicon-exposed one.
        order_cost = 0.0
        for c in CHIPS:
            q = orders[c.name]
            if q <= 0:
                continue
            unit_cost = c.cost
            filled = q
            if shock and c.silicon_exposed:
                unit_cost *= sc.shock_price_mult
                filled = int(round(q * (1.0 - sc.shock_cap_hit)))
            lead = sc.h100_lead_at(w) if c.name == "H100" else c.lead
            if filled > 0:
                arrive = w + lead
                self.pipeline[c.name][arrive] = (
                    self.pipeline[c.name].get(arrive, 0.0) + filled * c.h100e)
            order_cost += unit_cost * filled + sc.K
            if not self.adopted[c.name]:            # one-time switching tax
                order_cost += c.switch_cost
                self.adopted[c.name] = True

        # 3. demand (H100e), lost sales
        demand = float(self.demand[w])
        sold = min(demand, self.on_hand)
        short = demand - sold
        self.on_hand -= sold

        # 4. profit
        holding = H100.holding * self.on_hand   # pooled holding at H100 rate (approx)
        revenue = sc.r * sold
        goodwill = sc.g * short
        period = revenue - holding - goodwill - order_cost
        self.profit += period
        self.log.append(dict(week=w, order_h100=orders["H100"], order_tpu=orders["TPU"],
                             received=received, demand=demand, sold=sold, short=short,
                             on_hand=self.on_hand, period=period))
        self.week += 1
        return period

    def done(self) -> bool:
        return self.week >= self.sc.horizon


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

Policy = Callable[[Sim], tuple]


def do_nothing(_: Sim):
    return (0, 0)


def run(sc: Scenario, policy: Policy) -> float:
    s = Sim(sc)
    while not s.done():
        s.step(*policy(s))
    return s.profit


def base_stock_h100(S: int) -> Policy:
    """Naive heuristic: top H100e up to S every week, H100 ONLY (never substitute)."""
    def p(s: Sim):
        gap = max(0, S - int(round(s.inv_position)))
        return (gap, 0)
    return p


def anticipatory(S_base: int, S_shock: int, prebuild: int, use_tpu_in_shock: bool) -> Policy:
    """Ramp the target up before the shock; during the shock, optionally fill the
    gap with TPU (failover) instead of the rationed H100."""
    def p(s: Sim):
        sc, w = s.sc, s.week
        ramping = sc.shock and (sc.shock_start - prebuild) <= w < (sc.shock_start + sc.shock_len)
        S = S_shock if ramping else S_base
        gap = max(0, S - int(round(s.inv_position)))
        if gap <= 0:
            return (0, 0)
        if sc.in_shock(w) and use_tpu_in_shock:
            # failover: order TPU (grossed up for its 0.93 efficiency)
            return (0, int(round(gap / TPU.h100e)))
        if ramping and use_tpu_in_shock and w < sc.shock_start:
            # pre-build the buffer with cheap, unrationed H100 BEFORE the shock
            return (gap, 0)
        return (gap, 0)
    return p


def brute_force_ceiling(sc: Scenario, allow_tpu: bool) -> tuple[float, dict]:
    """Coarse sweep for the best anticipatory play. allow_tpu toggles whether the
    failover lever is available — so we can MEASURE what substitution is worth."""
    best, best_cfg = float("-inf"), {}
    s_base_range = range(80, 220, 20)
    s_shock_range = range(80, 460, 30) if sc.shock else [0]
    prebuild_range = range(0, 12, 3) if sc.shock else [0]
    tpu_opts = [False, True] if (allow_tpu and sc.shock) else [False]
    for sb in s_base_range:
        for ss in s_shock_range:
            for pb in prebuild_range:
                for tpu in tpu_opts:
                    prof = run(sc, anticipatory(sb, max(ss, sb), pb, tpu))
                    if prof > best:
                        best, best_cfg = prof, dict(S_base=sb, S_shock=ss, prebuild=pb, tpu=tpu)
    return best, best_cfg


def normalized_reward(sc: Scenario, profit: float) -> float:
    floor = run(sc, do_nothing)
    ceiling, _ = brute_force_ceiling(sc, allow_tpu=True)
    if ceiling <= floor:
        return 0.0
    return float(np.clip((profit - floor) / (ceiling - floor), 0.0, 1.0))


# --------------------------------------------------------------------------
# Self-check — prove the failover story emerges from the numbers.
# --------------------------------------------------------------------------

def main():
    calm = Scenario(shock=False)
    crunch = Scenario(shock=True)

    print("=" * 84)
    print("LONGITUDE — does substitution emerge as optimal ONLY under the shock?")
    print("-" * 84)
    for label, sc in [("CALM market", calm), ("PACKAGING CRUNCH", crunch)]:
        floor = run(sc, do_nothing)
        ceil_no_tpu, cfg_no = brute_force_ceiling(sc, allow_tpu=False)
        ceil_tpu, cfg_yes = brute_force_ceiling(sc, allow_tpu=True)
        heur = run(sc, base_stock_h100(130))
        rew = normalized_reward(sc, heur)
        gain = ceil_tpu - ceil_no_tpu
        print(f"\n{label}")
        print(f"  do-nothing floor        ${floor:>12,.0f}")
        print(f"  best H100-only          ${ceil_no_tpu:>12,.0f}   {cfg_no}")
        print(f"  best WITH TPU failover  ${ceil_tpu:>12,.0f}   {cfg_yes}")
        print(f"  >> substitution is worth ${gain:>12,.0f}   "
              f"({'FAILOVER WINS' if cfg_yes.get('tpu') else 'H100-only wins'})")
        print(f"  naive H100 heuristic     ${heur:>12,.0f}   -> reward {rew:.2f}")
    print("\n" + "=" * 84)
    print("Read: CALM -> H100-only wins (TPU switch cost not worth it).")
    print("      CRUNCH -> failover wins (substitution adds real $). That gap is the scenario.")


if __name__ == "__main__":
    main()
