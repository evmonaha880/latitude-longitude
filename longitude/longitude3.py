"""
Longitude3 — the 3-chip TRILEMMA (H100 + B200 + TPU), A+ model.

Isolated from the proven 2-chip longitude.py. Goal: a GENUINE three-way decision
where each chip wins on a distinct axis, and the agent is rewarded for POSITIONING
across all three BEFORE a packaging crunch hits — not reacting after.

Three chips, three distinct jobs:
  - B200 : best $/H100e (cheapest per unit of compute) — the value king WHEN you can
           get it. But it's the scarce flagship: a weekly availability cap that is
           loose in calm and COLLAPSES in the crunch (CoWoS scarcity + demand fleeing
           rationed H100 floods onto B200). Pre-buy it while it's available.
  - H100 : the abundant, uncapped workhorse — the incumbent / status-quo a buyer runs
           today. Priciest per H100e, so never strictly optimal, but ALWAYS available:
           the fill-in when the scarce B200 lane is shut.
  - TPU  : crunch-immune, cheap, instant-on — the escape hatch. Two frictions: a weekly
           on-demand QUOTA cap, and a QUALIFICATION LEAD (you can only draw quota some
           weeks after you commit). Qualify early or the hatch is shut when you need it.

Why these mechanics (the "act before the crowd" thesis, MODELED not asserted):
  - B200 availability is a SCHEDULED exogenous squeeze (in_shock → tighter cap). This is
    the single-agent-correct way to model a market-wide demand flood: one constrained
    buyer doesn't move the B200 market, so the flood is an event to ANTICIPATE, not one
    your own H100 choices drive (that endogenous coupling would be a multi-agent model).
  - TPU qualification lead makes "qualify early to bank capacity" TRUE in the engine
    (~5 compressed weeks; grounds the real CUDA→JAX/XLA migration, 3–6 mo scaled).
  - Both are HARD constraints (min() on fills) — no soft penalty to game, no cheat. The
    only "exploit" is pre-positioning before the crunch, which is exactly the skill.

Run:  ~/latitude/.venv312/bin/python ~/longitude/longitude3.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------
# Chips. shock_factor scales how hard the silicon crunch hits this chip
# (0 = immune). Costs are in the model's H100=2500 unit basis, set so $/H100e
# matches the real ordering: B200 cheapest, H100 middle, TPU also cheap.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Chip:
    name: str
    h100e: float
    cost: float
    lead: int
    holding: float
    shock_factor: float   # multiplier on the crunch's cap-hit + price-spike (0 = immune)
    switch_cost: float

# Grounded calibration (2026 on-demand rental, see provenance + web reconcile):
# $/H100e ordering TPU < B200 < H100 (TPU is CHEAP, ~4x perf/$ for inference).
#                      h100e   cost   lead  hold  shockF  switch
H100 = Chip("H100",    1.00,   2500.,   2,  120.,  1.00,      0.)   # priciest/H100e, abundant incumbent
B200 = Chip("B200",    2.53,   4360.,   3,  180.,  1.33,      0.)   # best $/H100e, scarce flagship. shockF=1.33 > H100's 1.00 ON PURPOSE:
                                                                   #   Blackwell is the most CoWoS-L-intensive part, so a packaging crunch hits it HARDEST —
                                                                   #   1.33x scales the ration (45%->60%) AND the price spike (3.0x->3.66x) on top of the 35->12/wk
                                                                   #   availability cap. This is the engine reason you PRE-BUY B200 in calm (1.0x) before the squeeze.
TPU  = Chip("TPU",     0.93,   1200.,   1,  110.,  0.00,      0.)   # cheap, immune, instant-on — but quota-capped + qual lead

CHIPS = [H100, B200, TPU]
BY_NAME = {c.name: c for c in CHIPS}


def dollars_per_h100e(c: Chip) -> float:
    return c.cost / c.h100e


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------

@dataclass
class Scenario:
    horizon: int = 26
    demand_mean: float = 40.0
    demand_std: float = 12.0
    r: float = 4000.0
    g: float = 900.0
    K: float = 5000.0

    shock: bool = False
    shock_start: int = 8
    shock_len: int = 10
    shock_cap_hit: float = 0.45      # H100 baseline ration; scaled per-chip by shock_factor
    shock_price_mult: float = 3.0    # H100 baseline price spike; scaled per-chip
    shock_lead_bump: int = 4         # extra weeks of lead for exposed chips in-window

    tpu_weekly_cap: float = 20.0     # max H100e of TPU you can spin up per week (on-demand quota)
    tpu_qual_lead: int = 6           # weeks from first TPU commit until quota is drawable (qualification)

    # B200 availability (H100e/wk). Scheduled squeeze: loose in calm, collapses in the
    # crunch — the demand flood (everyone fleeing rationed H100) + CoWoS scarcity.
    b200_cap_calm: float = 35.0
    b200_cap_crunch: float = 12.0

    seed: int = 7

    def in_shock(self, w: int) -> bool:
        return self.shock and self.shock_start <= w < self.shock_start + self.shock_len

    def cap_hit(self, c: Chip, w: int) -> float:
        if not self.in_shock(w):
            return 0.0
        return min(0.9, self.shock_cap_hit * c.shock_factor)

    def price_mult(self, c: Chip, w: int) -> float:
        if not self.in_shock(w):
            return 1.0
        return 1.0 + (self.shock_price_mult - 1.0) * c.shock_factor

    def lead_at(self, c: Chip, w: int) -> int:
        return c.lead + (self.shock_lead_bump if (self.in_shock(w) and c.shock_factor > 0) else 0)

    def b200_cap(self, w: int) -> float:
        return self.b200_cap_crunch if self.in_shock(w) else self.b200_cap_calm

    def demand_trace(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        d = rng.normal(self.demand_mean, self.demand_std, self.horizon)
        return np.clip(np.round(d), 0, None).astype(int)


# --------------------------------------------------------------------------
# Simulator — pooled H100e demand side, per-chip supply side. Orders = dict.
# --------------------------------------------------------------------------

class Sim:
    def __init__(self, sc: Scenario):
        self.sc = sc
        self.demand = sc.demand_trace()
        self.week = 0
        self.pipeline = {c.name: {} for c in CHIPS}
        self.on_hand = float(round(sc.demand_mean * (H100.lead + 1)))
        self.adopted = {c.name: False for c in CHIPS}
        self.tpu_qual_week = None       # week of first TPU commit (starts the qualification clock)
        self.profit = 0.0
        self.log = []

    @property
    def in_transit(self) -> float:
        return sum(sum(p.values()) for p in self.pipeline.values())

    @property
    def inv_position(self) -> float:
        return self.on_hand + self.in_transit

    def tpu_ready(self, w: int) -> bool:
        return self.tpu_qual_week is not None and (w - self.tpu_qual_week) >= self.sc.tpu_qual_lead

    def step(self, orders: dict):
        sc, w = self.sc, self.week
        orders = {c.name: max(0, int(orders.get(c.name, 0))) for c in CHIPS}

        received = 0.0
        for c in CHIPS:
            received += self.pipeline[c.name].pop(w, 0.0)
        self.on_hand += received

        order_cost = 0.0
        filled_mix = {c.name: 0.0 for c in CHIPS}
        for c in CHIPS:
            q = orders[c.name]
            if q <= 0:
                continue
            # Placing ANY TPU order is the qualifying commit — starts the lead clock
            # even though it delivers nothing until qualified.
            if c.name == "TPU" and self.tpu_qual_week is None:
                self.tpu_qual_week = w

            unit_cost = c.cost * sc.price_mult(c, w)
            filled = int(round(q * (1.0 - sc.cap_hit(c, w))))
            if c.name == "B200":                      # scheduled availability squeeze
                filled = min(filled, int(sc.b200_cap(w) / c.h100e))
            if c.name == "TPU":                       # quota cap AND must be qualified
                filled = min(filled, int(sc.tpu_weekly_cap / c.h100e)) if self.tpu_ready(w) else 0

            if filled > 0:                            # cost only on actual delivery
                arrive = w + sc.lead_at(c, w)
                self.pipeline[c.name][arrive] = self.pipeline[c.name].get(arrive, 0.0) + filled * c.h100e
                order_cost += unit_cost * filled + sc.K
                filled_mix[c.name] += filled * c.h100e
                if not self.adopted[c.name]:
                    order_cost += c.switch_cost
                    self.adopted[c.name] = True

        demand = float(self.demand[w])
        sold = min(demand, self.on_hand)
        short = demand - sold
        self.on_hand -= sold

        holding = H100.holding * self.on_hand
        period = sc.r * sold - holding - sc.g * short - order_cost
        self.profit += period
        self.log.append(dict(week=w, orders=orders, filled=filled_mix, demand=demand,
                             sold=sold, short=short))
        self.week += 1
        return period

    def done(self) -> bool:
        return self.week >= self.sc.horizon


Policy = Callable[[Sim], dict]


def run_sim(sc: Scenario, policy: Policy) -> Sim:
    s = Sim(sc)
    while not s.done():
        s.step(policy(s))
    return s


def run(sc: Scenario, policy: Policy) -> float:
    return run_sim(sc, policy).profit


def do_nothing(_: Sim) -> dict:
    return {}


def cascade_orders(sim: Sim, S: float, use_tpu: bool, qual_week: int) -> dict:
    """Top inventory position up to S, filling cheapest-available-first:
       TPU (quota-capped, must be pre-qualified) -> B200 (scheduled availability cap)
       -> H100 (abundant, uncapped). `qual_week` lets the policy place the TPU
       qualifying commit early, before a demand gap even exists."""
    sc, w = sim.sc, sim.week
    orders: dict = {}
    # pre-qualify TPU: commit by qual_week so the lead clock is running before the crunch
    if use_tpu and sim.tpu_qual_week is None and w >= qual_week:
        orders["TPU"] = 1

    gap = max(0.0, S - sim.inv_position)
    if gap <= 0:
        return orders

    if use_tpu and sim.tpu_ready(w):
        cap = int(sc.tpu_weekly_cap / TPU.h100e)
        tu = min(math.ceil(gap / TPU.h100e), cap)
        if tu > 0:
            orders["TPU"] = max(orders.get("TPU", 0), tu)
            gap -= tu * TPU.h100e
    if gap > 0:                                   # B200 next: best $/H100e, capped by availability
        bcap = int(sc.b200_cap(w) / B200.h100e)
        bu = min(math.ceil(gap / B200.h100e), bcap)
        if bu > 0:
            orders["B200"] = bu
            gap -= bu * B200.h100e
    if gap > 0:                                   # H100 fills the rest — abundant, uncapped
        orders["H100"] = math.ceil(gap / H100.h100e)
    return orders


def anticipatory(S_base: int, S_shock: int, prebuild: int,
                 use_tpu: bool, qual_week: int) -> Policy:
    """Build to S; ramp to S_shock starting `prebuild` weeks before the shock. Fills via
    the TPU->B200->H100 cascade, pre-qualifying TPU at `qual_week`."""
    def p(sim: Sim) -> dict:
        sc, w = sim.sc, sim.week
        ramping = sc.shock and (sc.shock_start - prebuild) <= w < (sc.shock_start + sc.shock_len)
        S = S_shock if ramping else S_base
        return cascade_orders(sim, S, use_tpu, qual_week)
    return p


def naive_no_tpu(S: int) -> Policy:
    """Competent-but-static buyer who never touches TPU: fixed base-stock, B200->H100
    only. The reward anchor — how far a no-escape-hatch policy lands below optimal."""
    return lambda sim: cascade_orders(sim, S, use_tpu=False, qual_week=0)


def brute_force(sc: Scenario, nvidia_only: bool) -> tuple[float, dict]:
    """Best anticipatory play. nvidia_only forbids TPU entirely, so we can measure what
    ACCESS TO THE TPU LANE (qualified early) is worth."""
    best, cfg = float("-inf"), {}
    use_tpu = not nvidia_only
    s_base_rng = range(80, 220, 20)
    s_shock_rng = range(80, 460, 30) if sc.shock else [0]
    pre_rng = range(0, 12, 3) if sc.shock else [0]
    qual_rng = ([0, 3, 6] if (use_tpu and sc.shock) else [0])
    for sb in s_base_rng:
        for ss in s_shock_rng:
            for pb in pre_rng:
                for qw in qual_rng:
                    prof = run(sc, anticipatory(sb, max(ss, sb), pb, use_tpu, qw))
                    if prof > best:
                        best, cfg = prof, dict(S_base=sb, S_shock=max(ss, sb), prebuild=pb,
                                               qual_week=qw, tpu=use_tpu)
    return best, cfg


def normalized_reward(sc: Scenario, profit: float) -> float:
    floor = run(sc, do_nothing)
    ceiling, _ = brute_force(sc, nvidia_only=False)
    if ceiling <= floor:
        return 0.0
    return float(np.clip((profit - floor) / (ceiling - floor), 0.0, 1.0))


# --------------------------------------------------------------------------
# Self-check — does the three-way race emerge, and does pre-positioning pay?
# --------------------------------------------------------------------------

def _mix(sc: Scenario, cfg: dict) -> dict:
    s = run_sim(sc, anticipatory(cfg["S_base"], cfg["S_shock"], cfg["prebuild"],
                                 cfg["tpu"], cfg["qual_week"]))
    return {k: sum(r["filled"][k] for r in s.log) for k in ("H100", "B200", "TPU")}


def main():
    sc0 = Scenario()
    print("=" * 96)
    print("LONGITUDE3 — 3-chip trilemma (A+): scheduled B200 squeeze + TPU qualification lead")
    print("-" * 96)
    print("$/H100e:  " + "  ".join(f"{c.name} ${dollars_per_h100e(c):,.0f}" for c in CHIPS))
    print(f"caps: TPU {int(sc0.tpu_weekly_cap)} H100e/wk (qual lead {sc0.tpu_qual_lead}wk)   "
          f"B200 {int(sc0.b200_cap_calm)}/wk calm -> {int(sc0.b200_cap_crunch)}/wk crunch (the demand flood)")
    print("-" * 96)

    for label, sc in [("CALM market", Scenario(shock=False)),
                      ("PACKAGING CRUNCH", Scenario(shock=True))]:
        floor = run(sc, do_nothing)
        nv, nv_cfg = brute_force(sc, nvidia_only=True)
        full, full_cfg = brute_force(sc, nvidia_only=False)
        mix = _mix(sc, full_cfg)
        naive = run(sc, naive_no_tpu(130))
        print(f"\n{label}")
        print(f"  do-nothing floor              ${floor:>12,.0f}")
        print(f"  best NVIDIA-ONLY (no TPU)     ${nv:>12,.0f}   {nv_cfg}")
        print(f"  best WITH TPU lane            ${full:>12,.0f}   {full_cfg}")
        print(f"  >> TPU lane worth             ${full - nv:>12,.0f}")
        print(f"  optimal chip mix (H100e):     H100={mix['H100']:.0f}  B200={mix['B200']:.0f}  TPU={mix['TPU']:.0f}")
        print(f"  naive 'no TPU, static' play   ${naive:>12,.0f}   -> reward {normalized_reward(sc, naive):.2f}")

    # Anticipation: what is pre-qualifying TPU worth vs waiting until the crunch hits?
    crunch = Scenario(shock=True)
    early, _ = brute_force(crunch, nvidia_only=False)        # free to qualify early
    late = float("-inf")
    for sb in range(80, 220, 20):
        for ss in range(80, 460, 30):
            for pb in range(0, 12, 3):
                late = max(late, run(crunch, anticipatory(sb, max(ss, sb), pb, True,
                                                          crunch.shock_start)))   # qualify only at onset
    print("\n" + "-" * 96)
    print(f"ANTICIPATION  early-qualify ${early:,.0f}  vs  qualify-at-crunch ${late:,.0f}"
          f"   -> pre-qualifying the TPU lane is worth ${early - late:,.0f}")
    print("=" * 96)
    print("Works if: CALM leans B200 (value king); CRUNCH is a real 3-way scramble (B200 squeezed,")
    print("H100 fills, TPU escapes); naive no-TPU collapses to a low reward; pre-qualifying pays.")

    # Guard: the brute-force ceiling must dominate any feasible cascade policy, or the
    # normalized reward would silently clip a beating run to 1.00. This caught the
    # engine/env TPU gap-fill gating bug (engine filled the gap with TPU during the
    # pre-qualification weeks, which step() then dropped — inflating the apparent
    # optimum's TPU lane while starving B200/H100; fixed 2026-06-21 by gating the
    # gap-fill on sim.tpu_ready(w), matching the env). The env greedy heuristic is
    # exactly anticipatory(S=100), and the brute_force grid includes S_base=100, so
    # the ceiling can never legitimately fall below the exact-arithmetic greedy.
    print("\n" + "-" * 96)
    for label, sc in [("CALM", Scenario(shock=False)), ("CRUNCH", Scenario(shock=True))]:
        ceiling, _ = brute_force(sc, nvidia_only=False)
        greedy = run(sc, anticipatory(100, 100, 0, True, 0))
        assert ceiling >= greedy, (
            f"{label}: brute_force ceiling ${ceiling:,.0f} < feasible greedy ${greedy:,.0f} "
            f"-> reward would falsely clip to 1.00 (TPU gating regression?)")
        print(f"GUARD ok  {label:7s} ceiling ${ceiling:>12,.0f} >= greedy ${greedy:>12,.0f}")
    print("(Note: env self-check rounds inv_position to 0.1 H100e in the observation layer,")
    print(" so its faithful crunch profit ~$1,211,159 can sit a hair over this exact-arithmetic")
    print(" ceiling — a <0.25% numerical artifact of that rounding, NOT the qualification bug.)")


if __name__ == "__main__":
    main()
