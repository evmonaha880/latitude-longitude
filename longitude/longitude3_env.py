"""
Longitude3 — HUD v6 RL environment (the 3-chip TRILEMMA, A+ model).

Sibling of longitude_env.py (the proven 2-chip wiring). Same pattern, but the
agent's weekly action is THREE numbers — allocate(h100, b200, tpu) — so it must
position across all three lanes BEFORE a packaging crunch hits, not react after.
Pure HUD wiring around longitude3.py's engine (zero engine changes).

Three chips, three jobs (see longitude3.py docstring):
  - B200 : best $/H100e (value king) — scarce flagship, availability collapses in crunch.
  - H100 : abundant, uncapped workhorse — priciest/H100e, the fill-in.
  - TPU  : crunch-immune escape hatch — cheap + instant-on, but quota-capped AND a
           qualification lead (commit early or the hatch is shut when you need it).

Local self-check (no key):  ~/latitude/.venv312/bin/python longitude3_env.py
Run a model:                env -u TINKER_API_KEY ~/latitude/.venv312/bin/hud eval longitude3_env.py claude --max-steps 30
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

from fastmcp import FastMCP

from hud.capabilities import Capability
from hud.environment import Environment

from longitude3 import (
    H100, B200, TPU, CHIPS,
    Scenario, Sim,
    dollars_per_h100e, normalized_reward,
)

# --------------------------------------------------------------------------
# The taskset: calm control + the packaging-crunch trilemma. Mirrors the
# self-check in longitude3.main() exactly (Scenario(shock=False/True)).
# --------------------------------------------------------------------------

SCENARIOS = {
    "calm": dict(
        title="Calm compute market",
        brief=("Steady demand, all three chips available. B200 is the best value "
               "(cheapest per H100e) — no crunch to escape, so leaning B200 wins."),
        scenario=Scenario(shock=False),
    ),
    "crunch": dict(
        title="CoWoS packaging crunch",
        brief=("Advanced-packaging capacity collapses in weeks 8-17. H100 is rationed "
               "and price-spiked; the demand flood off rationed H100 squeezes B200 "
               "availability (~35 -> ~12 H100e/wk). TPU rides a different supply chain "
               "(crunch-immune) but is quota-capped AND takes ~6 weeks to qualify. "
               "Position the TPU lane and pre-buy B200 BEFORE the crunch."),
        scenario=Scenario(shock=True),
    ),
}

# --------------------------------------------------------------------------
# Per-rollout state (fresh subprocess per rollout — module globals are safe).
# --------------------------------------------------------------------------

_SIM: Sim | None = None
_NAME: str = "calm"

server = FastMCP(name="longitude3")
env = Environment(name="longitude3")
_server_task: asyncio.Task | None = None


def _start(name: str) -> None:
    global _SIM, _NAME
    _NAME = name
    _SIM = Sim(SCENARIOS[name]["scenario"])


def _chip_block(c, sc, w) -> dict:
    return {
        "lead_weeks": sc.lead_at(c, w),
        "unit_cost": round(c.cost * sc.price_mult(c, w)),
        "h100e_per_unit": c.h100e,
        "dollars_per_h100e": round(dollars_per_h100e(c)),
    }


def _state() -> dict:
    s = _SIM
    assert s is not None
    sc, w = s.sc, s.week
    in_shock = sc.in_shock(w)

    h100 = _chip_block(H100, sc, w)
    h100.update(rationed=bool(in_shock),
                ration_fraction=round(sc.cap_hit(H100, w), 2),
                weekly_availability_h100e="uncapped")

    b200 = _chip_block(B200, sc, w)
    b200.update(rationed=bool(in_shock),
                ration_fraction=round(sc.cap_hit(B200, w), 2),
                weekly_availability_h100e=round(sc.b200_cap(w)))

    qualified = s.tpu_ready(w)
    if qualified:
        weeks_until = 0
    elif s.tpu_qual_week is None:
        weeks_until = None  # not yet committed — commit to start the clock
    else:
        weeks_until = max(0, sc.tpu_qual_lead - (w - s.tpu_qual_week))
    tpu = _chip_block(TPU, sc, w)
    tpu.update(rationed=False,
               weekly_quota_h100e=sc.tpu_weekly_cap,
               qualified=qualified,
               qual_lead_weeks=sc.tpu_qual_lead,
               qual_committed_week=s.tpu_qual_week,
               weeks_until_qualified=weeks_until)

    return {
        "week": w,
        "weeks_total": sc.horizon,
        "weeks_left": sc.horizon - w,
        "in_shock_now": in_shock,
        "on_hand_h100e": round(s.on_hand, 1),
        "in_transit_h100e": round(s.in_transit, 1),
        "inventory_position_h100e": round(s.inv_position, 1),
        "recent_demand_h100e": [r["demand"] for r in s.log[-4:]],
        "H100": h100,
        "B200": b200,
        "TPU": tpu,
        "profit_so_far": round(s.profit),
        "done": s.done(),
    }


def _observe_impl() -> dict:
    if _SIM is None:
        return {"error": "episode not started"}
    return _state()


def _allocate_impl(h100: int = 0, b200: int = 0, tpu: int = 0) -> dict:
    if _SIM is None:
        return {"error": "episode not started"}
    if _SIM.done():
        return {"error": "episode already complete", **_state()}
    period = _SIM.step({"H100": int(h100), "B200": int(b200), "TPU": int(tpu)})
    r = _SIM.log[-1]
    return {
        "week": r["week"],
        "ordered": r["orders"],
        "filled_h100e": {k: round(v, 1) for k, v in r["filled"].items()},
        "demand_h100e": r["demand"],
        "sold_h100e": round(r["sold"], 1),
        "short_h100e": round(r["short"], 1),
        "period_profit": round(period),
        "state": _state(),
    }


@server.tool
async def observe() -> dict:
    """Return the current state: demand, inventory, and the live cost / lead /
    availability terms for ALL THREE chips (H100, B200, TPU) — including whether
    the TPU lane is qualified yet — so you can decide what to deploy."""
    return _observe_impl()


@server.tool
async def allocate(h100: int = 0, b200: int = 0, tpu: int = 0) -> dict:
    """Order `h100`, `b200`, and `tpu` units of capacity this week, then advance one
    week. Each chip arrives after its own lead time. In a crunch H100 is rationed +
    price-spiked, B200 availability collapses, and TPU is crunch-immune but capped by
    a weekly quota AND must be qualified first (placing any TPU order starts the
    ~6-week qualification clock; it delivers nothing until qualified). Returns the week
    result (what actually filled per chip) and the new state."""
    return _allocate_impl(h100, b200, tpu)


@env.initialize
async def _up() -> None:
    global _server_task
    if _server_task is None:
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        _server_task = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1",
                             port=port, show_banner=False)
        )
        await asyncio.sleep(0.3)
        env.add_capability(Capability.mcp(name="tools", url=f"http://127.0.0.1:{port}/mcp"))


@env.shutdown
async def _down() -> None:
    global _server_task
    if _server_task is not None:
        _server_task.cancel()
        with contextlib.suppress(Exception):
            await _server_task
        _server_task = None


def _briefing(name: str) -> str:
    spec = SCENARIOS[name]
    return (
        f"You run a GPU supply chain and choose WHAT TO DEPLOY across three chips. "
        f"Scenario: {spec['title']}.\n{spec['brief']}\n\n"
        "Demand is measured in H100-equivalent compute (H100e). You serve it from a "
        "mixed fleet — each chip contributes a different amount of H100e per unit:\n"
        "  - H100: 1.00 H100e/unit. Abundant and uncapped, but priciest per H100e; "
        "rationed + price-spiked during a silicon crunch.\n"
        "  - B200: 2.53 H100e/unit. Best value (cheapest per H100e), BUT scarce — a "
        "weekly availability cap that collapses during the crunch. Pre-buy it early.\n"
        "  - TPU: 0.93 H100e/unit. Crunch-immune and cheap, BUT capped by a weekly quota "
        "AND a ~6-week qualification lead — your FIRST tpu order starts that clock and "
        "delivers nothing; quota only becomes drawable once qualified. Commit early.\n\n"
        "You have 26 weeks. Each week: call observe() to read demand, inventory, and the "
        "live cost / lead / availability terms for all three chips, then "
        "allocate(h100=..., b200=..., tpu=...) to procure capacity and advance one week. "
        "Idle capacity costs holding; running short loses high-margin revenue and incurs "
        "penalties. Maximize total profit across all 26 weeks. Keep ordering each week "
        "until the episode is done. Start by calling observe()."
    )


def _make_template(name: str):
    async def task():
        _start(name)
        _ = yield _briefing(name)
        sim = _SIM
        assert sim is not None
        while not sim.done():        # auto-close any weeks left unmanaged
            sim.step({})
        yield normalized_reward(SCENARIOS[name]["scenario"], sim.profit)
    task.__name__ = name
    return env.template()(task)


TASKS_BY_NAME = {name: _make_template(name) for name in SCENARIOS}
tasks = [tmpl() for tmpl in TASKS_BY_NAME.values()]


# --------------------------------------------------------------------------
# Self-check — drive the tool loop (observe -> allocate) WITHOUT the MCP server,
# proving the wiring runs end-to-end and emits a defensible decision. A simple
# cascade heuristic (qualify TPU early -> fill TPU quota -> B200 cap -> H100).
# --------------------------------------------------------------------------

def _selfcheck(name: str, S: int = 100, qual_at: int = 0) -> None:
    import math
    _start(name)
    sc = SCENARIOS[name]["scenario"]
    while True:
        st = _observe_impl()
        if st["done"]:
            break
        w = st["week"]
        h100 = b200 = tpu = 0
        # pre-qualify TPU early: place the committing order before any gap exists
        if st["TPU"]["qual_committed_week"] is None and w >= qual_at:
            tpu = 1
        gap = max(0.0, S - st["inventory_position_h100e"])
        if gap > 0:
            if st["TPU"]["qualified"]:                       # TPU first (cheap, immune)
                cap = int(st["TPU"]["weekly_quota_h100e"] / TPU.h100e)
                tu = min(math.ceil(gap / TPU.h100e), cap)
                tpu = max(tpu, tu)
                gap -= tu * TPU.h100e
            if gap > 0:                                       # B200 next (best $/H100e, capped)
                bcap = int(st["B200"]["weekly_availability_h100e"] / B200.h100e)
                bu = min(math.ceil(gap / B200.h100e), bcap)
                b200 = bu
                gap -= bu * B200.h100e
            if gap > 0:                                       # H100 fills the rest (uncapped)
                h100 = math.ceil(gap / H100.h100e)
        res = _allocate_impl(h100, b200, tpu)
        assert "error" not in res, res
    profit = _SIM.profit
    reward = normalized_reward(sc, profit)
    mix = {k: sum(r["filled"][k] for r in _SIM.log) for k in ("H100", "B200", "TPU")}
    print(f"{name:7s}  profit ${profit:>12,.0f}   reward {reward:.2f}   "
          f"mix H100={mix['H100']:.0f} B200={mix['B200']:.0f} TPU={mix['TPU']:.0f}")


if __name__ == "__main__":
    print("=" * 80)
    print("LONGITUDE3_ENV self-check — tool loop (observe -> allocate), no MCP server")
    print("-" * 80)
    for nm in SCENARIOS:
        _selfcheck(nm)
    print("-" * 80)
    print(f"registered tasks: {[t for t in TASKS_BY_NAME]}")
    print("tools: observe(), allocate(h100, b200, tpu)")
    print("=" * 80)
