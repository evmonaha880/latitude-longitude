"""
Longitude — HUD v6 RL environment (chip-switching / substitution).

Sibling to Latitude's env.py. The difference: the agent's weekly action is TWO
numbers — order_h100 and order_tpu — so it can FAIL OVER to a different chip when
the silicon supply chain seizes. Pure HUD wiring around longitude.py's engine.

Local test (no key):  ~/latitude/.venv312/bin/python test_local.py
Run a model:          env -u TINKER_API_KEY ~/latitude/.venv312/bin/hud eval longitude_env.py claude --max-steps 30
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

from fastmcp import FastMCP

from hud.capabilities import Capability
from hud.environment import Environment

import longitude as lon
from longitude import H100, TPU, Scenario, Sim, normalized_reward

# --------------------------------------------------------------------------
# The taskset: calm control + two silicon shocks of different severity.
# --------------------------------------------------------------------------

SCENARIOS = {
    "calm": dict(
        title="Calm compute market",
        brief="Steady demand, H100 cheap and available. No reason to switch chips.",
        scenario=Scenario(shock=False),
    ),
    "crunch": dict(
        title="CoWoS packaging crunch",
        brief=("Advanced-packaging capacity collapses. H100 orders are rationed ~45%, "
               "prices triple, and lead times blow out. TPU (a different supply chain) "
               "is untouched — failing over is the play."),
        scenario=Scenario(shock=True, shock_cap_hit=0.45, shock_price_mult=3.0,
                          shock_lead=6, shock_start=8, shock_len=10),
    ),
    "export": dict(
        title="Export-control shock",
        brief=("New controls strand part of H100 supply: ~20% rationed, +50% price, "
               "a short lead bump. Milder — TPU failover helps but isn't mandatory."),
        scenario=Scenario(shock=True, shock_cap_hit=0.20, shock_price_mult=1.5,
                          shock_lead=4, shock_start=10, shock_len=6),
    ),
}

# --------------------------------------------------------------------------
# Per-rollout state (fresh subprocess per rollout — module globals are safe).
# --------------------------------------------------------------------------

_SIM: Sim | None = None
_NAME: str = "calm"

server = FastMCP(name="longitude")
env = Environment(name="longitude")
_server_task: asyncio.Task | None = None


def _start(name: str) -> None:
    global _SIM, _NAME
    _NAME = name
    _SIM = Sim(SCENARIOS[name]["scenario"])


def _state() -> dict:
    s = _SIM
    assert s is not None
    sc, w = s.sc, s.week
    shock = sc.in_shock(w)
    return {
        "week": w,
        "weeks_total": sc.horizon,
        "weeks_left": sc.horizon - w,
        "on_hand_h100e": round(s.on_hand, 1),
        "in_transit_h100e": round(s.in_transit, 1),
        "inventory_position_h100e": round(s.inv_position, 1),
        "recent_demand_h100e": [r["demand"] for r in s.log[-4:]],
        "H100": {
            "lead_weeks": sc.h100_lead_at(w),
            "unit_cost": round(H100.cost * (sc.shock_price_mult if shock else 1.0)),
            "rationed": bool(shock),
            "ration_fraction": sc.shock_cap_hit if shock else 0.0,
            "h100e_per_unit": H100.h100e,
        },
        "TPU": {
            "lead_weeks": TPU.lead,
            "unit_cost": round(TPU.cost),
            "rationed": False,
            "h100e_per_unit": TPU.h100e,
            "one_time_switch_cost": 0 if s.adopted["TPU"] else round(TPU.switch_cost),
        },
        "profit_so_far": round(s.profit),
        "done": s.done(),
    }


@server.tool
async def observe() -> dict:
    """Return the current state: demand, inventory, and the live cost/lead/ration
    terms for BOTH chips (H100 and TPU) so you can decide what to deploy."""
    if _SIM is None:
        return {"error": "episode not started"}
    return _state()


@server.tool
async def order(h100: int = 0, tpu: int = 0) -> dict:
    """Order `h100` units of H100 and `tpu` units of TPU capacity this week, then
    advance one week. Each chip arrives after its own lead time; H100 is rationed
    and pricier during a silicon shock, TPU is not but costs a one-time switching
    fee the first time you adopt it. Returns the week result and the new state."""
    if _SIM is None:
        return {"error": "episode not started"}
    if _SIM.done():
        return {"error": "episode already complete", **_state()}
    r = _SIM.step(int(h100), int(tpu))
    return {
        "week": r["week"],
        "ordered_h100": r["order_h100"],
        "ordered_tpu": r["order_tpu"],
        "received_h100e": round(r["received"], 1),
        "demand_h100e": r["demand"],
        "sold_h100e": round(r["sold"], 1),
        "short_h100e": round(r["short"], 1),
        "period_profit": round(r["period"]),
        "state": _state(),
    }


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
        f"You run a GPU supply chain and choose WHAT TO DEPLOY. Scenario: {spec['title']}.\n"
        f"{spec['brief']}\n\n"
        "Demand is measured in H100-equivalent compute (H100e). You serve it from a "
        "mixed fleet: each H100 unit gives 1.0 H100e, each TPU unit gives 0.93 H100e. "
        "Two chips to order from each week:\n"
        "  - H100: cheap and fast normally, but rationed + price-spiked during a silicon shock.\n"
        "  - TPU: a different supply chain (never rationed by the silicon shock), but a "
        "longer lead time and a ONE-TIME switching cost the first time you order it.\n\n"
        "You have 26 weeks. Each week: call observe() to read demand, inventory, and the "
        "live cost/lead/ration terms for both chips, then order(h100=..., tpu=...) to procure "
        "capacity and advance one week. Idle capacity costs holding; running short loses "
        "high-margin revenue and incurs penalties. Maximize total profit across all 26 weeks. "
        "Keep ordering each week until the episode is done. Start by calling observe()."
    )


def _make_template(name: str):
    async def task():
        _start(name)
        _ = yield _briefing(name)
        sim = _SIM
        assert sim is not None
        while not sim.done():        # auto-close any weeks left unmanaged
            sim.step(0, 0)
        yield normalized_reward(SCENARIOS[name]["scenario"], sim.profit)
    task.__name__ = name
    return env.template()(task)


TASKS_BY_NAME = {name: _make_template(name) for name in SCENARIOS}
tasks = [tmpl() for tmpl in TASKS_BY_NAME.values()]
