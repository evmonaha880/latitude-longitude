"""
Latitude — HUD v6 RL environment.

The agent runs a GPU supply chain through a disruption. Each episode it manages
26 weeks of procurement: every week it `observe`s the state and `order`s GPU
capacity, trying to maximize profit through the shock. Graded on realized profit,
normalized between a do-nothing floor (0.0) and the brute-forced optimal (1.0).

Architecture (HUD v6, NOT v5 — see hud-skill/SKILL.md):
  - an in-process FastMCP server exposes two tools: observe() and order(qty)
  - one @env.template() per disruption archetype == the taskset (the six shocks)
  - the second yield returns the normalized-profit reward

The simulator core (latitude.py) and compute calibration (compute.py) are
unchanged — this file is pure wiring.

Local test (no cloud/key needed for the tool loop):
  ~/latitude/.venv312/bin/python test_env_local.py
Run against a model (needs HUD gateway / model access):
  ~/latitude/.venv312/bin/hud eval env.py claude --max-steps 30
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

from fastmcp import FastMCP

from hud.capabilities import Capability
from hud.environment import Environment

import latitude as lat
from compute import SCENARIOS, make_scenario, normalized_reward

# --------------------------------------------------------------------------
# Per-rollout state. Each HUD rollout is a fresh subprocess, so module globals
# reset between tasks — safe to hold the live episode here.
# --------------------------------------------------------------------------

_SIM: lat.Simulator | None = None
_NAME: str = "baseline"

server = FastMCP(name="latitude")
env = Environment(name="latitude")
_server_task: asyncio.Task | None = None


def _start_episode(name: str) -> None:
    global _SIM, _NAME
    _NAME = name
    _SIM = lat.Simulator(make_scenario(name))


def _state() -> dict:
    s = _SIM
    assert s is not None
    sc = s.s
    w = s.week
    recent = [r.demand for r in s.log[-4:]]
    return {
        "week": w,
        "weeks_total": sc.horizon,
        "weeks_left": sc.horizon - w,
        "on_hand": s.on_hand,
        "in_transit": sum(s.pipeline.values()),
        "inventory_position": s.inventory_position,
        "current_lead_time_weeks": sc.lead_time_at(w),
        "supply_disrupted": sc.in_shock(w),
        "recent_demand": recent,
        "profit_so_far": round(s.total_profit(), 0),
        "done": s.done(),
    }


@server.tool
async def observe() -> dict:
    """Return the current state of the GPU supply chain for this week."""
    if _SIM is None:
        return {"error": "episode not started"}
    return _state()


@server.tool
async def order(units: int) -> dict:
    """Order `units` of GPU capacity this week (arrives after the lead time),
    then advance one week. Returns the week's result and the new state."""
    if _SIM is None:
        return {"error": "episode not started"}
    if _SIM.done():
        return {"error": "episode already complete", **_state()}
    r = _SIM.step(int(units))
    return {
        "week": r.week,
        "ordered": r.order_qty,
        "received_this_week": r.received,
        "demand": r.demand,
        "sold": r.sold,
        "stockout": r.stockout,
        "period_profit": round(r.period_profit, 0),
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
        f"You run a GPU supply chain. Scenario: {spec['title']}.\n"
        f"{spec['brief']}\n\n"
        "You have 26 weeks. Each week: call observe() to read the state, then "
        "order(units) to procure GPU capacity (it arrives after the lead time) "
        "and advance one week. Capacity left idle costs holding; running short "
        "loses high-margin inference revenue and incurs penalties. Maximize total "
        "profit across all 26 weeks. Keep ordering each week until the episode is "
        "done. Start by calling observe()."
    )


def _make_template(name: str):
    async def task():
        _start_episode(name)
        _ = yield _briefing(name)
        # auto-close any weeks the agent left unmanaged (treated as no-order)
        sim = _SIM
        assert sim is not None
        while not sim.done():
            sim.step(0)
        yield normalized_reward(name, sim.total_profit())
    task.__name__ = name
    return env.template()(task)


# Register one task per disruption archetype — this is the taskset.
TASKS_BY_NAME = {name: _make_template(name) for name in SCENARIOS}

# The taskset HUD runs: instantiate each template into a Task.
tasks = [tmpl() for tmpl in TASKS_BY_NAME.values()]
