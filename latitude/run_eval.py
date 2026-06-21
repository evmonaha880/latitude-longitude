"""
Headless eval driver — runs the HUD tasks programmatically (no CLI / no TTY).
Routes the model through HUD's gateway via the stored API key.

Usage:
  ~/latitude/.venv312/bin/python run_eval.py smoke      # 1 task, 1 rollout
  ~/latitude/.venv312/bin/python run_eval.py [model] [group]
"""
import asyncio, sys
from hud.agents import create_agent
import env

MODEL = "claude-haiku-4-5"


async def run_one(agent, task, group):
    job = await task.run(agent, group=group, max_concurrent=group, rollout_timeout=180)
    return getattr(job, "reward", None)


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "smoke":
        agent = create_agent(MODEL)
        t = next(x for x in env.tasks if x.id == "cowos")
        r = await run_one(agent, t, 1)
        print(f"SMOKE cowos reward={r}")
        return
    model = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "all" else MODEL
    group = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    agent = create_agent(model)
    print(f"model={model} group={group}\n" + "-" * 40)
    out = {}
    for t in env.tasks:
        try:
            r = await run_one(agent, t, group)
        except Exception as e:
            r = f"ERR {type(e).__name__}: {e}"
        out[t.id] = r
        print(f"{t.id:<10} reward={r}")
    print("-" * 40)
    nums = [v for v in out.values() if isinstance(v, (int, float))]
    if nums:
        print(f"avg reward={sum(nums)/len(nums):.2f}  (target band 0.20–0.50)")


if __name__ == "__main__":
    asyncio.run(main())
