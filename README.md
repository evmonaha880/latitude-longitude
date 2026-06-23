# Latitude × Longitude

**RL environments for the million-dollar compute call.**

**[▶ Pitch deck](https://evmonaha880.github.io/latitude-longitude/pitch.html)** · **[Judge brief](https://evmonaha880.github.io/latitude-longitude/)** · **[Live on HUD](https://hud.ai/environments/610f6fb5)**

Interpreting and predicting are basically solved. Deciding and acting — under fire, with real money on the line — is where AI's value is heading. These two composable HUD environments grade exactly that: the GPU buy-and-hedge calls a compute team makes every week, through real supply-chain crises instead of calm demand.

Built solo for the HUD Frontier RSI RL Environments Hackathon (June 2026).

## The two environments

**Latitude — the timing question.** *When and how much* do you buy of a single SKU (H100) across a 26-week horizon, through six real disruption shocks: a CoWoS packaging crunch, an HBM shortage, export controls, a power-deployment ceiling, a viral demand spike, and a compound cascade. Graded on realized profit vs. a brute-forced optimum.

**Longitude — the substitution question.** *What* do you deploy when your top chip gets rationed. A three-chip trilemma: H100 (abundant), B200 (best value, squeezed hardest in a crunch), TPU (crunch-immune but quota-capped, 6-week qualification lead). Same shock, shared substrate.

**Why two.** On a packaging ration, perfect buy-*timing* on one chip still loses money (−$117,600). Adding B200 is the lifeline (+$160,014). Opening the TPU lane on top is the multiplier (+$1,208,193). Timing is insufficient; substitution is the lever — proven in dollars on the same crisis.

## Results (reproducible)

| Environment | Model | Hero metric |
|---|---|---|
| **Latitude** | RL-trained Qwen3-4B | CoWoS crunch **reward 1.000**; solves 5 of 6 shocks |
| **Longitude** | Frontier Claude Sonnet 4.6 (no training) | Same crunch **reward 1.000** (full run 0.996 ±0.004) |

Latitude's compound-cascade shock scores 0.000 — not a bug, the measured generalization frontier. The env shows exactly where it breaks, repeatably. That's the v2 training target.

Longitude's agent independently found the winning policy: qualify TPU on day 0, pre-buy B200 while calm, dodge the H100 spike during the crunch ($1,723 vs $6,305 per H100e — 73% saved), wind down clean. Full decision log: [`longitude/crunch_rollout_trace_2026-06-21.json`](longitude/crunch_rollout_trace_2026-06-21.json).

## Status

- **Latitude is deployed live on HUD:** https://hud.ai/environments/610f6fb5
- **Longitude** runs locally via `hud eval` against `longitude3_env.py`; the reproducible reward-1.000 rollout trace is included above.

## Judge brief

Self-contained HTML walkthrough, served live at **[evmonaha880.github.io/latitude-longitude](https://evmonaha880.github.io/latitude-longitude/)** (source: [`web/judge-brief.html`](web/judge-brief.html) — zero external dependencies, opens offline).

## Run it

```bash
# engines are deterministic and brute-forceable — no soft penalties, no cheats
python latitude/compute.py        # Latitude: six shocks, profit vs optimum
python longitude/longitude3.py    # Longitude: three-chip trilemma economics

# live rollout (frontier model via HUD)
hud eval longitude/.hud_eval.toml --yes
```

Calibration sources (SemiAnalysis, Omdia, SEC filings, ERCOT queue) and assumptions are documented in the engine docstrings and the judge brief.
