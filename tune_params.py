"""Parameter sweep for ProducerLiteConfig.

Compares each candidate config (tuned agent) against the default config agent.
Runs N_GAMES per config, alternating which side plays the tuned agent.
"""
import sys
import dataclasses

sys.path.insert(0, ".")

from main import ProducerLiteConfig, make_agent
from kaggle_environments import make

N_GAMES = 6  # must be even (3 as P0, 3 as P1)


def run_matchup(cfg_a: ProducerLiteConfig, cfg_b: ProducerLiteConfig, n: int) -> tuple[int, int, int]:
    """Returns (wins_a, wins_b, draws)."""
    wins_a = wins_b = draws = 0
    for i in range(n):
        agent_a = make_agent(cfg_a, comet_targeting=False)
        agent_b = make_agent(cfg_b, comet_targeting=False)
        # Alternate sides to cancel position bias
        if i % 2 == 0:
            agents = [agent_a, agent_b]
            idx_a, idx_b = 0, 1
        else:
            agents = [agent_b, agent_a]
            idx_a, idx_b = 1, 0
        env = make("orbit_wars", debug=False)
        results = env.run(agents)
        final = results[-1]
        ra, rb = float(final[idx_a].reward), float(final[idx_b].reward)
        if ra > rb:
            wins_a += 1
        elif rb > ra:
            wins_b += 1
        else:
            draws += 1
    return wins_a, wins_b, draws


DEFAULT = ProducerLiteConfig()

sweeps: list[tuple[str, list[ProducerLiteConfig]]] = [
    ("roi_threshold", [
        dataclasses.replace(DEFAULT, roi_threshold=v)
        for v in [1.2, 1.35, 1.5, 1.65, 1.8]
    ]),
    ("horizon", [
        dataclasses.replace(DEFAULT, horizon=v)
        for v in [14, 16, 18, 22]
    ]),
    ("min_ships_to_launch", [
        dataclasses.replace(DEFAULT, min_ships_to_launch=v)
        for v in [2.0, 4.0, 6.0, 8.0]
    ]),
]

print(f"{'param':<24} {'value':>8} {'W':>4} {'L':>4} {'D':>4}  win%")
print("-" * 56)

for param_name, configs in sweeps:
    for cfg in configs:
        val = getattr(cfg, param_name)
        wa, wb, d = run_matchup(cfg, DEFAULT, N_GAMES)
        pct = 100 * wa / max(wa + wb + d, 1)
        marker = " ← default" if val == getattr(DEFAULT, param_name) else ""
        print(f"  {param_name:<22} {val:>8.2f} {wa:>4} {wb:>4} {d:>4}  {pct:>4.0f}%{marker}")
    print()
