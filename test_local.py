"""Local test runner for orbit-wars agents."""
import sys
import math
import argparse
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

sys.path.insert(0, ".")


# Simple baseline agent from the tutorial
def nearest_planet_sniper(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not targets:
        return moves
    for mine in my_planets:
        nearest = min(targets, key=lambda t: (mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
        ships_needed = max(nearest.ships + 1, 20)
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])
    return moves


def run_game(agents, *, n_players=2, verbose=True):
    env = make("orbit_wars", debug=False)
    results = env.run(agents)
    final = results[-1]
    steps = len(results)

    if verbose:
        print(f"  Steps: {steps}/500")
        for i, s in enumerate(final):
            print(f"  Player {i}: reward={s.reward:+.0f}  status={s.status}")
    return [s.reward for s in final], steps


def run_n_games(agents, n, *, verbose=False):
    wins = [0] * len(agents)
    draws = 0
    for i in range(n):
        rewards, steps = run_game(agents, verbose=verbose)
        if all(r == rewards[0] for r in rewards):
            draws += 1
        else:
            winner = rewards.index(max(rewards))
            wins[winner] += 1
        if verbose:
            print(f"  Game {i+1}/{n} done")
    return wins, draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=["self", "sniper", "random"], default="self")
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()

    from main import agent as producer

    opponents = {
        "self": producer,
        "sniper": nearest_planet_sniper,
        "random": "random",
    }
    opponent = opponents[args.opponent]
    agents = [producer, opponent]

    print(f"Producer Hybrid v4  vs  {args.opponent}  ({args.games} game(s))")
    print("-" * 50)

    if args.games == 1:
        rewards, steps = run_game(agents, verbose=True)
    else:
        wins, draws = run_n_games(agents, args.games)
        total = args.games
        print(f"  Producer wins : {wins[0]}/{total} ({100*wins[0]/total:.0f}%)")
        print(f"  {args.opponent:8s} wins : {wins[1]}/{total} ({100*wins[1]/total:.0f}%)")
        if draws:
            print(f"  Draws         : {draws}/{total}")


if __name__ == "__main__":
    main()
