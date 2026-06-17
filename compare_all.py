"""Universal comparison script: test any variant main.py against current main.py."""
import sys, importlib, importlib.util, argparse
from pathlib import Path
from kaggle_environments import make

sys.path.insert(0, str(Path(__file__).parent))


_load_counter = 0

def load_agent(path: str) -> callable:
    """Load agent function from a file path, isolated in its own module namespace."""
    global _load_counter
    _load_counter += 1
    mod_name = f"_orbit_variant_{_load_counter}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod   # register before exec so @dataclass resolves __module__
    spec.loader.exec_module(mod)
    return mod.agent


def run_games(agent_a, agent_b, n: int, *, n_players: int = 2, label_a="A", label_b="B"):
    wins = [0, 0]
    draws = 0
    for i in range(n):
        if n_players == 2:
            agents = [agent_a, agent_b]
        else:
            agents = [agent_a, agent_b, agent_b, agent_b]
        env = make("orbit_wars", debug=False)
        results = env.run(agents)
        final = results[-1]
        rewards = [s.reward for s in final]
        if n_players == 2:
            if rewards[0] == rewards[1]:
                draws += 1
            elif rewards[0] > rewards[1]:
                wins[0] += 1
            else:
                wins[1] += 1
        else:
            if rewards[0] > max(rewards[1], rewards[2], rewards[3]):
                wins[0] += 1
            elif rewards[0] < max(rewards[1], rewards[2], rewards[3]):
                wins[1] += 1
            else:
                draws += 1
        print(f"  Game {i+1}/{n}: rewards={[f'{r:+.0f}' for r in rewards[:2]]}  "
              f"{'A wins' if rewards[0]>rewards[1] else ('B wins' if rewards[1]>rewards[0] else 'draw')}",
              flush=True)
    return wins, draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", help="Path to variant main.py")
    parser.add_argument("--games", "-n", type=int, default=50)
    parser.add_argument("--players", type=int, default=2, choices=[2, 4])
    parser.add_argument("--label", default="variant")
    args = parser.parse_args()

    print(f"Loading baseline: main.py")
    baseline = load_agent("main.py")
    print(f"Loading variant:  {args.variant}")
    variant = load_agent(args.variant)

    print(f"\n=== {args.label} vs baseline  ({args.games} games, {args.players}P) ===")
    wins, draws = run_games(variant, baseline, args.games,
                            n_players=args.players,
                            label_a=args.label, label_b="baseline")
    N = args.games
    label = args.label
    print(f"\n--- Result ---")
    print(f"  {label:20s} wins: {wins[0]}/{N} ({100*wins[0]/N:.1f}%)")
    print(f"  {'baseline':20s} wins: {wins[1]}/{N} ({100*wins[1]/N:.1f}%)")
    print(f"  {'draws':20s}     : {draws}/{N}")


if __name__ == "__main__":
    main()
