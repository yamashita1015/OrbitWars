"""Ring conquest multiplier vs baseline (ring disabled) comparison."""
import sys, dataclasses
sys.path.insert(0, ".")

import argparse
from kaggle_environments import make
import torch

from main import (
    ProducerLiteConfig, CONFIG_4P, ProducerLiteRuntime,
    largest_initial_player_count, _config_for,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


def make_agent(*, enable_ring: bool):
    runtime = ProducerLiteRuntime()

    def _agent(obs):
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        player_id = int(player)
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        mem = runtime.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
            mem.movement = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        config = dataclasses.replace(config, enable_ring_conquest=enable_ring)
        with torch.no_grad():
            from main import run_turn
            row = run_turn(
                obs_tensors, config=config,
                player_count=int(mem.cached_player_count), memory=mem,
            )
        mem.last_sparse_action_row = row
        return sparse_action_row_to_moves(row, obs, player_id=player_id)

    return _agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=30)
    args = parser.parse_args()

    ring_agent = make_agent(enable_ring=True)
    base_agent = make_agent(enable_ring=False)

    wins = [0, 0]
    draws = 0
    for i in range(args.games):
        env = make("orbit_wars", debug=False)
        results = env.run([ring_agent, base_agent])
        final = results[-1]
        rewards = [s.reward for s in final]
        if all(r == rewards[0] for r in rewards):
            draws += 1
        else:
            wins[rewards.index(max(rewards))] += 1
        print(f"  Game {i+1}/{args.games}: {rewards}", flush=True)

    N = args.games
    print(f"\n=== Ring conquest ON vs OFF ({N} games) ===")
    print(f"  Ring ON   wins: {wins[0]}/{N} ({100*wins[0]/N:.0f}%)")
    print(f"  Ring OFF  wins: {wins[1]}/{N} ({100*wins[1]/N:.0f}%)")
    print(f"  Draws         : {draws}/{N}")
