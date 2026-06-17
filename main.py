from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

# Make the sibling ``orbit_lite`` package importable wherever this file runs:
# loaded in place, dropped at a submission-archive root, or exec'd by
# kaggle_environments with no ``__file__`` (fall back to the working dir).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import math

import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache, min_distance_to_targets
from orbit_lite.planner_core import attack_target_mask, friendly_flip_targets
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

TOTAL_STEPS = 500

@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs.  """

    
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5              # fire if score > this
    min_ships_to_launch: float = 4.0
    # --- ETA-aware reinforcement risk ----------------------------------------
    reinforce_size_beta: float = 2.2    # 0 = disabled
    reinforce_eta_free: float = 3.0     # turns before ramp starts
    reinforce_eta_scale: float = 12.0   # turns over which ramp reaches 1
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    ffa_leader_attack_bonus: float = 0.0
    ffa_target_prod_bonus: float = 0.0


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def _prod_weighted_shortlist(obs, obs_tensors, garrison_status, cache, *,
                             config, K_eta, H, prod, source_mask):
    """Target shortlist ranked by prod/(proximity+1) instead of pure proximity.

    Surfaces high-production planets that pure-proximity ranking would skip.
    Defensive flip-targets are unchanged.
    """
    P = obs.P
    device = obs.device
    dtype = torch.float32
    n_attack = max(1, min(int(config.max_offensive_targets), P))
    R = max(0, min(int(config.max_defensive_targets), P))

    attack_mask = attack_target_mask(obs, obs_tensors)
    proximity = min_distance_to_targets(cache, source_mask, attack_mask, max_k=K_eta)
    prod_f = prod.to(dtype=dtype)
    score = prod_f / (proximity.to(dtype=dtype) + 1.0)
    attack_pref = torch.where(attack_mask, score, torch.full_like(score, float("-inf")))
    atk_idx, atk_exists = _candidate_indices(attack_pref, attack_mask, n_attack)

    if R > 0:
        flip_mask, urgency = friendly_flip_targets(obs, garrison_status, H=H, prod=prod)
        def_idx, def_exists = _candidate_indices(urgency, flip_mask, R)
        target_idx = torch.cat([atk_idx, def_idx], dim=0)
        target_exists = torch.cat([atk_exists, def_exists], dim=0)
    else:
        target_idx, target_exists = atk_idx, atk_exists
    return target_idx, target_exists


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.

    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.

    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye              # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                       # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)                                            # [P] summed over sources


def _owner_strength(obs, prod: Tensor, player_count: int) -> Tensor:
    """Production + 2.5% ships as per-owner strength proxy. [player_count]"""
    dtype = prod.dtype
    device = prod.device
    strength = torch.zeros(int(player_count), dtype=dtype, device=device)
    owner = obs.owner_abs.to(device=device)
    ships = obs.ships.to(dtype=dtype, device=device)
    prod_v = prod.to(dtype=dtype, device=device)
    for oid in range(int(player_count)):
        mask = obs.alive & (owner == oid)
        if bool(mask.any()):
            strength[oid] = prod_v[mask].sum() + 0.025 * ships[mask].sum()
    return strength


def _adjust_config(
    config: ProducerLiteConfig,
    *,
    obs,
    prod: Tensor,
    step: int,
    player_count: int,
) -> ProducerLiteConfig:
    """Continuously lower ROI and add waves when losing; restore when winning."""
    pid = int(obs.player_id)
    strength = _owner_strength(obs, prod, int(player_count))
    if pid < 0 or pid >= int(player_count):
        return config

    my = float(strength[pid].item())
    leader = float(strength.max().item())
    ratio = my / max(leader, 1e-6)

    new_roi = float(config.roi_threshold)
    if ratio < 1.0:
        deficit = 1.0 - ratio
        new_roi = max(1.10, new_roi - 0.25 * deficit * deficit)  # quadratic drop
        remaining = TOTAL_STEPS - step
        if remaining < 150 and ratio < 0.90:
            time_urgency = (150 - remaining) / 150.0
            new_roi = max(1.10, new_roi - 0.10 * time_urgency * deficit)

    base_waves = int(config.max_waves_per_turn)
    if ratio < 0.70:
        base_waves = min(7, base_waves + 1)
    if (TOTAL_STEPS - step) < 100 and ratio < 0.95:
        base_waves = min(7, base_waves + 1)

    return dataclasses.replace(config, roi_threshold=new_roi, max_waves_per_turn=base_waves)


def _suppress_late_candidates(
    *,
    score: Tensor,
    obs,
    target_idx: Tensor,
    cand_tgt_short: Tensor,
    cand_is_def: Tensor,
    cand_eta: Tensor,
    step: int,
    player_id: int,
) -> Tensor:
    """Filter attacks that arrive too late to matter; devalue late neutral captures."""
    remaining = TOTAL_STEPS - step
    if remaining > 120:
        return score
    P = int(obs.P)
    if P <= 0 or score.numel() == 0:
        return score
    device = score.device
    dtype = score.dtype
    pid = int(player_id)
    tgt_owner = obs.owner_abs.to(device=device)[target_idx[cand_tgt_short].clamp(0, P - 1)].long()
    eta = cand_eta.reshape(score.shape).to(dtype=dtype)

    is_neutral = tgt_owner < 0
    is_enemy = (tgt_owner >= 0) & (tgt_owner != pid) & (~cand_is_def)

    too_late = (
        (is_neutral & (eta > max(1.0, float(remaining) - 8.0)))
        | (is_enemy  & (eta > max(1.0, float(remaining) - 4.0)))
    )
    neutral_factor = ((float(remaining) - eta) / max(1.0, 80.0)).clamp(min=0.20, max=1.0)
    score = torch.where(is_neutral, score * neutral_factor, score)
    return torch.where(too_late, torch.full_like(score, float("-inf")), score)


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    step = int(obs_tensors["step"].reshape(-1)[0].item())

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    # Enemy pressure — computed once and reused for both the reinforcement floor
    # and the regroup gradient further down.
    beta = float(config.reinforce_size_beta)
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]

    # ETA-aware reinforcement risk: inflate capture floor by β·ρ(k)·enemy_mass(target).
    # ρ(k) ramps from 0 at k=eta_free to 1 at k=eta_free+eta_scale, so short flights
    # (enemy has no time to react) get no extra floor while long flights are penalised.
    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]                     # [T]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange,
            eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )                                                                        # [K_eta]
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)       # [T, K_eta]

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=reinforcement,
    )                                                                            # [T, K]
    K = int(floor.shape[-1])

    # Multi-size candidates: try SIZE_FRACS fractions of safe_drain so a source
    # can split ships across targets (e.g. 50% to T1, 50% to T2 in two waves).
    _SIZE_FRACS = (0.5, 1.0)
    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    min_send_f = float(config.min_ships_to_launch)

    _parts: list[dict] = []
    for _frac in _SIZE_FRACS:
        sizes_f = (drain.view(S, 1) * _frac).floor().expand(S, T)               # [S, T]
        active_f = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_f.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)                                                            # [S, T]
        aim_f = intercept_angle(
            movement,
            source_idx.unsqueeze(1),                                             # [S, 1]
            target_idx.unsqueeze(0),                                             # [1, T]
            sizes_f,                                                              # [S, T]
            active=active_f,
        )
        angle_f = aim_f["angle"]
        eta_f = aim_f["eta"]
        viable_f = aim_f["viable"] & (eta_f <= eta_cap.view(1, T))
        if K > 0:
            k_arr_f = (eta_f.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_f = floor.unsqueeze(0).expand(S, T, K).gather(
                -1, k_arr_f.unsqueeze(-1)
            ).squeeze(-1)
        else:
            floor_at_arr_f = torch.ones(S, T, dtype=dtype, device=device)
        valid_f = (
            viable_f & (sizes_f >= floor_at_arr_f) & (sizes_f >= min_send_f) & src_neq_tgt
            & source_exists.view(S, 1) & target_exists.view(1, T)
        )
        C_f = S * T
        tgt_short_f = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C_f)
        _parts.append(dict(
            src=source_idx.view(S, 1).expand(S, T).reshape(C_f, 1),
            tgt_slot=target_idx.view(1, T).expand(S, T).reshape(C_f),
            tgt_short=tgt_short_f,
            send=torch.where(valid_f, sizes_f, torch.zeros_like(sizes_f)).reshape(C_f, 1),
            angle=angle_f.reshape(C_f, 1),
            eta=torch.where(valid_f, eta_f, torch.ones_like(eta_f)).reshape(C_f, 1),
            active=valid_f.reshape(C_f, 1),
            valid=valid_f.reshape(C_f),
            is_def=target_is_mine[tgt_short_f],
        ))

    L = 1
    cand_src       = torch.cat([p["src"]       for p in _parts], dim=0)
    cand_tgt_slot  = torch.cat([p["tgt_slot"]  for p in _parts], dim=0)
    cand_tgt_short = torch.cat([p["tgt_short"] for p in _parts], dim=0)
    cand_send      = torch.cat([p["send"]      for p in _parts], dim=0)
    cand_angle     = torch.cat([p["angle"]     for p in _parts], dim=0)
    cand_eta       = torch.cat([p["eta"]       for p in _parts], dim=0)
    cand_active    = torch.cat([p["active"]    for p in _parts], dim=0)
    cand_valid     = torch.cat([p["valid"]     for p in _parts], dim=0)
    cand_is_def    = torch.cat([p["is_def"]    for p in _parts], dim=0)
    C = int(cand_src.shape[0])

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )                                                                            # [C]
    if int(player_count) >= 4 and (
        float(config.ffa_leader_attack_bonus) > 0.0
        or float(config.ffa_target_prod_bonus) > 0.0
    ):
        owner = obs.owner_abs.to(torch.long)
        owner_valid = (owner >= 0) & (owner < int(player_count)) & obs.alive
        owner_idx = owner.clamp(min=0, max=max(int(player_count) - 1, 0))
        prod_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        ships_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        prod_by_owner.scatter_add_(0, owner_idx, torch.where(owner_valid, prod.to(dtype), torch.zeros_like(prod.to(dtype))))
        ships_by_owner.scatter_add_(0, owner_idx, torch.where(owner_valid, obs.ships.to(dtype), torch.zeros_like(obs.ships.to(dtype))))
        strength = prod_by_owner + 0.025 * ships_by_owner
        my_strength = strength[pid].detach()

        target_owner = owner[target_idx.clamp(0, P - 1)].clamp(min=0, max=max(int(player_count) - 1, 0))
        target_owned_enemy = (
            target_exists
            & obs.is_enemy[target_idx.clamp(0, P - 1)]
            & (obs.owner_abs[target_idx.clamp(0, P - 1)] >= 0)
        )
        owner_strength = strength[target_owner]
        leader_delta = (owner_strength - my_strength).clamp(min=0.0)
        target_bonus_short = torch.where(
            target_owned_enemy,
            float(config.ffa_leader_attack_bonus) * leader_delta
            + float(config.ffa_target_prod_bonus) * prod[target_idx.clamp(0, P - 1)].to(dtype),
            torch.zeros_like(owner_strength),
        )
        score = score + target_bonus_short[cand_tgt_short]
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_offensive_targets=7,
    max_defensive_targets=2,
    roi_threshold=1.55,
    min_ships_to_launch=5.0,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    ffa_leader_attack_bonus=0.035,
    ffa_target_prod_bonus=0.08,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Comet targeting
# ---------------------------------------------------------------------------

def _fleet_speed(n_ships: float) -> float:
    if n_ships <= 1.0:
        return 1.0
    return 1.0 + 5.0 * (math.log(max(n_ships, 1.0)) / math.log(1000.0)) ** 1.5


def _comet_moves(obs, existing_moves: list) -> list:
    """Append moves that capture neutral comets when a fleet can intercept them."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    comets_data = obs.get("comets", []) if isinstance(obs, dict) else getattr(obs, "comets", [])
    comet_ids_raw = obs.get("comet_planet_ids", []) if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", [])
    comet_planet_ids = {int(x) for x in comet_ids_raw if int(x) >= 0}

    if not comet_planet_ids or not comets_data:
        return []

    # Ships already committed by existing moves
    committed: dict[int, int] = {}
    for move in existing_moves:
        pid = int(move[0])
        committed[pid] = committed.get(pid, 0) + int(move[2])

    # Build comet_id → (path_list, path_index)
    comet_path_info: dict[int, tuple[list, int]] = {}
    for group in comets_data:
        if isinstance(group, dict):
            path_idx = int(group.get("path_index", 0))
            ids = group.get("planet_ids", [])
            paths = group.get("paths", [])
        else:
            path_idx = int(getattr(group, "path_index", 0))
            ids = getattr(group, "planet_ids", [])
            paths = getattr(group, "paths", [])
        for ci, cid in enumerate(ids):
            cid = int(cid)
            if cid >= 0 and cid in comet_planet_ids and ci < len(paths):
                comet_path_info[cid] = (paths[ci], path_idx)

    planets_by_id = {int(p[0]): p for p in raw_planets if int(p[0]) >= 0}
    my_planets = [p for p in raw_planets if int(p[0]) >= 0 and int(p[1]) == player_id]

    targeted: set[int] = set()
    moves: list = []

    for p in my_planets:
        p_id = int(p[0])
        p_x, p_y = float(p[2]), float(p[3])
        available = float(p[5]) - committed.get(p_id, 0)

        best: tuple | None = None  # (comet_id, angle, ships, eta)
        best_eta = float("inf")

        for c_id in comet_planet_ids:
            if c_id in targeted or c_id not in planets_by_id:
                continue
            cp = planets_by_id[c_id]
            if int(cp[1]) == player_id:
                continue  # already ours
            ships_to_send = max(int(float(cp[5])) + 1, 1)
            if available < ships_to_send:
                continue

            speed = _fleet_speed(ships_to_send)

            if c_id not in comet_path_info:
                continue
            path, path_idx = comet_path_info[c_id]

            for delta_t in range(1, 25):
                idx = path_idx + delta_t
                if idx >= len(path):
                    break
                cx, cy = float(path[idx][0]), float(path[idx][1])
                if cx < 0 or cx > 100 or cy < 0 or cy > 100:
                    break  # comet will leave board
                dist = math.sqrt((cx - p_x) ** 2 + (cy - p_y) ** 2)
                if dist <= speed * delta_t:
                    if delta_t < best_eta:
                        angle = math.atan2(cy - p_y, cx - p_x)
                        best = (c_id, angle, ships_to_send, delta_t)
                        best_eta = delta_t
                    break

        if best is not None:
            c_id, angle, ships, _ = best
            targeted.add(c_id)
            committed[p_id] = committed.get(p_id, 0) + ships
            moves.append([p_id, angle, ships])

    return moves


# ---------------------------------------------------------------------------
# Agent factory (useful for experiments with custom configs)
# ---------------------------------------------------------------------------

def make_agent(
    config_2p: ProducerLiteConfig | None = None,
    config_4p: ProducerLiteConfig | None = None,
    *,
    comet_targeting: bool = True,
):
    """Return a fresh agent closure using the given configs."""
    _cfg_2p = config_2p if config_2p is not None else ProducerLiteConfig()
    _cfg_4p = config_4p if config_4p is not None else CONFIG_4P
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
        cfg = _cfg_4p if int(mem.cached_player_count) >= 4 else _cfg_2p
        with torch.no_grad():
            row = run_turn(obs_tensors, config=cfg, player_count=int(mem.cached_player_count), memory=mem)
        mem.last_sparse_action_row = row
        moves = sparse_action_row_to_moves(row, obs, player_id=player_id)
        if comet_targeting:
            moves += _comet_moves(obs, moves)
        return moves

    return _agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
