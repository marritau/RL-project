"""Environment contract and reproducibility tests."""

from __future__ import annotations

from pacman_rldp.env import PacmanEnv, PacmanEnvConfig


def pick_first_legal_action(info: dict[str, object]) -> int:
    """Return deterministic first legal action id from info payload."""
    legal_action_ids = info.get("legal_action_ids", [])
    if not isinstance(legal_action_ids, list) or not legal_action_ids:
        return 4
    return int(sorted(int(action_id) for action_id in legal_action_ids)[0])


def test_reset_and_step_contract() -> None:
    """Verify Gym-style reset/step signatures and observation compatibility."""
    env = PacmanEnv(PacmanEnvConfig())
    observation, info = env.reset(seed=7)

    assert isinstance(observation, dict)
    assert isinstance(info, dict)
    assert env.observation_space.contains(observation)

    action = pick_first_legal_action(info)
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(next_info, dict)
    env.close()


def test_termination_and_truncation_flags() -> None:
    """Verify episode truncation by step limit is reported correctly."""
    config = PacmanEnvConfig(max_steps=1)
    env = PacmanEnv(config)
    observation, info = env.reset(seed=11)
    action = pick_first_legal_action(info)

    del observation
    _, _, terminated, truncated, _ = env.step(action)

    if not terminated:
        assert truncated is True
    env.close()


def test_invalid_action_raises_value_error() -> None:
    """Verify invalid Pacman action handling in raise mode."""
    env = PacmanEnv(PacmanEnvConfig(invalid_action_mode="raise"))
    _, info = env.reset(seed=3)
    legal = set(int(action_id) for action_id in info["legal_action_ids"])
    illegal_candidates = [action_id for action_id in range(5) if action_id not in legal]

    if illegal_candidates:
        illegal_action = illegal_candidates[0]
        try:
            env.step(illegal_action)
            raised = False
        except ValueError:
            raised = True
        assert raised is True
    env.close()


def test_seed_reproducibility_for_state_trace() -> None:
    """Verify identical seeds produce identical transition traces."""
    cfg = PacmanEnvConfig(seed=99)
    env_a = PacmanEnv(cfg)
    env_b = PacmanEnv(cfg)

    obs_a, info_a = env_a.reset(seed=99)
    obs_b, info_b = env_b.reset(seed=99)

    del obs_a, obs_b
    trace_a: list[tuple[tuple[float, float], float]] = []
    trace_b: list[tuple[tuple[float, float], float]] = []

    for _ in range(5):
        action_a = pick_first_legal_action(info_a)
        action_b = pick_first_legal_action(info_b)

        _, _, term_a, trunc_a, info_a = env_a.step(action_a)
        _, _, term_b, trunc_b, info_b = env_b.step(action_b)

        pos_a = env_a.runtime_state.getPacmanPosition()
        pos_b = env_b.runtime_state.getPacmanPosition()
        trace_a.append(((float(pos_a[0]), float(pos_a[1])), float(info_a["score"])))
        trace_b.append(((float(pos_b[0]), float(pos_b[1])), float(info_b["score"])))

        if term_a or trunc_a or term_b or trunc_b:
            break

    assert trace_a == trace_b
    env_a.close()
    env_b.close()
