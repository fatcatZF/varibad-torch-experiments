"""Meta-test evaluation: online return within the first episode(s) of a new task."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from utils.rollout import initial_belief, reset_done_envs, sample_action, update_belief


@torch.no_grad()
def run_meta_eval(
    model,
    envs: Any,
    device: torch.device,
    is_discrete: bool,
    max_steps: int,
    action_dim: int,
    action_low=None,
    action_high=None,
    num_episodes: int = 2,
) -> Dict[str, np.ndarray]:
    """Run the policy with deterministic actions on a batch of new tasks.

    The belief (GRU state) starts at the prior and is **never reset** between episodes of the same
    task (the encoder's ``done`` input tells it when an episode ended).  The paper's metric is the
    return of the *first* episode, i.e. performance while still learning the task.

    Parameters
    ----------
    envs : vector env with ``N`` tasks (``make_envs_eval(tasks)``)
    max_steps : int
        Safety cap on the number of steps.
    num_episodes : int
        Episodes per task to record.

    Returns
    -------
    dict with ``returns`` (N, num_episodes) undiscounted episode returns (NaN if not finished within
    ``max_steps``) and ``mean`` (num_episodes,) = ``nanmean`` over tasks.
    """
    model.eval()
    N = envs.num_envs
    obs, _ = envs.reset()
    mu, logvar, h = initial_belief(model, N, device)
    ep_idx = np.zeros(N, dtype=int)
    ep_ret = np.zeros(N)
    returns = np.full((N, num_episodes), np.nan)

    for _ in range(max_steps):
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        act, _, _, _ = sample_action(model, obs_t, mu, logvar, is_discrete, action_low, action_high, deterministic=True)
        next_obs, rew, term, trunc, _ = envs.step(act)
        done = np.logical_or(term, trunc)
        ep_ret += rew
        mu, logvar, h = update_belief(model, h, obs, act, rew, next_obs, done.astype(np.float32), device)
        for i in np.nonzero(done)[0]:
            if ep_idx[i] < num_episodes:
                returns[i, ep_idx[i]] = ep_ret[i]
            ep_idx[i] += 1
            ep_ret[i] = 0.0
        obs = reset_done_envs(envs, obs, next_obs, done)
        if (ep_idx >= num_episodes).all():
            break

    model.train()
    with np.errstate(all="ignore"):
        mean = np.nanmean(returns, axis=0)
    return {"returns": returns, "mean": mean}
