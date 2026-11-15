"""Belief-tracking environment stepping shared by training and evaluation."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from utils.distributions import make_action_dist


def reset_done_envs(envs, obs: np.ndarray, next_obs: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Start a new episode in the envs that just finished, without any dummy step.

    gymnasium's default NEXT_STEP autoreset would waste the following ``step`` call on a reset
    (the action is ignored and reward is 0), which would pollute the decoder targets.  Instead the
    finished sub-envs are reset immediately with ``reset_mask`` (gymnasium >= 1.1).

    Parameters
    ----------
    obs, next_obs : (N, obs_dim)
        ``next_obs`` is the real successor returned by ``step`` (the terminal observation for done envs).
    done : (N,) bool

    Returns
    -------
    (N, obs_dim) observations to act on next: reset observation for done envs, ``next_obs`` otherwise.
    """
    if not done.any():
        return next_obs
    reset_obs, _ = envs.reset(options={"reset_mask": done})
    out = np.array(next_obs, copy=True)
    out[done] = reset_obs[done]
    return out


@torch.no_grad()
def sample_action(model, obs_t: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, is_discrete: bool,
                  action_low=None, action_high=None, deterministic: bool = False):
    """Policy step on a batch of ``N`` envs.

    Parameters
    ----------
    obs_t : (N, obs_dim) raw observations; mu, logvar : (N, Z) current belief.

    Returns
    -------
    act_env (N,) or (N, A) numpy in env space, raw_act numpy (pre-tanh for bounded continuous),
    logp (N,) numpy, value (N,) numpy.
    """
    policy_out, value = model.policy_forward(obs_t, mu, logvar)
    dist = make_action_dist(policy_out, is_discrete, action_low, action_high)
    value = value.reshape(-1).cpu().numpy()
    if is_discrete:
        a = dist.probs.argmax(-1) if deterministic else dist.sample()
        return a.cpu().numpy(), a.cpu().numpy(), dist.log_prob(a).cpu().numpy(), value
    if hasattr(dist, "log_prob_from_u"):                       # bounded: TanhNormal
        if deterministic:
            u = dist.base.mean
            a = dist._squash(u)
        else:
            a, u = dist.sample()
        logp = dist.log_prob_from_u(u).sum(-1)
        return a.cpu().numpy(), u.cpu().numpy(), logp.cpu().numpy(), value
    a = dist.mean if deterministic else dist.sample()
    return a.cpu().numpy(), a.cpu().numpy(), dist.log_prob(a).sum(-1).cpu().numpy(), value


def initial_belief(model, num_envs: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prior belief ``N(0, I)`` and zero GRU state for ``num_envs`` fresh BAMDP trajectories."""
    return model.prior(num_envs, device)


@torch.no_grad()
def update_belief(model, h, obs, act_env, rew, next_obs, done, device):
    """Feed one transition per env to the encoder and return ``(mu, logvar, h)``.

    Shapes: obs / next_obs (N, obs_dim) numpy, act_env (N,) or (N, A), rew (N,), done (N,) bool.
    """
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), device=device, dtype=dt).unsqueeze(0)
    a = t(act_env, torch.int64) if model.is_discrete else t(act_env)
    mu, logvar, h = model.encode(t(obs), a, t(rew).unsqueeze(-1), t(next_obs), t(done).unsqueeze(-1), h)
    return mu[0], logvar[0], h
