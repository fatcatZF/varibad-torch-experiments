"""Buffers for VariBAD.

* :class:`RolloutBuffer` — on-policy data for PPO (only the most recent rollout), including the
  belief ``(mu, logvar)`` that was fed to the policy at each step.
* :class:`VAEBuffer` — a separate, larger buffer of complete BAMDP trajectories used to train the
  encoder / decoders (paper, Sec. 3.2: "for the VAE we maintain a separate, larger buffer").

All buffers are time-major: arrays have shape ``(T, N, ...)`` with ``T`` steps and ``N`` envs.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


class RolloutBuffer:
    """PPO rollout storage for ``N`` parallel envs (each running one BAMDP trajectory).

    Parameters
    ----------
    device : torch.device
    is_discrete : bool
    """

    KEYS = ("obs", "act", "raw_act", "logp", "val", "rew", "term", "done", "next_obs", "mu", "logvar")

    def __init__(self, device: torch.device, is_discrete: bool):
        self.device, self.is_discrete = device, is_discrete
        self.reset()

    def reset(self) -> None:
        self.data: Dict[str, List[np.ndarray]] = {k: [] for k in self.KEYS}
        self.adv = None
        self.ret = None

    def add(self, **kw) -> None:
        """Append one time step.  Every value has a leading dimension ``N`` (number of envs).

        ``obs`` / ``next_obs`` (N, obs_dim) — ``next_obs`` is the *real* successor (before any reset);
        ``act`` (N,) or (N, A) in env space; ``raw_act`` same shape (pre-tanh sample for bounded
        continuous actions); ``logp``, ``val`` (N,); ``rew`` (N,); ``term`` / ``done`` (N,) float
        (``done = terminated or truncated``); ``mu`` / ``logvar`` (N, Z).
        """
        for k in self.KEYS:
            self.data[k].append(np.asarray(kw[k]))

    def __len__(self) -> int:
        return len(self.data["rew"])

    def stacked(self, key: str) -> np.ndarray:
        """Stack one field to shape (T, N, ...)."""
        return np.stack(self.data[key])

    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float, cut_at_episode_end: bool = True,
                    reward_scale: float = 1.0) -> None:
        """Generalised advantage estimation over the whole BAMDP trajectory.

        Parameters
        ----------
        last_value : Tensor (N,)
            ``V`` of the observation after the final step (bootstrap).
        cut_at_episode_end : bool
            True (default, same as RL2-Torch-Experiments): returns do not cross episode boundaries,
            i.e. the mask is ``1 - done``.  False: the return runs across episode resets for the full
            horizon, which is the BAMDP objective of Eq. (3) in the paper.
        """
        T, N = len(self), self.data["rew"][0].shape[0]
        f = lambda k: torch.as_tensor(self.stacked(k), device=self.device, dtype=torch.float32).view(T, N)
        vals, rews, dones = f("val"), f("rew") * reward_scale, f("done")
        full = torch.cat([vals, last_value.view(1, N).to(self.device)], dim=0)
        adv = torch.zeros(T, N, device=self.device)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            mask = (1.0 - dones[t]) if cut_at_episode_end else torch.ones(N, device=self.device)
            delta = rews[t] + gamma * mask * full[t + 1] - full[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
        self.adv, self.ret = adv, adv + vals

    def flat(self) -> Dict[str, torch.Tensor]:
        """Flatten ``(T, N, ...)`` to ``(T*N, ...)`` tensors for minibatch PPO."""
        assert self.adv is not None, "call compute_gae first"
        T, N = self.adv.shape
        out = {}
        for k in ("obs", "mu", "logvar"):
            out[k] = torch.as_tensor(self.stacked(k), device=self.device, dtype=torch.float32).reshape(T * N, -1)
        for k in ("act", "raw_act"):
            a = torch.as_tensor(self.stacked(k), device=self.device)
            out[k] = a.reshape(T * N).long() if self.is_discrete else a.float().reshape(T * N, -1)
        out["logp"] = torch.as_tensor(self.stacked("logp"), device=self.device, dtype=torch.float32).reshape(-1)
        out["adv"], out["ret"] = self.adv.reshape(-1), self.ret.reshape(-1)
        return out

    def trajectories(self) -> Dict[str, np.ndarray]:
        """Per-env trajectories ``(T, N, ...)`` for the VAE buffer."""
        d = {k: self.stacked(k) for k in ("obs", "act", "rew", "next_obs", "done")}
        d["rew"], d["done"] = d["rew"][..., None], d["done"][..., None]
        return d


class VAEBuffer:
    """Ring buffer of complete BAMDP trajectories, one task per trajectory.

    Parameters
    ----------
    capacity : int
        Maximum number of trajectories.
    is_discrete : bool
    """

    def __init__(self, capacity: int, is_discrete: bool):
        self.capacity, self.is_discrete = capacity, is_discrete
        self.store: Dict[str, np.ndarray] = {}
        self.ptr = 0
        self.size = 0

    def add(self, traj: Dict[str, np.ndarray]) -> None:
        """Add ``N`` trajectories given time-major ``(T, N, ...)`` arrays (as from ``RolloutBuffer.trajectories``)."""
        N = traj["obs"].shape[1]
        if not self.store:
            for k, v in traj.items():
                dtype = np.int64 if (k == "act" and self.is_discrete) else np.float32
                self.store[k] = np.zeros((self.capacity, v.shape[0], *v.shape[2:]), dtype)
        for n in range(N):
            for k, v in traj.items():
                self.store[k][self.ptr] = v[:, n]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device) -> Dict[str, torch.Tensor]:
        """Sample ``batch`` trajectories uniformly; returns time-major tensors ``(T, B, ...)``."""
        idx = np.random.randint(0, self.size, size=batch)
        return {k: torch.as_tensor(v[idx], device=device).transpose(0, 1).contiguous() for k, v in self.store.items()}
