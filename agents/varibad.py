"""VariBAD model container: encoder + decoders + belief-conditioned actor-critic.

Implements the training objective of Eq. (10) in the paper::

    L = E[ J(policy)  +  lambda * sum_t ELBO_t ]
    ELBO_t = E_{q(m | tau_{:t})} [ log p(tau_{:H+} | m) ]  -  KL( q(m | tau_{:t}) || q(m | tau_{:t-1}) )

* the encoder sees the past ``tau_{:t}``; the decoders reconstruct **all** transitions of the
  BAMDP trajectory (past and future) from a sample ``m ~ q(m | tau_{:t})``;
* the prior at ``t = 0`` is ``N(0, I)``; afterwards the prior is the previous posterior;
* as recommended in the paper, the RL loss is **not** back-propagated through the encoder
  (separate optimisers; the policy receives the detached belief ``(mu, logvar)``).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.encoder import RNNEncoder, kl_diag_gaussians
from agents.nn_tools import mlp


class RewardDecoder(nn.Module):
    """``p(r_{i+1} | s_i, a_i, s_{i+1}, m)``, Gaussian with fixed variance (MSE loss).

    Input ``[m, s_i, a_i, s_{i+1}]`` -> predicted (normalised) reward ``(..., 1)``.
    """

    def __init__(self, latent_dim, obs_dim, action_feat_dim, hidden=(64, 32)):
        super().__init__()
        self.net = mlp(latent_dim + 2 * obs_dim + action_feat_dim, 1, hidden, nn.ReLU, last_std=1.0)

    def forward(self, m, obs, act_feat, next_obs):
        return self.net(torch.cat([m, obs, act_feat, next_obs], dim=-1))


class StateDecoder(nn.Module):
    """``p(s_{i+1} | s_i, a_i, m)``: predicts the normalised state change ``s_{i+1} - s_i``.

    Input ``[m, s_i, a_i]`` -> ``(..., obs_dim)``.
    """

    def __init__(self, latent_dim, obs_dim, action_feat_dim, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(latent_dim + obs_dim + action_feat_dim, obs_dim, hidden, nn.ReLU, last_std=1.0)

    def forward(self, m, obs, act_feat):
        return self.net(torch.cat([m, obs, act_feat], dim=-1))


class BeliefActorCritic(nn.Module):
    """Feed-forward policy and value function on ``[s, mu, logvar]`` (the belief state).

    Because the belief is computed once during the rollout and stored, PPO minibatches need no
    recurrent re-computation (this is why VariBAD's PPO is cheaper than RL²'s, see paper C.4).

    Parameters
    ----------
    obs_dim, latent_dim, action_dim : int
    is_discrete : bool
    hidden : Sequence[int]
        Hidden sizes of actor and critic MLPs (paper, MuJoCo: 2 x 128, tanh).
    """

    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int, is_discrete: bool,
                 hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.is_discrete = is_discrete
        in_dim = obs_dim + 2 * latent_dim
        self.actor = mlp(in_dim, action_dim if is_discrete else 2 * action_dim, hidden)
        self.critic = mlp(in_dim, 1, hidden, last_std=1.0)

    def forward(self, obs, mu, logvar):
        """Returns ``(policy_out, value)``.

        ``policy_out`` is logits (discrete) or ``(mu, log_std)`` (continuous); value is ``(..., 1)``.
        """
        x = torch.cat([obs, mu, logvar], dim=-1)
        out = self.actor(x)
        value = self.critic(x)
        if self.is_discrete:
            return out, value
        a_mu, log_std = torch.chunk(out, 2, dim=-1)
        return (a_mu, log_std.clamp(-20, 2)), value


def _gather_time(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``x`` (T, B, ...) and ``idx`` (K, B) int64 -> (K, B, ...) with ``out[k, b] = x[idx[k, b], b]``."""
    idx_e = idx.view(*idx.shape, *([1] * (x.dim() - 2))).expand(*idx.shape, *x.shape[2:])
    return torch.gather(x, 0, idx_e)


class VariBADModel(nn.Module):
    """Encoder, decoders, policy and input normalisers.

    Parameters
    ----------
    obs_dim, action_dim : int
    is_discrete : bool
    latent_dim : int
        Task-embedding size (paper: 5).
    obs_norm, rew_norm : nn.Module | None
        ``RunningMeanStd1DNormalizer`` for observations / rewards.  All encoder / decoder inputs and
        decoder targets live in the normalised space.
    decode_reward, decode_state : bool
        Which parts of ``log p(tau | m)`` to model.  The paper uses the reward decoder only on its
        MuJoCo tasks, but if the task changes the **dynamics but not the reward function**
        (the wind / action-offset environments here: reward is a function of the observation only)
        the reward decoder carries no task information, so ``decode_state`` must be True.
    use_prev_state : bool
        See :class:`RNNEncoder`.
    encoder_kwargs : dict
        Extra arguments for :class:`RNNEncoder` (embedding sizes, hidden size).
    policy_hidden : Sequence[int]
    """

    def __init__(self, obs_dim: int, action_dim: int, is_discrete: bool, latent_dim: int = 5,
                 obs_norm: Optional[nn.Module] = None, rew_norm: Optional[nn.Module] = None,
                 decode_reward: bool = True, decode_state: bool = True, use_prev_state: bool = True,
                 policy_hidden: Sequence[int] = (128, 128), **encoder_kwargs):
        super().__init__()
        assert decode_reward or decode_state, "at least one decoder is needed"
        self.obs_dim, self.action_dim, self.is_discrete = obs_dim, action_dim, is_discrete
        self.latent_dim = latent_dim
        self.obs_norm, self.rew_norm = obs_norm, rew_norm
        self.encoder = RNNEncoder(obs_dim, action_dim, is_discrete, latent_dim,
                                  use_prev_state=use_prev_state, **encoder_kwargs)
        self.reward_decoder = RewardDecoder(latent_dim, obs_dim, action_dim) if decode_reward else None
        self.state_decoder = StateDecoder(latent_dim, obs_dim, action_dim) if decode_state else None
        self.policy = BeliefActorCritic(obs_dim, latent_dim, action_dim, is_discrete, policy_hidden)

    # ---------------------------------------------------------------- parameters
    def vae_parameters(self):
        """Encoder + decoder parameters (optimised with the VAE learning rate)."""
        mods = [self.encoder, self.reward_decoder, self.state_decoder]
        return [p for m in mods if m is not None for p in m.parameters()]

    def policy_parameters(self):
        return self.policy.parameters()

    # ------------------------------------------------------------- normalisation
    def norm_obs(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.obs_norm is None else self.obs_norm(x)

    def norm_rew(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.rew_norm is None else self.rew_norm(x)

    @torch.no_grad()
    def update_normalizers(self, obs: torch.Tensor, rew: torch.Tensor) -> None:
        """Update running statistics (obs ``(..., obs_dim)``, rew ``(..., 1)``)."""
        if self.obs_norm is not None:
            self.obs_norm.update(obs)
        if self.rew_norm is not None:
            self.rew_norm.update(rew)

    def act_feat(self, act: torch.Tensor) -> torch.Tensor:
        """Action features for the decoders: one-hot (discrete) or the action itself."""
        return F.one_hot(act.long(), self.action_dim).float() if self.is_discrete else act

    # ------------------------------------------------------------------ belief
    def prior(self, batch: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Belief before any data: ``mu = 0, logvar = 0`` (each (batch, Z)) and zero GRU state."""
        z = torch.zeros(batch, self.latent_dim, device=device)
        return z, z.clone(), self.encoder.initial_state(batch, device)

    def encode(self, obs, act, rew, next_obs, done, h0=None):
        """Normalise raw transitions and run the encoder.  Shapes as in :mod:`agents.encoder`."""
        return self.encoder(self.norm_obs(obs), act, self.norm_rew(rew), self.norm_obs(next_obs), done, h0)

    def policy_forward(self, obs, mu, logvar):
        """Policy / value on raw observations and a (detached) belief."""
        return self.policy(self.norm_obs(obs), mu, logvar)

    # --------------------------------------------------------------------- VAE
    def vae_loss(self, batch: Dict[str, torch.Tensor], num_elbo: int = 50, num_decode: int = 50,
                 kl_weight: float = 0.1, reward_coef: float = 1.0, state_coef: float = 1.0
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Negative ELBO on a batch of full BAMDP trajectories (Eq. 8-10).

        For each trajectory the encoder is unrolled over the whole horizon.  ``num_elbo`` belief
        times ``t`` are sampled per trajectory (the paper allows sub-sampling the sum over ``t``);
        for each of them ``num_decode`` transitions anywhere in the trajectory (past *and* future)
        are reconstructed from ``m ~ q(m | tau_{:t})``.

        Parameters
        ----------
        batch : dict of time-major tensors
            ``obs`` / ``next_obs`` (T, B, obs_dim), ``act`` (T, B) or (T, B, A), ``rew`` / ``done`` (T, B, 1).

        Returns
        -------
        loss : scalar tensor, stats : dict of floats
        """
        obs, act, rew, nxt, done = (batch[k] for k in ("obs", "act", "rew", "next_obs", "done"))
        T, B = obs.shape[:2]
        dev = obs.device
        n_obs, n_nxt, n_rew = self.norm_obs(obs), self.norm_obs(nxt), self.norm_rew(rew)

        mu_seq, lv_seq, _ = self.encoder(n_obs, act, n_rew, n_nxt, done)          # beliefs after transitions 0..T-1
        zeros = torch.zeros(1, B, self.latent_dim, device=dev)
        mu_all, lv_all = torch.cat([zeros, mu_seq]), torch.cat([zeros, lv_seq])    # (T+1, B, Z); index t = after t transitions

        # KL( q_t || q_{t-1} ) for t = 1..T, prior of q_0 is N(0, I) (KL = 0).
        kl = kl_diag_gaussians(mu_all[1:], lv_all[1:], mu_all[:-1], lv_all[:-1])    # (T, B)

        t_idx = torch.randint(1, T + 1, (num_elbo, B), device=dev)                  # sampled belief times
        mu_t, lv_t = _gather_time(mu_all, t_idx), _gather_time(lv_all, t_idx)       # (S, B, Z)
        m = mu_t + torch.randn_like(mu_t) * (0.5 * lv_t).exp()                      # reparameterised samples
        kl_t = _gather_time(kl, t_idx - 1)                                          # (S, B)

        d_idx = torch.randint(0, T, (num_decode, B), device=dev)                    # transitions to reconstruct
        g = lambda x: _gather_time(x, d_idx).unsqueeze(0).expand(num_elbo, *([-1] * x.dim()))  # (S, D, B, ...)
        m_e = m.unsqueeze(1).expand(num_elbo, num_decode, B, self.latent_dim)
        s_i, s_n, a_f, r_i = g(n_obs), g(n_nxt), g(self.act_feat(act)), g(n_rew)

        stats: Dict[str, float] = {}
        recon = torch.zeros((), device=dev)
        if self.reward_decoder is not None:
            l_r = F.mse_loss(self.reward_decoder(m_e, s_i, a_f, s_n), r_i)
            recon = recon + reward_coef * l_r
            stats["reward_loss"] = float(l_r.detach())
        if self.state_decoder is not None:
            l_s = F.mse_loss(self.state_decoder(m_e, s_i, a_f), s_n - s_i)
            recon = recon + state_coef * l_s
            stats["state_loss"] = float(l_s.detach())
        kl_mean = kl_t.mean()
        stats["kl"] = float(kl_mean.detach())
        stats["post_std"] = float((0.5 * lv_t).exp().mean().detach())
        return recon + kl_weight * kl_mean, stats
