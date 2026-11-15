"""VariBAD meta-training (Zintgraf et al., ICLR 2020) with the same factory-style interface as
``train_rl2_ppo`` in RL2-Torch-Experiments.

Per iteration:

1. sample ``num_envs`` new tasks; roll out the policy for ``horizon`` steps per env.  Each env is one
   BAMDP trajectory: the belief ``q(m | tau_{:t})`` is carried over episode resets (the encoder gets a
   ``done`` flag) and is fed, detached, to the policy;
2. PPO update of the policy / value function on the stored ``[s, mu, logvar]`` inputs (no recurrence);
3. a few ELBO updates of encoder + decoders on trajectories drawn from a separate, larger VAE buffer.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.buffer import RolloutBuffer, VAEBuffer
from utils.distributions import make_action_dist
from utils.evaluation import run_meta_eval
from utils.rollout import initial_belief, reset_done_envs, sample_action, update_belief


def collect_rollout(model, envs, buffer: RolloutBuffer, horizon: int, device, is_discrete: bool,
                    action_low=None, action_high=None):
    """Roll out ``horizon`` steps in every env and fill ``buffer``.

    Returns
    -------
    last_value : Tensor (N,)  bootstrap value of the final (obs, belief)
    stats : dict with ``episodes`` and ``mean_episode_return`` of finished episodes.
    """
    model.eval()
    buffer.reset()
    N = envs.num_envs
    obs, _ = envs.reset()
    mu, logvar, h = initial_belief(model, N, device)
    ep_ret, finished = np.zeros(N), []

    for _ in range(horizon):
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        act, raw, logp, val = sample_action(model, obs_t, mu, logvar, is_discrete, action_low, action_high)
        next_obs, rew, term, trunc, _ = envs.step(act)
        done = np.logical_or(term, trunc)
        buffer.add(obs=obs, act=act, raw_act=raw, logp=logp, val=val, rew=rew, term=term.astype(np.float32),
                   done=done.astype(np.float32), next_obs=next_obs, mu=mu.cpu().numpy(), logvar=logvar.cpu().numpy())
        model.update_normalizers(obs_t, torch.as_tensor(rew, device=device, dtype=torch.float32).unsqueeze(-1))
        new_mu, new_logvar, h = update_belief(model, h, obs, act, rew, next_obs, done.astype(np.float32), device)
        ep_ret += rew
        for i in np.nonzero(done)[0]:
            finished.append(ep_ret[i])
            ep_ret[i] = 0.0
        obs = reset_done_envs(envs, obs, next_obs, done)
        mu, logvar = new_mu, new_logvar

    with torch.no_grad():
        _, last_v = model.policy_forward(torch.as_tensor(obs, device=device, dtype=torch.float32), mu, logvar)
    return last_v.reshape(-1), {"episodes": len(finished),
                                "mean_episode_return": float(np.mean(finished)) if finished else float("nan")}


def ppo_update(model, optimizer, data: Dict[str, torch.Tensor], is_discrete: bool, action_low, action_high,
               epochs: int, minibatches: int, clip: float, vf_coef: float, ent_coef: float,
               max_grad_norm: float) -> Dict[str, float]:
    """PPO on flattened ``(T*N, ...)`` data.  The belief is a fixed input, so the policy is feed-forward.

    The value loss is a Huber loss (paper, Appendix C.6).
    """
    model.train()
    n = data["adv"].shape[0]
    stats = {"pi_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0}
    count = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=data["adv"].device)
        for idx in perm.chunk(minibatches):
            out, val = model.policy_forward(data["obs"][idx], data["mu"][idx], data["logvar"][idx])
            dist = make_action_dist(out, is_discrete, action_low, action_high)
            if is_discrete:
                new_lp = dist.log_prob(data["act"][idx])
            elif hasattr(dist, "log_prob_from_u"):
                new_lp = dist.log_prob_from_u(data["raw_act"][idx]).sum(-1)
            else:
                new_lp = dist.log_prob(data["act"][idx]).sum(-1)
            adv = data["adv"][idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(new_lp - data["logp"][idx])
            pi_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            vf_loss = F.smooth_l1_loss(val.squeeze(-1), data["ret"][idx])
            entropy = dist.entropy().mean()
            loss = pi_loss + vf_coef * vf_loss - ent_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.policy_parameters(), max_grad_norm)
            optimizer.step()
            stats["pi_loss"] += float(pi_loss.detach()); stats["vf_loss"] += float(vf_loss.detach())
            stats["entropy"] += float(entropy.detach()); count += 1
    return {k: v / max(count, 1) for k, v in stats.items()}


def train_varibad(
    # Model & Env Factories (same as train_rl2_ppo)
    model_factory: Callable[[], nn.Module],
    sample_tasks_train: Callable[[int], List[Dict[str, Any]]],
    make_envs_train: Callable[[List[Dict[str, Any]]], Any],
    sample_tasks_eval: Callable[[int], List[Dict[str, Any]]],
    make_envs_eval: Callable[[List[Dict[str, Any]]], Any],
    # Dimensions & Config
    action_dim: int,
    is_discrete: bool = True,
    action_low=None, action_high=None,
    # RL (PPO) hyper-parameters; defaults from paper Appendix C.6 where given
    num_envs: int = 16,
    total_updates: int = 1000,
    horizon: int = 1024,
    max_eval_steps: int = 2000,
    ppo_epochs: int = 2,
    ppo_minibatches: int = 4,
    clip: float = 0.1,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr_policy: float = 7e-4,
    cut_returns_at_episode_end: bool = True,
    # VAE hyper-parameters
    lr_vae: float = 1e-3,
    kl_weight: float = 0.1,
    vae_updates_per_iter: int = 3,
    vae_batch: int = 16,
    vae_buffer_size: int = 1000,
    num_elbo: int = 50,
    num_decode: int = 50,
    # Evaluation
    eval_interval: int = 20,
    eval_tasks_count: int = 10,
    eval_episodes: int = 2,
    seed: int = 0,
    device: Optional[torch.device] = None,
    save_path: str = "best_varibad_model.pt",
    log_fn: Callable[[str], None] = print,
):
    """Meta-train VariBAD with PPO.  Returns the best model (by first-episode eval return).

    ``save_path`` always holds the best policy so far (highest mean first-episode return over the
    evaluation tasks); it is overwritten only on improvement.  The first-episode return is the paper's
    metric: performance while the task is still being inferred.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory().to(device)
    opt_pi = torch.optim.Adam(model.policy_parameters(), lr=lr_policy, eps=1e-5)
    opt_vae = torch.optim.Adam(model.vae_parameters(), lr=lr_vae)
    rollout = RolloutBuffer(device, is_discrete)
    vae_buf = VAEBuffer(vae_buffer_size, is_discrete)
    best = -float("inf")

    for update in range(1, total_updates + 1):
        envs = make_envs_train(sample_tasks_train(num_envs))
        last_v, roll_stats = collect_rollout(model, envs, rollout, horizon, device, is_discrete, action_low, action_high)
        envs.close()
        vae_buf.add(rollout.trajectories())

        rollout.compute_gae(last_v, gamma, lam, cut_returns_at_episode_end)
        ppo = ppo_update(model, opt_pi, rollout.flat(), is_discrete, action_low, action_high,
                         ppo_epochs, ppo_minibatches, clip, vf_coef, ent_coef, max_grad_norm)

        vae_stats: Dict[str, float] = {}
        model.train()
        for _ in range(vae_updates_per_iter):
            batch = vae_buf.sample(min(vae_batch, vae_buf.size), device)
            loss, s = model.vae_loss(batch, num_elbo, num_decode, kl_weight)
            opt_vae.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.vae_parameters(), max_grad_norm)
            opt_vae.step()
            for k, v in s.items():
                vae_stats[k] = vae_stats.get(k, 0.0) + v / vae_updates_per_iter

        msg = (f"[it {update:4d}/{total_updates}] train_ep_ret={roll_stats['mean_episode_return']:.1f} "
               f"({roll_stats['episodes']} eps) " + " ".join(f"{k}={v:.3g}" for k, v in {**ppo, **vae_stats}.items()))

        if update % eval_interval == 0:
            envs_e = make_envs_eval(sample_tasks_eval(eval_tasks_count))
            ev = run_meta_eval(model, envs_e, device, is_discrete, max_eval_steps, action_dim,
                               action_low, action_high, num_episodes=eval_episodes)
            envs_e.close()
            msg += " | eval return per episode: " + " ".join(f"{m:.1f}" for m in ev["mean"])
            if ev["mean"][0] > best:                       # first-episode return, the paper's metric
                best = float(ev["mean"][0])
                torch.save(model.state_dict(), save_path)
                msg += f" | new best, saved to {save_path}"
        log_fn(msg)

    try:
        model.load_state_dict(torch.load(save_path, map_location=device))
    except FileNotFoundError:
        pass
    return model
