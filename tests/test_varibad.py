"""Unit tests.  Fast tests: ``uv run pytest``.  Slow end-to-end check: ``uv run pytest -m slow``."""
import numpy as np
import pytest
import torch
from torch.distributions import Normal, kl_divergence

import cartpole_wind as cp
import pendulum_wind as pw
from agents.encoder import kl_diag_gaussians
from agents.varibad import VariBADModel, _gather_time
from train import collect_rollout
from utils.buffer import RolloutBuffer, VAEBuffer
from utils.evaluation import run_meta_eval
from utils.rollout import initial_belief, reset_done_envs, update_belief

CPU = torch.device("cpu")


def _model(discrete=True, **kw):
    obs_dim, act_dim = (4, 2) if discrete else (3, 1)
    return VariBADModel(obs_dim, act_dim, discrete, latent_dim=3, **kw)


def _batch(model, T=12, B=5, discrete=True):
    d = model.obs_dim
    act = torch.randint(0, model.action_dim, (T, B)) if discrete else torch.rand(T, B, model.action_dim)
    return dict(obs=torch.randn(T, B, d), act=act, rew=torch.randn(T, B, 1), next_obs=torch.randn(T, B, d),
                done=(torch.rand(T, B, 1) < 0.2).float())


def test_kl_matches_torch_distributions():
    m1, l1, m0, l0 = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
    ref = kl_divergence(Normal(m1, (0.5 * l1).exp()), Normal(m0, (0.5 * l0).exp())).sum(-1)
    assert torch.allclose(kl_diag_gaussians(m1, l1, m0, l0), ref, atol=1e-5)


def test_gather_time():
    x = torch.arange(24).view(4, 3, 2)
    idx = torch.tensor([[0, 3, 1], [2, 2, 0]])
    out = _gather_time(x, idx)
    assert out.shape == (2, 3, 2)
    assert (out[1, 0] == x[2, 0]).all() and (out[0, 1] == x[3, 1]).all()


@pytest.mark.parametrize("discrete", [True, False])
def test_stepwise_belief_equals_sequence_encoding(discrete):
    """Online belief updates during rollouts must match the full-sequence encoding used for the ELBO."""
    model = _model(discrete)
    b = _batch(model, T=9, B=3, discrete=discrete)
    with torch.no_grad():
        mu_seq, lv_seq, _ = model.encode(b["obs"], b["act"], b["rew"], b["next_obs"], b["done"])
        mu, lv, h = initial_belief(model, 3, CPU)
        for t in range(9):
            mu, lv, h = update_belief(model, h, b["obs"][t].numpy(), b["act"][t].numpy(), b["rew"][t, :, 0].numpy(),
                                      b["next_obs"][t].numpy(), b["done"][t, :, 0].numpy(), CPU)
            assert torch.allclose(mu, mu_seq[t], atol=1e-5) and torch.allclose(lv, lv_seq[t], atol=1e-5)


def test_prior_belief():
    mu, lv, h = _model().prior(4, CPU)
    assert mu.shape == lv.shape == (4, 3) and mu.abs().sum() == 0 and lv.abs().sum() == 0 and h.shape == (1, 4, 128)


@pytest.mark.parametrize("discrete", [True, False])
def test_vae_loss_gradients(discrete):
    model = _model(discrete)
    loss, stats = model.vae_loss(_batch(model, discrete=discrete), num_elbo=4, num_decode=4)
    assert torch.isfinite(loss) and {"reward_loss", "state_loss", "kl"} <= set(stats)
    loss.backward()
    for name, mod in [("encoder", model.encoder), ("reward_decoder", model.reward_decoder), ("state_decoder", model.state_decoder)]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()), name
    assert all(p.grad is None for p in model.policy.parameters()), "RL/ELBO losses are separate: policy gets no ELBO gradient"


def test_state_decoder_is_optional():
    model = _model(decode_state=False)
    assert model.state_decoder is None
    loss, stats = model.vae_loss(_batch(model), 3, 3)
    assert "state_loss" not in stats and torch.isfinite(loss)


def test_policy_input_is_detached_belief():
    """The belief handed to the policy is a plain tensor: RL gradients cannot reach the encoder."""
    model = _model()
    mu, lv, h = initial_belief(model, 2, CPU)
    mu, lv, h = update_belief(model, h, np.zeros((2, 4), np.float32), np.array([0, 1]), np.zeros(2), np.ones((2, 4), np.float32),
                              np.zeros(2), CPU)
    assert not mu.requires_grad and not lv.requires_grad


def test_gae_cut_and_no_cut():
    buf = RolloutBuffer(CPU, True)
    rews = [1.0, 1.0, 1.0]
    dones = [0.0, 1.0, 0.0]
    for r, d in zip(rews, dones):
        buf.add(obs=np.zeros((1, 4)), act=np.zeros(1), raw_act=np.zeros(1), logp=np.zeros(1), val=np.zeros(1), rew=np.array([r]),
                term=np.array([d]), done=np.array([d]), next_obs=np.zeros((1, 4)), mu=np.zeros((1, 2)), logvar=np.zeros((1, 2)))
    buf.compute_gae(torch.tensor([5.0]), gamma=0.5, lam=1.0, cut_at_episode_end=True)
    # t=2: 1 + .5*5 = 3.5 ; t=1 (done): 1 ; t=0: 1 + .5*1 = 1.5
    assert torch.allclose(buf.ret[:, 0], torch.tensor([1.5, 1.0, 3.5]))
    buf.compute_gae(torch.tensor([5.0]), gamma=0.5, lam=1.0, cut_at_episode_end=False)
    assert torch.allclose(buf.ret[:, 0], torch.tensor([1 + .5 * (1 + .5 * 3.5), 1 + .5 * 3.5, 3.5]))


def test_reset_done_envs_has_no_dummy_step():
    envs = cp.make_envs([{"wind_mag": 5.0}, {"wind_mag": -5.0}])
    obs, _ = envs.reset(seed=0)
    for _ in range(200):
        nxt, rew, term, trunc, _ = envs.step(np.array([1, 1]))          # always push right: falls soon
        done = np.logical_or(term, trunc)
        if done.any():
            assert (np.abs(nxt[done][:, 2]) > 0.2).all(), "next_obs must be the real terminal state"
            new = reset_done_envs(envs, obs, nxt, done)
            assert (np.abs(new[done]) < 0.06).all(), "done envs restart from a fresh initial state"
            assert np.allclose(new[~done], nxt[~done])
            nxt2, rew2, *_ = envs.step(np.array([1, 1]))
            assert (rew2[done] > 0.9).all(), "first step after the reset is a real step (cos(angle) ~ 1), not a reward-0 dummy"
            return
        obs = nxt
    pytest.fail("no episode ended")


def test_vae_buffer_roundtrip():
    vb = VAEBuffer(3, True)
    T, N = 6, 2
    for _ in range(3):   # 6 trajectories into capacity 3 -> ring buffer
        vb.add(dict(obs=np.random.randn(T, N, 4), act=np.random.randint(0, 2, (T, N)), rew=np.random.randn(T, N, 1),
                    next_obs=np.random.randn(T, N, 4), done=np.zeros((T, N, 1))))
    assert vb.size == 3
    b = vb.sample(5, CPU)
    assert b["obs"].shape == (T, 5, 4) and b["act"].dtype == torch.int64 and b["rew"].shape == (T, 5, 1)


@pytest.mark.parametrize("env", ["cartpole", "pendulum"])
def test_rollout_and_eval(env):
    mod, disc = (cp, True) if env == "cartpole" else (pw, False)
    model = mod.cartpole_model_factory() if env == "cartpole" else mod.pendulum_model_factory()
    low, high = (None, None) if disc else (-2.0, 2.0)
    tasks = mod.sample_tasks_train(3)
    envs = mod.make_envs(tasks)
    buf = RolloutBuffer(CPU, disc)
    last_v, stats = collect_rollout(model, envs, buf, 60, CPU, disc, low, high)
    assert len(buf) == 60 and last_v.shape == (3,)
    tr = buf.trajectories()
    assert tr["obs"].shape[:2] == (60, 3) and tr["rew"].shape == (60, 3, 1)
    buf.compute_gae(last_v, 0.99, 0.95)
    flat = buf.flat()
    assert flat["obs"].shape[0] == 180 and flat["adv"].shape == (180,) and flat["mu"].shape == (180, 5)
    envs.close()
    ev = run_meta_eval(model, mod.make_envs(mod.sample_tasks_eval(3)), CPU, disc, 600, 2 if disc else 1, low, high, num_episodes=1)
    assert ev["returns"].shape == (3, 1)


@pytest.mark.slow
def test_posterior_identifies_wind_from_random_policy_data():
    """Train only the VAE on CartPole-wind data and check that the posterior mean linearly predicts the wind."""
    torch.manual_seed(0); np.random.seed(0)
    model = cp.cartpole_model_factory()
    buf, vb = RolloutBuffer(CPU, True), VAEBuffer(300, True)
    winds = []
    for _ in range(19):
        tasks = cp.sample_tasks_train(16)
        envs = cp.make_envs(tasks)
        collect_rollout(model, envs, buf, 150, CPU, True)
        envs.close()
        vb.add(buf.trajectories()); winds += [t["wind_mag"] for t in tasks]
    opt = torch.optim.Adam(model.vae_parameters(), lr=1e-3)
    for _ in range(400):
        loss, _ = model.vae_loss(vb.sample(16, CPU), 30, 30, 0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        b = {k: torch.as_tensor(v[:vb.size]).transpose(0, 1) for k, v in vb.store.items()}
        mu, _, _ = model.encode(b["obs"], b["act"], b["rew"], b["next_obs"], b["done"])
    w = np.array(winds)[:vb.size]
    X = np.c_[mu[-1].numpy(), np.ones(len(w))]
    coef = np.linalg.lstsq(X[:200], w[:200], rcond=None)[0]
    pred = X[200:] @ coef
    r2 = 1 - ((w[200:] - pred) ** 2).sum() / ((w[200:] - w[200:].mean()) ** 2).sum()
    assert r2 > 0.6, f"held-out R^2 of wind from posterior mean is only {r2:.2f}"
