# VariBAD-Torch: Bayes-Adaptive Deep RL via Meta-Learning

A PyTorch implementation of **VariBAD** (Zintgraf et al., ICLR 2020, *VariBAD: A Very Good Method for
Bayes-Adaptive Deep RL via Meta-Learning*, [arXiv:1910.08348](https://arxiv.org/abs/1910.08348)), laid
out like RL2-Torch-Experiments and using the **same three environments**, so RL², PEARL and VariBAD can
be compared on identical tasks.

| Experiment | Env | Task parameter | Actions |
|---|---|---|---|
| `experiments/cartpole_wind.py` | CartPole + wind force, reward `cos(pole angle)` | wind ∈ [-5, 5] | discrete (2) |
| `experiments/pendulum_wind.py` | Pendulum + wind torque | torque ∈ [-0.5, 0.5] | continuous, bounds ±2 |
| `experiments/halfcheetah_action.py` | HalfCheetah-v5 with additive action offset | offset ∈ [-0.2, 0.2] | continuous, bounds ±1 (6-D) |

The environment code (`CartPoleEnvWithWind`, `RewardShapingWrapper`, `sample_tasks_*`, `make_envs`, …) is copied unchanged from the RL² repo.

## The method in one picture

```
 transition (s, a, r, s', done) ──► GRU encoder ──► q(m | τ:t) = N(mu_t, exp(logvar_t))     prior at t=0: N(0, I)
                                                       │ (detached)                │ sample m
                                                       ▼                           ▼
                              policy π(a | s, mu_t, logvar_t)        decoders: reward  p(r_{i+1} | s_i, a_i, s_{i+1}, m)
                              trained with PPO                                 state   p(s_{i+1} | s_i, a_i, m)
                                                                      for ALL i in the trajectory (past and future)
```

* **The belief is carried over episode resets.** One BAMDP trajectory spans several episodes of the same
  task; the encoder gets a `done` flag so it knows when an episode ended, and its GRU state is *not* reset.
* **ELBO (Eq. 8–10):** `ELBO_t = E_q[log p(τ_{:H+} | m)] − KL(q(m|τ:t) ‖ q(m|τ:t−1))`, with `q(m|τ:0) = N(0, I)`.
  The sum over `t` is sub-sampled (`num_elbo` belief times per trajectory, `num_decode` reconstructed
  transitions each).
* **Separate optimisers and buffers (paper Sec. 3.2):** the RL loss is not back-propagated through the encoder
  (policy LR 7e-4, VAE LR 1e-3); PPO uses only the latest rollout, the VAE trains on a larger buffer of trajectories.
* Because the belief is computed once during the rollout and stored, **PPO minibatches need no recurrence**
  (cheaper than RL², see paper Appendix C.4).

## Structure

```
agents/
  nn_tools.py        orthogonal-init MLP (same as RL²)
  encoder.py         RNNEncoder (GRU, outputs mu/logvar after every transition), KL between Gaussians
  varibad.py         RewardDecoder, StateDecoder, BeliefActorCritic, VariBADModel (+ vae_loss = negative ELBO)
utils/
  buffer.py          RolloutBuffer (PPO + GAE, stores the belief), VAEBuffer (full BAMDP trajectories)
  rollout.py         belief tracking, reset without dummy steps, action sampling
  evaluation.py      run_meta_eval: online return of the first episode(s) on new tasks
  distributions.py   TanhNormal (same as RL²)     normalizers.py  (same as RL²)
train.py             train_varibad(...)   (counterpart of train_rl2_ppo)
experiments/         one script per environment
tests/               unit tests (pytest)
```

`train_varibad` takes the same factories as `train_rl2_ppo` (`model_factory`, `sample_tasks_train/eval`,
`make_envs_train/eval`, `action_dim`, `is_discrete`, `action_low/high`, `save_path`, `num_envs`, `horizon`, …).

## Install and run (uv)

This project uses [uv](https://docs.astral.sh/uv/) ([install uv](https://docs.astral.sh/uv/getting-started/installation/)). From the repo root:

```bash
uv sync                          # creates .venv, installs torch, numpy, gymnasium and pytest
uv sync --extra mujoco           # additionally installs mujoco (needed for HalfCheetah)
```

Train (one script per environment; the best policy is written to `checkpoints/`):

```bash
uv run python experiments/cartpole_wind.py
uv run python experiments/pendulum_wind.py
uv run python experiments/halfcheetah_action.py      # needs the mujoco extra
```

Tests:

```bash
uv run pytest                    # fast unit tests
uv run pytest -m slow            # ~1 min: trains only the VAE and checks that the posterior identifies the wind
```

Long runs on a server (keeps running after you disconnect):

```bash
nohup uv run python experiments/cartpole_wind.py > cartpole.log 2>&1 &
```

On a CPU-only machine, torch's default wheel may pull in large CUDA libraries. To use the CPU build, run
`uv pip install torch --index-url https://download.pytorch.org/whl/cpu` after `uv sync`, and start scripts with `uv run --no-sync`.

During training, every `eval_interval` updates the log prints the mean return per episode on held-out tasks
(`eval return per episode: ep1 ep2`). **Episode 1 is the paper's metric** (performance while the task is still
unknown); `checkpoints/<name>.pt` always holds the best policy so far by that number (overwritten only on
improvement; it is chosen on the same eval tasks it is scored on, so the number is optimistically biased).

Load a checkpoint (from the repo root, e.g. with `uv run python`):

```python
import torch
from experiments.cartpole_wind import cartpole_model_factory
model = cartpole_model_factory()
model.load_state_dict(torch.load("checkpoints/cartpole_wind.pt", map_location="cpu"))
```

The belief at any time is available from `model.encode(...)` / `utils.rollout.update_belief`, which is what
you need for monitoring or probing the task embedding.

## Implementation notes and deviations (please read)

* **A state decoder is used (the paper uses a reward decoder only on MuJoCo).** In all three environments the
  task changes the *dynamics*, while the reward is a function of the observation only, so a reward decoder
  `p(r | s, a, s', m)` carries no task information. `decode_state=True` (default in the factories) adds
  `p(s' | s, a, m)`, predicting the normalised state change. Set `decode_state=False` to get the paper's
  reward-only setup.
* **The encoder sees `s_t` and `s_{t+1}`**, not only `(a, r, s')` as in the paper's description of `τ`
  (`use_prev_state=True`), so a single GRU step already sees a full transition.
* **Episode ends.** Finished sub-envs are reset immediately with gymnasium's `reset_mask`
  (gymnasium ≥ 1.1) instead of the default NEXT_STEP autoreset, so there are no dummy steps (action ignored,
  reward 0) in the data. The stored `next_obs` is the real terminal observation, and the encoder receives a `done` flag.
* **Returns do not cross episode boundaries by default** (`cut_returns_at_episode_end=True`), exactly like the GAE
  in RL2-Torch-Experiments. This keeps the termination signal that makes CartPole learnable (its per-step reward
  `cos(angle)` is almost constant). The paper's BAMDP objective (Eq. 3) sums over the whole horizon `H+`; set
  `cut_returns_at_episode_end=False` for that.
* **Hyper-parameters.** From the paper's Appendix C.6 (MuJoCo): PPO clip 0.1, 2 epochs, 4 minibatches, value coef 0.5,
  entropy coef 0.01, grad-norm 0.5, Huber value loss, KL weight 0.1, policy LR 7e-4, VAE LR 1e-3, latent size 5,
  policy 2×128 tanh, encoder embeddings 32/16/16 + GRU 128, reward decoder 64/32 ReLU with MSE. **Not given in the
  paper PDF and chosen by me:** γ = 0.99, GAE λ = 0.95, 16 envs, horizon (1024 / 600 / 2000 steps), 3 VAE updates of 16
  trajectories per iteration, VAE buffer of 1000 trajectories, 50 ELBO terms × 50 decoded transitions, state-decoder size,
  normaliser sample limits. I did not consult the authors' code.
* **Normalisation.** Observations and rewards are normalised with the RL² `RunningMeanStd1DNormalizer` (updated during
  rollouts until `max_samples`, then frozen); all encoder / decoder inputs and decoder targets are in the normalised space.
* The prior at `t = 0` is a fixed `N(0, I)` (the paper notes that learning the prior is a natural extension). The policy
  conditions on the posterior parameters `(mu, logvar)`, not on a sample of `m`.
* Not included: the gridworld of Sec. 5.1, the Oracle / PEARL / ProMP / E-MAML baselines, and the Walker / Ant tasks.

## Status

* Unit tests pass (15 tests incl. the slow one). All three experiment scripts run end to end (HalfCheetah needs `mujoco`).
* **VAE check (slow test):** trained alone on near-random-policy CartPole-wind data, the posterior mean linearly predicts
  the wind with held-out R² ≈ 0.99 (one run, 25 × 16 trajectories, 600 updates).
* **CartPole wind, one run** (seed 0, 80 updates, about 4 min on 2 CPU cores, 8 evenly spaced evaluation winds):
  the first-episode eval return rose from about 93 (update 5) to about 499.9 (update 25) and stayed there; the maximum
  is about 500. Probing the trained policy on 64 random winds: the belief mean after 100 steps predicts the wind with
  held-out R² ≈ 0.88, but evaluating with the belief frozen at the prior gives a similar return (295.6 vs 300.0 over the first
  300 steps). So on CartPole wind the agent does infer the wind, but the task is easy enough that a wind-robust policy already
  reaches the ceiling, and this run does not show that the policy *uses* the belief. One seed; treat as a sanity check.
* Pendulum and HalfCheetah were only smoke-tested (one short update), not trained.

## Acknowledgements and references

* L. Zintgraf, K. Shiarlis, M. Igl, S. Schulze, Y. Gal, K. Hofmann, S. Whiteson. *VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning.* ICLR 2020. [arXiv:1910.08348](https://arxiv.org/abs/1910.08348)
* Environments, `TanhNormal`, normalizers, MLP helper and the factory-style trainer interface are taken from [RL2-Torch-Experiments](https://github.com/fatcatZF/RL2-Torch-Experiments).
