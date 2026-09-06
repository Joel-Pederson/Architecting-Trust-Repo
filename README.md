# Architecting Trust — Lunar Lander Proof of Concept

A MATLAB testbed for the **Safety Sidecar** architecture described in *Architecting Trust:
A Modular Framework for the Operational Deployment of Autonomous Systems*.

The paper argues that an autonomous system can be trusted operationally without trusting
its decision-making component, by wrapping it in a small, verifiable runtime barrier that
holds regardless of what the inner controller does. This repository is the concrete
demonstration: a full Apollo-class powered descent, flown by a neural policy, with a
formally simple barrier watching over it — and a fault-injection study showing what the
barrier is worth when the controller is wrong.

---

## The claim, and what evidence lives where

The architecture makes two claims that need **opposite** evidence, so there are two
experiments rather than one.

| Claim | Means | Script |
|---|---|---|
| **Non-intrusiveness** — a barrier must not degrade a competent controller | Identical landing rates with the sidecar attached and detached | `evaluate_final_agent` |
| **Necessity** — a barrier must save a controller that is wrong | Same controller, same scenario, corrupted sensor; barrier on vs off | `demo_sidecar_rescue`, `demo_agent_rescue`, `run_fault_injection_study` |

A healthy agent cannot demonstrate necessity — it never approaches the barrier, so nothing
happens. A faulty one cannot demonstrate non-intrusiveness. Both halves are needed.

---

## Requirements

- **MATLAB R2025a** (earlier releases have a different `trainFromData` / `rlReplayMemory`
  contract and are untested here)
- Reinforcement Learning Toolbox
- Deep Learning Toolbox
- Parallel Computing Toolbox (optional; used only by the training sweeps)

No GPU is required. Everything below was measured on Apple silicon in a single process.

---

## Quick start

```matlab
cd Lunar-Lander
addpath(genpath(pwd))
```

Every entry point calls `addpath(genpath(...))` on itself, so running one directly from a
fresh MATLAB session also works.

### 1. Check the install — 90 tests, no training required

```matlab
runtests('CI-tests', 'IncludeSubfolders', true)
```

This is the same suite CI runs. It exercises the physics, the reward landscape, the
barrier, the action interface, the fault models, the animator and the imitation pipeline.
It needs no trained network and takes a couple of minutes.

### 2. See the barrier work — still no training required

```matlab
demo_sidecar_rescue        % 10 m altimeter bias on the CLASSICAL pilot
```

Two flights, identical initial condition, identical controller. The pilot's altimeter
reads 10 m high, so it believes it has more room than it does and brakes late. It cannot
detect this: every sensor it has is self-consistent. The sidecar reads **true** state —
that split is the paper's Perception Gatekeeper boundary — and enforces one thing only:

```
h_alt = y − dy² / (2·a_max)     % altitude minus the distance needed to stop at full thrust
```

Guardian off: crash. Guardian on: landing. **This is the paper's core result and it
involves no machine learning at all.**

### 3. Everything past this point needs a trained network

`.mat` files are gitignored (see `.gitignore`), so a fresh clone has **no** agent and **no**
demonstration dataset. `run_trained_agent`, `demo_agent_rescue` and `demo_reel` will raise
an informative error until you build one. That is the next section.

---

## Training from a fresh clone

One command rebuilds everything:

```matlab
train_pipeline          % ~2 hours end to end
```

It runs four stages, each of which writes an artefact to `Lunar-Lander/` and can be resumed
independently with `train_pipeline(struct('stages', 3:4))`.

| # | Stage | Produces | ~Time |
|---|---|---|---|
| 1 | **Demonstrate** — fly the classical guidance law, record every (state, action) pair | `demonstrations.mat` | 20 min |
| 2 | **Clone** — fit 8 TD3 actors by regression, keep the one that flies best | `cloned_agent_4phase.mat` | 20 min |
| 3 | **DAgger** — roll out the clone, label the states *it* visits with expert actions | `dagger_corpus.mat` | 60 min |
| 4 | **Select** — re-clone from the aggregated corpus, screen wide then verify a shortlist | `cloned_agent_4phase.mat` | 15 min |

Then verify and watch:

```matlab
evaluate_final_agent                % the headline table, 30 episodes per cell
run_trained_agent('orbit')          % animate a full powered descent
```

### Why the pipeline has this shape

Each stage exists because the simpler thing was tried and measured to fail.

**Why not just train an agent?** Four architectures (DDPG, TD3, SAC, PPO) at 1200 episodes
each, plus two longer runs — roughly 25,000 episodes — produced **zero landings** under
greedy evaluation, in three distinct and separately diagnosed failure modes. The reward
landscape was verified on five independent properties, and the task is demonstrably
solvable: the classical controller in `core/` lands **100% of all four phases at n=30 per
phase** (mean impact 0.25 / 0.28 / 0.29 / 0.28 m/s) through the same `[-1,1]` action
interface the agents were given. The gap is **exploration**. A landing requires a
coordinated descent, lateral null and square-up, and undirected action sequences never
produce one.

**Why DAgger and not more demonstrations?** Plain cloning plateaued at P1 100% / P2 88% /
P3 88% / **P4 25%**. The failure scales with *horizon*, not difficulty: a Phase 4 powered
descent is ~8,900 agent steps against Phase 1's ~450, so accumulated action error has
twenty times the exposure before touchdown. More expert data does not help, because it all
lies on the expert's trajectory and the clone's problem is *everywhere else*. Nor does
capacity — 512-unit networks lowered validation RMSE from 0.131 to 0.121 while the best
landing rate **fell** from 72% to 66%.

**Why β-mixing inside DAgger?** Pure clone rollouts from round one moved Phase 4 not at
all, across two rounds and 137,000 corrective transitions. Over an 8,900-step descent the
clone drifts somewhere genuinely unrecoverable, and the expert's label at such a state
teaches nothing — no action recovers a vehicle 200 km downrange with the wrong energy.
Mixing keeps early rounds near the expert's distribution, where recovery is still possible.

**Why select on landing rate and not validation loss?** Across candidates the correlation
between validation RMSE and landing rate was **−0.021**. An epoch sweep had RMSE falling
monotonically 0.184 → 0.107 while landing rate bounced 43 / 10 / 57 / 47 / 3 / 20 percent.
Regression error selects a policy that hovers.

**Why screen-then-verify?** Taking the maximum of 8 noisy n=8 screens is the winner's
curse, and it bit: a clone that screened at 88% on Phase 4 scored **33%** when re-measured
at n=30. Stage 4 screens wide and cheap, then re-measures a shortlist on fresh seeds.

Retraining will not reproduce the published impact figures digit for digit — candidate
selection draws different initialisations. The landing rates should reproduce.

---

## The flight envelope

Four curriculum phases, addressed by name everywhere (`core/phase_from_name.m`):

| Name | Start | Steps | What it is |
|---|---|---|---|
| `'touchdown'` | 50 m, 2 m/s | ~450 | Final touchdown only |
| `'approach'` | 500 m, 10 m/s | ~1,200 | Glide-slope approach |
| `'terminal'` | 2.5 km, 25 m/s | ~2,500 | Full terminal descent |
| `'orbit'` | 15.2 km, 1697 m/s, 550 km downrange | ~8,900 | Apollo powered descent from PDI |

Numeric indices still work, but `run_trained_agent('orbit')` says what it does and
`run_trained_agent(true, struct('phase', 4))` did not.

---

## Reproducing the paper's figures

```matlab
demo_reel                       % the whole argument as four animated scenarios
demo_agent_rescue('orbit')      % the sidecar rescuing the NEURAL agent from a blind altimeter
run_fault_injection_study       % the full fault sweep, ~5 min
evaluate_final_agent            % the non-intrusiveness table
```

`demo_reel` plays, in order: a healthy guarded flight (the barrier is almost silent); the
same fault unguarded (it crashes); the same fault guarded (it lands); and a from-scratch
DDPG agent whose actor collapsed to a constant — shown because the negative result is part
of the argument about what the barrier has to contain.

---

## Results

### Non-intrusiveness — 30 episodes per cell, 240 episodes total

| phase | guardian ON |  |  | guardian OFF |  |  |
|---|---|---|---|---|---|---|
| | land% | impact | worst | land% | impact | worst |
| 1 | 100% | 0.24 | 0.26 | 100% | 0.26 | 0.35 |
| 2 | 100% | 0.21 | 0.21 | 100% | 0.30 | 0.38 |
| 3 | 100% | 0.19 | 0.20 | 100% | 0.18 | 0.20 |
| 4 | 100% | 0.46 | 0.50 | 100% | 0.54 | 0.60 |

Zero failures. Worst touchdown impact 0.60 m/s against a 1.0 m/s limit — including 60
complete powered descents from 15.2 km and 1697 m/s across 550 km of downrange. The
barrier costs nothing when the controller is competent, and stays inactive for the whole
flight.

### Necessity — fault injection on the classical pilot

`run_fault_injection_study` sweeps four fault models × five magnitudes × three phases,
25 episodes per cell, guardian on and off — 60 cells, 3,000 episodes, about five minutes.

**Excluding the control-delay fault** (the architecture's known boundary, below), across
the 36 perception and actuator cells with a non-zero fault:

| | guardian OFF | guardian ON |
|---|---|---|
| mean landing rate | 35.4% | **99.4%** |
| worst touchdown impact | 87.07 m/s | **0.76 m/s** |
| cells within the 1.0 m/s limit | — | **36 of 36** |

Phase 3 in detail (the most demanding phase swept — a 2.5 km terminal descent):

| fault | land% OFF | worst OFF | land% ON | worst ON |
|---|---|---|---|---|
| altimeter high by 5 m | 0% | 1.34 | **100%** | 0.52 |
| altimeter high by 10 m | 0% | 1.84 | **100%** | 0.53 |
| altimeter high by 20 m | 0% | 2.56 | **100%** | 0.52 |
| altimeter high by 40 m | 0% | 3.58 | **100%** | 0.53 |
| descent rate under-read 0.5 | 20% | 1.51 | **96%** | 0.49 |
| descent rate under-read 0.7 | 0% | 28.21 | **96%** | 0.53 |
| descent rate under-read 0.9 | 0% | 87.07 | **96%** | 0.76 |
| engine down 30% | 64% | 1.01 | **100%** | 0.50 |
| engine down 40% | 0% | 1.40 | **100%** | 0.50 |
| control delayed 10 steps | 0% | 2.00 | 60% | 0.45 |
| control delayed 20 steps | 0% | 69.29 | 0% | 60.63 |
| control delayed 40 steps | 0% | 129.18 | 0% | 113.41 |

The barrier does not know where the pad is and is not trying to fly better than the pilot.
It enforces one thing — never enter a state from which no recovery exists — and that alone
converts a total loss into a landing across every perception and actuator fault swept.

**The six cells it does not bound are all control delay ≥ 20 steps.** That is a real limit
and it is in the table deliberately: the barrier reacts to true state, but it cannot act
earlier than the actuator responds. Beyond roughly 2 seconds of latency there is no
recovery available to enforce, and the architecture has nothing to offer.


### Negative results worth keeping

These are in the repository deliberately; the architecture's boundary conditions are part
of the argument.

- **Zero-mean sensor noise does not discriminate.** Injecting Gaussian action noise up to
  σ = 0.5 left landing rates at ~100% with *and* without the guardian. A PD loop
  re-deciding at 10 Hz rejects zero-mean disturbance by construction. Every fault model in
  `core/apply_sensor_fault.m` is therefore **systematic** — something the controller can
  neither see nor correct.
- **Control delay beyond ~20 steps is unrecoverable.** The barrier reacts to true state,
  but it cannot act earlier than the actuator responds. This is a real limit of the
  architecture, not a tuning failure.
- **Severe velocity under-read used to be unrecoverable, and no longer is.** Before the
  regime-aware barrier rework, a 0.7 under-read went 0% → 0% with the guardian. It now goes
  0% → 96%, and a 0.9 under-read — an 87 m/s impact unguarded — lands at 0.76 m/s. The
  barrier and the guidance law both changed between those two measurements, so the credit
  is not cleanly attributable to one fix; the largest candidate is that the barrier was
  computing available deceleration as `T_max·cos(θ)`, which goes negative past 90°.
- **From-scratch RL never landed.** Four architectures, ~25,000 episodes, three distinct
  failure modes. `run_algorithm_trade.m` reproduces the comparison.

---

## Repository map

```
Lunar-Lander/
  core/                          the plant and the classical stack
    get_sim_params.m             SINGLE SOURCE OF TRUTH for every physical constant
    lunar_lander_dynamics.m      6-DOF-in-plane physics, curvilinear (flat-Moon) frame
    scripted_pilot.m             terminal-phase guidance law (the expert)
    braking_guidance.m           Apollo P63 braking; blends smoothly into scripted_pilot
    safety_sidecar_filter.m      THE BARRIER. Regime-aware, reads true state only.
    action_to_command.m          the [-1,1] <-> thrust interface, defined once
    command_to_action.m          its inverse
    apply_sensor_fault.m         systematic fault models
    get_ai_observation.m         signed-log observation normalisation
    phase_from_name.m            scenario names -> curriculum indices
    animate_lunar_lander.m       the visualiser
    main_simulation.m            classical test bench

  RL-training-harness/
    LunarLanderEnv.m             the rl environment (classdef)
    generate_demonstrations.m    stage 1
    pretrain_actor_supervised.m  stage 2 / 4 regression
    dagger_refine.m              stage 3
    run_fault_injection_study.m  the necessity experiment
    run_algorithm_trade.m        the four-architecture negative result
    agent_architectures/         DDPG / TD3 / SAC / PPO builders

  CI-tests/                      90 tests across 18 files
  train_pipeline.m               rebuild the agent from nothing
  evaluate_final_agent.m         the headline table
  run_trained_agent.m            watch one episode
  demo_sidecar_rescue.m          barrier vs faulty CLASSICAL pilot
  demo_agent_rescue.m            barrier vs faulty NEURAL agent
  demo_reel.m                    all of it, in sequence
```

**The barrier itself is `core/safety_sidecar_filter.m`.** It is a few hundred lines of
deterministic arithmetic with no learned components and no state, which is the point: it is
small enough to argue about directly, and that is what makes the wrapped system trustable
when the thing inside it is not.

---

## Continuous integration

`.github/workflows/matlab-tests.yaml` runs the full suite on every push and pull request.
Failures are republished as GitHub `::error::` annotations by
`.github/scripts/run_ci_suite.m`, because job logs need admin rights to read while
annotations are public — a red run should be diagnosable from a fork.

---

## Licence

See [LICENSE](LICENSE).
