# Architecting Trust — Lunar Lander Proof of Concept

A MATLAB testbed for the **Safety Sidecar** architecture described in *Architecting Trust:
A Modular Framework for the Operational Deployment of Autonomous Systems*.

A neural policy flies a full Apollo-class powered descent — 15.2 km, 1697 m/s, 550 km of
downrange — wrapped in a small runtime barrier that holds regardless of what the policy
does. A fault-injection study shows what the barrier is worth when the controller is wrong.

**This README is a walkthrough for someone who has just cloned the repo.** Results and the
reasoning behind the design are further down, after the instructions.

---

## Shortest path to seeing it work

```matlab
cd Lunar-Lander
addpath(genpath(pwd))
demo_sidecar_rescue          % needs no trained network — this is the paper's core result
```

Two flights, one crash, one landing, difference is the barrier. Nothing to train.

Everything involving the **neural agent** needs about two hours of training first, because
all `.mat` files are gitignored. That is [step 4](#4-train-the-networks--2-hours).

---

## Requirements

| | |
|---|---|
| MATLAB | **R2025a**. Earlier releases have a different `trainFromData` / `rlReplayMemory` contract and are untested. |
| Toolboxes | Reinforcement Learning, Deep Learning. Parallel Computing is optional (training sweeps only). |
| Hardware | No GPU needed. Everything below was measured on Apple silicon, single process. |
| Disk | ~150 MB for the generated datasets and agents (`dagger_corpus.mat` is the big one at ~50 MB). |

Check what you have:

```matlab
ver          % look for Reinforcement Learning Toolbox and Deep Learning Toolbox
```

If a toolbox is missing, steps 1–3 still work — they use no learning at all. Only step 4
onward needs them.

---

## The walkthrough

### 1. Clone and open

```bash
git clone git@github.com:Joel-Pederson/Architecting-Trust-Repo.git
cd Architecting-Trust-Repo
```

Then, in MATLAB:

```matlab
cd Lunar-Lander
addpath(genpath(pwd))
```

Every entry point calls `addpath(genpath(...))` on itself, so running one directly from a
fresh session also works. The `addpath` above just saves repeating it.

### 2. Verify the install — 99 tests, no training required

```matlab
runtests('CI-tests', 'IncludeSubfolders', true)
```

Expect **99 passed, 0 failed** in about two minutes. This is the same suite CI runs. It
exercises the physics, the reward landscape, the barrier, the action interface, the fault
models, the animator and the imitation pipeline — none of which need a trained network.

To run one file while working on it:

```matlab
runtests('CI-tests/safety_sidecar_test.m')
```

### 3. See the barrier work — still no training required

```matlab
demo_sidecar_rescue          % 10 m altimeter bias on the CLASSICAL pilot
```

```
=== SAFETY SIDECAR RESCUE DEMO ===
Fault: altimeter reads 10 m HIGH. The pilot cannot detect this.
Identical scenario and controller in both runs; only the barrier differs.

  guardian OFF : crashed   impact   1.39 m/s  (dy  -1.39, dx  +0.06)  vetoes 0
  guardian ON  : landed    impact   0.47 m/s  (dy  -0.46, dx  +0.05)  vetoes 153

Touchdown limits: |dy| <= 1.0 m/s, |dx| <= 0.5 m/s, |theta| <= 0.10 rad
```

Two animation windows open, one per run. Same initial condition, same controller; the
pilot's altimeter reads 10 m high so it brakes late. It cannot detect this — every sensor
it has is self-consistent. The sidecar reads **true** state (the paper's Perception
Gatekeeper boundary) and enforces one thing:

```
h_alt = y − dy² / (2·a_max)     % altitude minus the distance needed to stop at full thrust
```

Variations:

```matlab
demo_sidecar_rescue(20)          % harsher fault
demo_sidecar_rescue(10, false)   % numbers only, no animation windows
```

**This is the core result and it involves no machine learning at all.** If you only run one
thing, run this.

### 4. Train the networks — ~2 hours

Everything past this point needs a trained agent. `.mat` files are gitignored (see
[.gitignore](.gitignore)), so a fresh clone has **no** agent and **no** demonstration
dataset. `run_trained_agent`, `evaluate_final_agent` and `demo_agent_rescue` raise a
`:NoAgent` error naming this step until you do it; `demo_reel` skips the scenarios it
cannot fly and plays the rest.

```matlab
train_pipeline
```

Four stages, each writing an artefact into `Lunar-Lander/`:

| # | Stage | Writes | ~Time |
|---|---|---|---|
| 1 | **Demonstrate** — fly the classical guidance law, record every (state, action) pair | `demonstrations.mat` | 20 min |
| 2 | **Clone** — fit 8 TD3 actors by regression, keep the one that flies best | `cloned_agent_4phase.mat` | 20 min |
| 3 | **DAgger** — roll out the clone, label the states *it* visits with expert actions | `dagger_corpus.mat` | 60 min |
| 4 | **Select** — re-clone from the corpus, screen wide then verify a shortlist | `cloned_agent_4phase.mat` | 15 min |

It prints a running table and is safe to leave unattended. Stage 3 dominates because each
round flies complete powered descents (~8,900 steps each).

**Resuming.** Stages are independent as long as their input artefact exists:

```matlab
train_pipeline(struct('stages', 3:4))     % keep the dataset and seed clone, redo DAgger
train_pipeline(struct('stages', 4))       % just re-select from an existing corpus
```

**Going faster** (noisier selection, worse Phase 4 — fine for a smoke test, not for a
result you would quote):

```matlab
train_pipeline(struct('n_candidates', 3, 'dagger_rounds', 2))
```

### 5. Check what you trained

```matlab
evaluate_final_agent
```

30 episodes per phase, guardian on and off, worst case reported alongside the mean. Takes
about six minutes. Compare against the reference table under [Results](#results) — landing
rates should reproduce, the impact figures will drift, because candidate selection draws
different initialisations every time.

### 6. Watch it fly

```matlab
run_trained_agent('orbit')          % the full powered descent from 15.2 km
run_trained_agent('terminal')       % 2.5 km terminal descent
run_trained_agent('approach')       % 500 m glide slope
run_trained_agent('touchdown')      % 50 m final touchdown
```

```
Agent: cloned_agent_4phase.mat   sidecar ON
Found a landed on attempt 1 of 25.
  outcome landed | touchdown dy -0.45 m/s, dx +0.04 m/s, theta +0.001 rad
  duration 1084.8 s | sidecar engagements 11
```

An animation window opens. Options:

```matlab
run_trained_agent('orbit', struct('sidecar', 'off'))    % barrier detached
run_trained_agent('orbit', struct('show', 'any'))       % next episode, pass or fail
run_trained_agent('orbit', struct('animate', false))    % numbers only
```

Scenario names are resolved by [core/phase_from_name.m](Lunar-Lander/core/phase_from_name.m);
numeric indices 1–4 still work.

### 7. Reproduce the paper's experiments

```matlab
demo_reel                       % the whole argument as four animated scenarios
demo_agent_rescue('orbit')      % the barrier rescuing the NEURAL agent from a blind altimeter
run_fault_injection_study       % the full fault sweep, ~5 min, 3000 episodes
run_algorithm_trade             % the from-scratch RL negative result, hours
```

`demo_reel` plays, in order: a healthy guarded flight (the barrier is almost silent); the
same fault unguarded (crash); the same fault guarded (landing); and a from-scratch DDPG
agent whose actor collapsed to a constant. That fourth scenario needs an agent from
`run_algorithm_trade` and is skipped silently if you have not run it; scenarios 1–3 carry
the actual argument and need only `train_pipeline`.

### 8. Build the barrier as flight code — no MATLAB needed

```bash
cd Lunar-Lander/flight-code/harness
make check      # does the generated C reproduce the MATLAB barrier?
make demo       # watch the barrier take control, step by step
make bench      # how long one evaluation takes
```

That is a C compiler and nothing else. See [Flight code](#flight-code-the-barrier-as-c)
below for what it is and why it exists.

---

## Flight code: the barrier as C

The architecture's claim is that a **small verifiable component can bound a large
unverifiable one**. Everything above measures whether the barrier *works*. This part
measures whether "small" is true — because in the paper that word is doing real work, and
until now it was an adjective rather than a number.

### What it is

`flight-code/` holds the Safety Sidecar barrier, generated by MATLAB Coder as standalone
portable C, committed to the repository, and buildable by anyone with a C compiler and no
MATLAB licence at all.

| | barrier | the policy it wraps |
|---|---|---|
| what a reviewer reads | **448 lines of C** | **69,122 learnable parameters** |
| memory | static, zero dynamic allocation | tensor runtime |
| configuration | none — constants compiled in | weights file |
| auditable by a human | in an afternoon | no |

That contrast *is* the architecture's argument, stated in numbers rather than adjectives.
The entry point carries no configuration at all, because `params` is passed to the code
generator as a constant and every physical value is compiled in as a literal:

```c
void safety_sidecar_filter(const double x[8], const double u_nominal[2],
                           double u_actual[2], boolean_T *VetoTriggered,
                           double *h_alt, double *h_fuel);
```

### How to run it

```bash
cd Lunar-Lander/flight-code/harness
make check
```

Requires a C99 compiler and `make`, nothing else. The build links `-lm` explicitly and
declares `_POSIX_C_SOURCE` for the timing harness — both needed on glibc, both implicit on
macOS, and both discovered the way portability problems always are: by building it
somewhere other than where it was written.

```
  rows compared        : 400
  vetoes  C / MATLAB   : 19 / 19
  worst |d command|    : 1.019e-10 N
  veto decisions differing : 0
  RESULT: the generated C reproduces the MATLAB barrier
```

The fixture is the 10 m altimeter-bias episode from step 3 — chosen because the barrier
**actually fires** there. An equivalence check flown on a healthy controller would agree
perfectly while exercising none of the code that matters.

`make demo` prints the descent step by step, showing the pilot's command and what the
barrier allowed. `make bench` reports 28.8 ns per evaluation against a 100 ms control
period. Throughput is irrelevant at 10 Hz; the number matters because *bounded* execution
time is a verification property, and a component that cannot overrun its slot is one fewer
thing a safety case has to argue about.

To regenerate the C after changing the barrier, from MATLAB:

```matlab
generate_flight_code(struct('verify', true))
```

### What this proves, and what it does not

It shows the barrier contains nothing that resists static compilation: no dynamic
allocation, no unbounded loops, no recursion, no external dependencies, and — since the
sentinel rework — no IEEE special-case handling. That is a checkable property, and it is
the one a certification-minded reader cares about.

It does **not** prove the barrier is correct. Code generation is a translation; a wrong
barrier compiles just as cleanly. The claim that the barrier bounds the vehicle rests on
the fault-injection study above, not on what language it is written in. This is
**deployability** evidence and should not be presented as verification evidence.

### The exercise found a real defect

Worth recording, because it is the argument for doing this at all rather than asserting
the barrier is simple. The first generation attempt emitted **18 files instead of 9**: the
barrier used `Inf` as a sentinel in three places, which forces MATLAB Coder to emit
`rt_nonfinite.c`, `rtGetInf.c`, `rtGetNaN.c` and the IEEE special-case paths that go with
them. Flight-software review generally objects to `Inf` and `NaN` in the first place.

Two of the three were only ever *compared*, so finite equivalents are exact. The third was
`t_stop = Inf` **multiplied** by a burn rate, where a large finite value would have
overflowed straight back to `Inf`, so that branch sets the fuel demand directly instead.
The rework was verified bit-identical across 200,000 randomised states covering the
regimes the infinities lived in — zero differing veto decisions, zero output difference.

Reading the file would not have surfaced that. Attempting to compile it did.

### Two caveats before publishing

- Files generated under an **academic MATLAB licence** carry an "Academic License … not
  for government, commercial, or other organizational use" header. It is preserved
  verbatim in the committed sources and does constrain how the C may be presented.
- `rtwtypes.h` is **replaced** by `generate_flight_code.m` with a self-contained
  equivalent. Coder's own version ends in `#include "tmwtypes.h"`, a MathWorks header
  outside this repository, which would defeat the point of shipping the artefact.

---

## Command reference

| Command | Needs a trained agent? | Time | What it does |
|---|---|---|---|
| `runtests('CI-tests','IncludeSubfolders',true)` | no | 2 min | the CI suite, 99 tests |
| `demo_sidecar_rescue` | no | 30 s | barrier vs faulty **classical** pilot |
| `main_simulation('HARDCODED_PILOT')` | no | 30 s | classical test bench, writes a telemetry plot |
| `run_fault_injection_study` | no | 5 min | 60-cell fault sweep, the necessity experiment |
| `train_pipeline` | builds one | 2 h | rebuild the agent from nothing |
| `evaluate_final_agent` | yes | 6 min | the headline table, both guardian arms |
| `run_trained_agent('orbit')` | yes | 1 min | animate one episode |
| `demo_agent_rescue('orbit')` | yes | 2 min | barrier vs faulty **neural** agent |
| `demo_reel` | yes | 5 min | all four scenarios in sequence |
| `run_algorithm_trade` | builds four | hours | DDPG / TD3 / SAC / PPO from scratch |
| `train_rl_agent('td3')` | builds one | ~1 h | train a single architecture from scratch |
| `train_from_demonstrations` | builds one | ~1 h | reproduces the offline-RL negative result |
| `generate_flight_code` | no | 30 s | regenerate the C from the MATLAB barrier |
| `make check` (in `flight-code/harness`) | no | 2 s | C-vs-MATLAB equivalence, **no MATLAB** |

---

## Where things get written

Everything lands in `Lunar-Lander/`, and all of it is gitignored:

| File | Written by |
|---|---|
| `demonstrations.mat` | `train_pipeline` stage 1 |
| `cloned_agent_4phase.mat` | `train_pipeline` stages 2 and 4 — **the agent every demo loads** |
| `dagger_corpus.mat` | `train_pipeline` stage 3 |
| `fault_injection_results.mat` | `run_fault_injection_study` |
| `algorithm_trade_results.mat`, `trade_agent_*.mat` | `run_algorithm_trade` |
| `core/Flight_Logs/telemetry_*.png` | `main_simulation` |

To point an entry point at a different agent:

```matlab
evaluate_final_agent(struct('agent_file', 'my_other_agent.mat'))
```

---

## Troubleshooting

**`Agent file not found: cloned_agent_4phase.mat`**
Expected on a fresh clone — the `.mat` files are gitignored. Run `train_pipeline` (step 4).

**`Required file not found: .../demonstrations.mat`**
You asked `train_pipeline` to start at a stage whose input does not exist yet. The error
names the stage to run first.

**`Unrecognized function or variable`**
The path is not set. `cd Lunar-Lander; addpath(genpath(pwd))`.

**`Unknown scenario "descend"`**
The error lists the valid names. They are `'touchdown'`, `'approach'`, `'terminal'`,
`'orbit'`.

**No animation window appears**
The demos open figures, so they need an interactive MATLAB session — `matlab -batch` will
run them but display nothing. Add `struct('animate', false)` if you only want the numbers.

**`train_pipeline` produces a weak Phase 4**
Some spread is normal; candidate selection is a draw. If mean Phase 4 lands under ~60%,
re-run stages 3:4 rather than the whole pipeline — the demonstration dataset is not the
problem, the corpus and the draw are.

**Tests pass individually but fail in the suite**
Almost always an unseeded `rng`. `reward_ordering_test` had exactly this bug; the fix is a
`rng(...)` in the test, not in the code under test.

---

## Results

### Non-intrusiveness — 30 episodes per cell, 240 episodes, zero failures

| phase | ON land% | ON impact | ON worst | OFF land% | OFF impact | OFF worst |
|---|---|---|---|---|---|---|
| 1 | 100% | 0.24 | 0.26 | 100% | 0.26 | 0.35 |
| 2 | 100% | 0.21 | 0.21 | 100% | 0.30 | 0.38 |
| 3 | 100% | 0.19 | 0.20 | 100% | 0.18 | 0.20 |
| 4 | 100% | 0.46 | 0.50 | 100% | 0.54 | 0.60 |

Worst touchdown impact 0.60 m/s against a 1.0 m/s limit, including 60 complete powered
descents from orbit. The barrier costs a competent controller nothing and barely engages —
11 engagements across the 1,085-second powered descent in the step 6 sample, against 153 in
the faulty-pilot demo in step 3.

The classical pilot on the same envelope: **100% on all four phases at n=30**, mean impact
0.25 / 0.28 / 0.29 / 0.28 m/s.

### Necessity — fault injection on the classical pilot

`run_fault_injection_study` sweeps four fault models × five magnitudes × three phases, 25
episodes per cell, guardian on and off — 60 cells, 3,000 episodes, about five minutes.

**Excluding the control-delay fault** (the architecture's known boundary, below), across
the 36 perception and actuator cells with a non-zero fault:

| | guardian OFF | guardian ON |
|---|---|---|
| mean landing rate | 35.4% | **99.4%** |
| worst touchdown impact | 87.07 m/s | **0.76 m/s** |
| cells within the 1.0 m/s limit | — | **36 of 36** |

Phase 3 in detail (the most demanding phase swept):

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

**The six cells it does not bound are all control delay ≥ 20 steps.** That limit is in the
table deliberately: the barrier reacts to true state, but it cannot act earlier than the
actuator responds. Beyond roughly 2 seconds of latency there is no recovery left to enforce.

### Negative results worth keeping

- **Zero-mean sensor noise does not discriminate.** Gaussian action noise up to σ = 0.5
  left landing rates at ~100% *with and without* the guardian. A PD loop re-deciding at
  10 Hz rejects zero-mean disturbance by construction. Every fault model in
  [core/apply_sensor_fault.m](Lunar-Lander/core/apply_sensor_fault.m) is therefore
  **systematic** — something the controller can neither see nor correct.
- **Control delay beyond ~20 steps is unrecoverable**, as above.
- **Severe velocity under-read used to be unrecoverable, and no longer is.** Before the
  regime-aware barrier rework a 0.7 under-read went 0% → 0%; it now goes 0% → 96%. The
  barrier and the guidance law both changed between those measurements, so the credit is
  not cleanly attributable; the largest candidate is that the barrier was computing
  available deceleration as `T_max·cos(θ)`, which goes negative past 90°.
- **From-scratch RL never landed.** Four architectures, ~25,000 episodes, three distinct
  and separately diagnosed failure modes. `run_algorithm_trade` reproduces it.

---

## The flight envelope

Four curriculum phases, addressed by name everywhere:

| Name | Start | Agent steps | What it is |
|---|---|---|---|
| `'touchdown'` | 50 m, 2 m/s | ~450 | final touchdown only |
| `'approach'` | 500 m, 10 m/s | ~1,200 | glide-slope approach |
| `'terminal'` | 2.5 km, 25 m/s | ~2,500 | full terminal descent |
| `'orbit'` | 15.2 km, 1697 m/s, 550 km downrange | ~8,900 | Apollo powered descent from PDI |

---

## Why the pipeline has this shape

Each stage exists because the simpler thing was tried and measured to fail. If you are
extending this, these are the walls to avoid walking into again.

**Why not just train an agent?** Four architectures (DDPG, TD3, SAC, PPO) at 1200 episodes
each, plus two longer runs — roughly 25,000 episodes — produced **zero landings** under
greedy evaluation. The reward landscape was verified on five independent properties, and
the task is demonstrably solvable: the classical controller lands 100% of all four phases
through the same `[-1,1]` action interface the agents were given. The gap is
**exploration**. A landing requires a coordinated descent, lateral null and square-up, and
undirected action sequences never produce one.

**Why DAgger and not more demonstrations?** Plain cloning plateaued at P1 100% / P2 88% /
P3 88% / **P4 25%**. The failure scales with *horizon*, not difficulty: a Phase 4 descent
is ~8,900 agent steps against Phase 1's ~450, so accumulated action error has twenty times
the exposure before touchdown. More expert data does not help — it all lies on the expert's
trajectory, and the clone's problem is *everywhere else*. Nor does capacity: 512-unit
networks lowered validation RMSE from 0.131 to 0.121 while the best landing rate **fell**
from 72% to 66%.

**Why β-mixing inside DAgger?** Pure clone rollouts from round one moved Phase 4 not at all
across two rounds and 137,000 corrective transitions. Over an 8,900-step descent the clone
drifts somewhere genuinely unrecoverable, and the expert's label at such a state teaches
nothing — no action recovers a vehicle 200 km downrange with the wrong energy. Mixing keeps
early rounds near the expert's distribution, where recovery is still possible.

**Why select on landing rate and not validation loss?** Across candidates the correlation
between validation RMSE and landing rate was **−0.021**. An epoch sweep had RMSE falling
monotonically 0.184 → 0.107 while landing rate bounced 43 / 10 / 57 / 47 / 3 / 20 percent.
Regression error selects a policy that hovers.

**Why screen-then-verify?** Taking the maximum of 8 noisy n=8 screens is the winner's curse,
and it bit: a clone that screened at 88% on Phase 4 scored **33%** at n=30. Stage 4 screens
wide and cheap, then re-measures a shortlist on fresh seeds.

**Reproducing the dead ends.** Two entry points exist only to re-run the paths that failed,
because a negative result nobody can reproduce is an anecdote. `run_algorithm_trade` trains
all four architectures from scratch; `train_rl_agent('td3')` does one. `train_from_demonstrations`
runs the *other* imitation route — `trainFromData` with a behaviour-cloning regulariser,
which reached 0% landings and 70% timeouts because the actor loss is dominated by an
uninformative critic early in offline training. That measurement is why `train_pipeline`
uses direct regression instead, and the script is kept so the claim can be checked.

**Why the braking-to-terminal handoff is a smooth blend and not a switch.** A discontinuous
handoff cannot be fitted by regression; the clone inherits a step it has no way to
represent. This is a constraint the *learning* half imposes on the *classical* half, and it
is easy to miss.

---

## Repository map

```
Lunar-Lander/
  core/                          the plant and the classical stack
    get_sim_params.m             SINGLE SOURCE OF TRUTH for every physical constant
    lunar_lander_dynamics.m      in-plane physics, curvilinear (flat-Moon) frame
    scripted_pilot.m             terminal-phase guidance law (the expert)
    braking_guidance.m           Apollo P63 braking; blends smoothly into scripted_pilot
    safety_sidecar_filter.m      THE BARRIER. Regime-aware, reads true state only.
    action_to_command.m          the [-1,1] <-> thrust interface, defined once
    command_to_action.m          its inverse
    apply_sensor_fault.m         systematic fault models
    get_ai_observation.m         signed-log observation normalisation
    phase_from_name.m            scenario names -> curriculum indices
    select_phase.m               restrict an environment to one phase, by name or index
    animate_lunar_lander.m       the visualiser
    main_simulation.m            classical test bench

  RL-training-harness/
    LunarLanderEnv.m             the rl environment (classdef)
    generate_demonstrations.m    pipeline stage 1
    pretrain_actor_supervised.m  pipeline stages 2 and 4
    dagger_refine.m              pipeline stage 3
    phase_landing_rates.m        THE measurement everything is selected on
    run_fault_injection_study.m  the necessity experiment
    run_algorithm_trade.m        the four-architecture negative result
    train_rl_agent.m             train one architecture from scratch
    train_from_demonstrations.m  reproduces the offline-RL negative result
    agent_architectures/         DDPG / TD3 / SAC / PPO builders

  flight-code/                   the barrier as standalone portable C
    generate_flight_code.m       MATLAB Coder driver
    export_flight_fixtures.m     records the episode the equivalence check uses
    src/                         the generated C, committed on purpose
    harness/                     make check | make demo | make bench
    fixtures/                    telemetry and the MATLAB reference output

  CI-tests/                      99 tests across 19 files
  train_pipeline.m               rebuild the agent from nothing
  evaluate_final_agent.m         the headline table
  run_trained_agent.m            watch one episode
  demo_sidecar_rescue.m          barrier vs faulty CLASSICAL pilot
  demo_agent_rescue.m            barrier vs faulty NEURAL agent
  demo_reel.m                    all of it, in sequence
```

**The barrier itself is [core/safety_sidecar_filter.m](Lunar-Lander/core/safety_sidecar_filter.m).**
A few hundred lines of deterministic arithmetic, no learned components, no state. That is
the point: it is small enough to argue about directly, which is what makes the wrapped
system trustable when the thing inside it is not.

---

## Continuous integration

[.github/workflows/matlab-tests.yaml](.github/workflows/matlab-tests.yaml) runs the full
suite on every push and pull request. Failures are republished as GitHub `::error::`
annotations by [.github/scripts/run_ci_suite.m](.github/scripts/run_ci_suite.m), because
job logs need admin rights to read while annotations are public — a red run should be
diagnosable from a fork.

---

## Licence

See [LICENSE](LICENSE).
