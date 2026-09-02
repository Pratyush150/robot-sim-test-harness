# Testing robot behaviour in simulation

This document is the reasoning behind the code in this repository: why
simulation testing pays for itself, what actually transfers to hardware and
what does not, how to choose between Gazebo, AirSim and Isaac Sim without
marketing input, how to keep a simulated test reproducible, and how to run the
whole thing in CI.

---

## 1. Why bother

The usual objection is fair: *the simulator is not the robot, so what does a
green simulation prove?*

It proves less than a flight test and more than nothing, and the gap between
those two is where most of your engineering time goes. Three concrete returns.

**It compresses the loop from hours to seconds.** A real outdoor test is a
site, a battery cycle, a safety pilot, weather, and a two-hour round trip. The
scenario suite in this repo runs 9 scenarios in well under a second on one
core. You will run a test you can run in a second. You will not run one that
costs an afternoon, and the test you do not run does not protect you.

**It makes the rare case ordinary.** The failure you care about is the GPS
dropout during the turn, the crosswind gust at the worst phase of the
approach, the moving obstacle that arrives exactly when you do. In the field
those happen once a month and you have no logs of them. In simulation you make
them happen on demand, every commit, at exactly the same phase of the
trajectory, with the same seed. `scenarios/sensor_dropout.yaml` in this repo is
a 40-line file. Reproducing it outdoors is a research project.

**It turns "it seems fine" into a number.** Nobody can eyeball a 3% path-length
regression or a settling time that crept from 5.2 s to 6.1 s. A harness that
records those and compares them to a committed baseline catches the slow decay
that no single test flight ever reveals.

What simulation does **not** do is tell you the robot works. It tells you the
robot's *logic* still does what it did last week. That distinction is the whole
of section 2.

---

## 2. The sim-to-real gap: what transfers and what does not

Be blunt about this with yourself and with whoever is paying for the work.

### Transfers well

- **Mission and state-machine logic.** Sequencing, mode transitions, abort
  paths, geofence decisions, waypoint retirement, timeout handling. This is
  discrete logic operating on numbers; the numbers being simulated changes
  almost nothing about whether the logic is right.
- **Coordinate frames and conventions.** ENU vs NED, FLU vs FRD, quaternion
  order, degrees vs radians, altitude datums. A frame bug is a frame bug in any
  environment, and simulation is the cheapest place to find it.
- **Interface and protocol behaviour.** Message rates, QoS mismatches, stale
  data handling, what your node does when a topic goes quiet.
- **Gross geometry and planning.** Does the planner produce a path around the
  obstacle at all. Does it deadlock in a corridor. Does the avoidance term have
  the sign you think it has.
- **Relative comparisons.** "Gain set B has 20% less overshoot than gain set A"
  usually survives the transfer even when the absolute numbers do not. This is
  the single most useful thing simulation gives you, and it is exactly what a
  baseline-diffing harness is built to exploit.

### Transfers badly, or not at all

- **Absolute timing.** Real sensor latency, driver buffering, USB enumeration
  order, scheduler jitter, an SD card that stalls for 200 ms. Simulators are
  optimistic about all of it.
- **Actuator reality.** Motor thermal derating, ESC startup transients, prop
  wash, wheel slip on a dusty floor, backlash, a servo that browns out under
  load.
- **Sensor physics.** Rolling shutter, motion blur, lens flare, LiDAR returns
  off wet asphalt or glass, multipath GNSS between buildings, magnetometer
  disturbance from your own power wiring. Simulated noise is Gaussian; real
  noise is structured, correlated and occasionally malicious.
- **Perception performance.** A detector's simulated mAP tells you almost
  nothing about its field mAP unless you have done serious domain
  randomisation. Do not quote one for the other.
- **Contact dynamics.** Landing gear compliance, bounce, ground effect,
  friction cones. Contact-rich simulation is the least trustworthy part of any
  physics engine and the least reproducible across GPU drivers.
- **Compute budget.** Everything is fast on a workstation. Nothing is fast on a
  Jetson at 45 °C with the camera pipeline already eating the memory bandwidth.

### The honest framing

Simulation testing is a **regression net, not a certificate**. Its job is:
"nothing you changed this week broke something that used to work." It is not:
"this will fly." Anyone who tells you the second is selling something.

The corollary is that your scenarios should test the things that transfer.
Assert on mission sequencing, clearance, geofence containment, path shape,
settling behaviour and energy budget. Do not build a scenario whose verdict
depends on the fourth decimal place of a contact force.

---

## 3. Choosing a simulator

The three that come up, with the tradeoffs stated plainly.

### Gazebo (Sim, i.e. Garden / Harmonic; formerly Ignition)

**Use it when** you are in the ROS 2 ecosystem, you care about sensor plugins
and TF, and you want the suite to run on a CPU-only build agent.

| | |
|---|---|
| Fidelity | Good rigid-body dynamics (DART/Bullet). Sensor models are functional rather than photoreal. |
| GPU cost | None required for physics-only headless runs. Cameras want a GPU but degrade gracefully. |
| Sensor support | Broad and mature: LiDAR, depth, IMU, GPS, contact, magnetometer, air pressure. |
| Maturity | The most mature of the three for robotics. Also the most churned: Gazebo Classic to Ignition to Gazebo Sim, with renamed packages at every step. |
| Licensing | Apache 2.0. No account, no EULA, no download portal. |
| The catch | Version and package-name fragmentation. A tutorial from 2021 will not run. Determinism requires you to start the world **paused** and drive it with `multi_step` (see `src/simharness/simulators/gazebo.py`). |

### AirSim (and its maintained fork, Colosseum)

**Use it when** you want visually rich outdoor aerial scenes, camera-driven
perception, and a genuinely convenient Python API.

| | |
|---|---|
| Fidelity | Unreal Engine visuals are excellent. Multirotor dynamics are decent; the fixed-wing model is thin. |
| GPU cost | Real. Unreal wants a discrete GPU. Headless helps but does not eliminate it. |
| Sensor support | Cameras (RGB, depth, segmentation) are the strong suit. LiDAR exists and is adequate. |
| Maturity | Upstream Microsoft AirSim is archived. Colosseum is the community continuation. The API surface is unchanged, which is why the adapter here works against either. |
| Licensing | MIT, on top of Unreal's EULA. |
| The catch | The pip package is unmaintained and pinned to old msgpack/numpy; expect to fight the install. `simPause` + `simContinueForTime` is the only honest way to step it — `ClockSpeed` is wall-clock rescaling, not determinism. |

### NVIDIA Isaac Sim

**Use it when** you need photoreal synthetic data, domain randomisation at
scale, or PhysX articulations, and you already have the GPUs.

| | |
|---|---|
| Fidelity | The highest of the three. RTX rendering, PhysX 5, real material models. |
| GPU cost | Highest. An RTX-class GPU is a hard requirement, and a CI runner with one is a real line item. |
| Sensor support | Excellent, including physically-based camera and LiDAR models with ray tracing. |
| Maturity | Improving quickly, and changing quickly with it. API churn between releases is significant, and it is a large download, not a package. |
| Licensing | Proprietary NVIDIA licence, free for individual use, with terms for commercial and cloud deployment. Read them before you plan a fleet of CI runners. |
| The catch | Your process *is* the simulator. `SimulationApp` must be constructed before any `omni.*` import, and the whole thing only runs under Isaac's own `python.sh`. That constraint alone is a strong argument for an abstraction layer. |

### A pragmatic recommendation

Most teams do not need to pick one. They need:

1. A **fast, headless, deterministic** default for the regression suite that
   runs on every commit. That is what `MockSimulator` is in this repo, and what
   a physics-only headless Gazebo world is in a bigger one.
2. A **high-fidelity** environment for the small number of questions the fast
   one cannot answer — perception, contact, visual realism — run nightly or on
   demand, on hardware that has a GPU.

The whole reason `src/simharness/simulators/` exists is so those are the same
scenarios, the same assertions and the same reports.

---

## 4. Keeping scenarios deterministic

Non-determinism destroys the value of a regression suite. A test that fails one
run in twenty gets muted within a fortnight, and then it is not a test.

**Seed everything from one place.** Every random draw here comes from
`scenario.sim.seed`, through named sub-streams
(`random.Random(f"simharness:sensor:{seed}")`,
`random.Random(f"simharness:turb:{seed}")`). No module-level `random` calls, no
seeding from the clock. Separate streams matter: if sensor noise and turbulence
share one generator, adding a turbulence step shifts every subsequent noise
sample and the whole trace moves.

**Step by simulated time, never wall-clock time.** `dt` comes from the
scenario. The loop bound is `time_limit / dt`, computed up front. A slow build
agent must produce the same trace as a fast laptop. If you find yourself
writing `time.sleep`, you have lost determinism.

**Make obstacle motion a closed-form function of `t`.** `Obstacle.position_at(t)`
is analytic, not integrated. That means halving `dt` moves the obstacles to
*the same places at the same times*, instead of accumulating a different
integration error.

**Use an unconditionally stable integrator for the actuator model.** The
actuator lag here uses the exact discrete solution `alpha = 1 - exp(-dt/tau)`
rather than explicit Euler's `dt/tau`. With Euler, a `dt` anywhere near `tau`
makes the model ring or diverge, so changing your step rate silently changes
your physics.

**Quantise recorded floats.** Every value in a trace is rounded to 9 decimal
places on record. That is what lets the determinism test compare *JSON strings*
rather than doing an epsilon dance, and it is what makes the trace diff report
a real divergence instead of last-bit noise.

**Assert on it.** `tests/test_determinism.py` runs every shipped scenario twice
and requires byte-identical trace JSON, checks that a different seed *does*
produce a different trace, and checks that a 4-worker parallel run matches a
serial one exactly. Without that test the property is an intention, not a
guarantee.

**With a native backend, expect to work harder.** DART and PhysX are
deterministic for a fixed step size, a fixed scene and a fixed build. They are
not deterministic across engine versions, plugin sets, or (for GPU physics) GPU
drivers. Practical consequences:

- Pin the simulator version in your CI image, and treat a bump as a deliberate
  baseline refresh, reviewed like any other change.
- Give every sensor plugin an explicit `<seed>`, or it will seed from the clock.
- Make `dt` an exact multiple of the engine's physics step. Both the Gazebo and
  Isaac adapters here refuse a mismatched `dt` rather than let the engine
  quietly quantise it and drift your timestamps.
- For contact-rich scenarios, compare with a tolerance (`diff --tolerance`)
  instead of demanding exactness. Choose the tolerance from measured run-to-run
  spread, not from optimism.

---

## 5. Folding it into CI

The pipeline in `.github/workflows/scenarios.yml` is three jobs.

**Unit tests** on a matrix of Python versions. Pure logic, no simulator, no
network. Fast enough that there is no excuse to skip it.

**The scenario suite**, run headless with the mock backend, emitting four
artifacts:

```bash
simharness suite scenarios/ --workers 4 \
  --junit reports/junit.xml \
  --html  reports/report.html \
  --json  reports/results.json \
  --trace-dir reports/traces
```

JUnit XML gives you per-assertion results in the CI UI. The HTML report is a
single self-contained file with inline SVG trajectory and time-series plots,
which is what you actually open when something goes red. The traces are the raw
evidence, and `simharness replay` re-evaluates assertions against them without
re-simulating.

**The baseline comparison**, which is the part that catches slow decay:

```bash
simharness ci scenarios/ --baseline ci/baseline.json --tolerances ci/tolerances.yaml
```

Rules, chosen deliberately:

- A scenario that was green and is now red **fails the build**.
- A scenario in the baseline that is missing from the run **fails the build**.
  Deleting a failing test is not a fix.
- A scenario marked `expect_failure: true` that *passes* fails the build. A
  negative test that has gone green is a broken negative test.
- Metric drift beyond tolerance is **reported but advisory** by default. A 12%
  longer path deserves a human look, not a 2 a.m. merge block. Use
  `--strict-metrics` where you want it enforced.

**Refresh the baseline deliberately.** `simharness ci --update-baseline`
rewrites `ci/baseline.json`; commit it in its own change with an explanation.
A baseline that gets refreshed automatically on failure is not a baseline.

**Keep expensive work off the critical path.** Monte Carlo sweeps and boundary
bisection belong on a nightly or manual trigger, not on every push. The
`envelope-sweep` job in the workflow is wired to `workflow_dispatch` for
exactly that reason.

**A note on native backends in CI.** Gazebo headless on a standard GitHub
runner is feasible for physics-only worlds and slow for anything with cameras.
AirSim and Isaac Sim need a GPU runner, which means self-hosted. The realistic
split is: fast backend on every commit, native backend nightly on your own
hardware, same scenarios in both. That split only works if the harness is
backend-agnostic, which is why `Simulator` has six methods and no leaked
Gazebo or Unreal types.

---

## 6. Writing scenarios that are worth having

A few habits that separate a suite people trust from one they mute.

**Assert on behaviour, not on trajectories.** "Minimum clearance stayed above
0.10 m" survives a controller retune. "The robot was at (3.42, 0.18) at t=4.2 s"
does not, and will fail on the first legitimate improvement.

**Give every assertion a name.** `name: keeps_its_distance` is what appears in
the JUnit output and in the failure message. `min_clearance` is not.

**Prefer temporal assertions for anything with a "then" in it.** "It slowed
down eventually" is `eventually`. "It never got closer than X *before* it
arrived" is `until`. "It settled and stayed settled" is `eventually_always`,
which catches the case a first-crossing check misses: entering the tolerance
band and then leaving it again.

**Own at least one scenario that is supposed to fail.** This repo ships
`expected_failure_no_avoidance.yaml`. Its job is to prove that collision
detection, the clearance assertion and the `always` operator still *fire*. If a
refactor makes them silently pass everything, every other scenario stays green
and only this one flips — to `XPASS`, which fails the build.

**Bound the envelope, do not just sample it.** One passing run is an anecdote.
`simharness sweep --boundary` bisects the parameter and gives you a number:
this controller holds up to a certain crosswind and fails past it. That is the
number that belongs in a design review.
