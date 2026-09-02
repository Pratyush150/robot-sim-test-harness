# Assertion reference

Every assertion returns the same shape: **pass/fail**, the **worst-case value**,
the **timestamp and step index** it happened at, the threshold it was compared
against, and a sentence you can paste into a bug report.

Assertions live in a scenario's `assertions:` list. `name:` is optional but
strongly recommended — it is what appears in the JUnit output.

```yaml
assertions:
  - {type: min_clearance, name: keeps_its_distance, threshold: 0.10}
```

---

## Scalar assertions

| `type` | Parameters | Passes when | Reports as worst value |
|---|---|---|---|
| `reached_goal` | `tolerance` (default: goal's), `within` (s), `after` (s) | The robot came within `tolerance` of the goal, no later than `within` | Arrival time, or closest approach if it never arrived |
| `no_collision` | – | No obstacle contact at any step | Minimum clearance (negative = penetration depth) |
| `min_clearance` | `threshold` (m, required), `after` (s) | Hull-to-obstacle gap never dropped below `threshold` | The minimum gap, and which obstacle |
| `geofence` | `polygon` or `min_xy`/`max_xy`, `min_z`, `max_z` | The robot stayed inside the fence | Signed margin to the boundary (negative = outside) |
| `max_velocity` | `limit` (m/s, required), `after` (s) | Ground speed never exceeded `limit` | Peak speed |
| `max_acceleration` | `limit` (m/s², required), `after` (s) | Acceleration magnitude never exceeded `limit` | Peak acceleration |
| `max_jerk` | `limit` (m/s³, required), `after` (s) | Jerk magnitude never exceeded `limit` | Peak jerk |
| `heading_error` | `limit` (rad, required), `after` (s) | Heading error vs the bearing-to-goal stayed under `limit` | Peak heading error |
| `path_length` | `max_ratio` (default 1.25), `optimal` (m) | Distance travelled ≤ `max_ratio` × optimal | The ratio |
| `settling_time` | `signal`, `target`, `band`, `limit` (s, required) | Signal entered the band around `target` **and stayed there**, by `limit` | Stabilisation time |
| `overshoot` | `signal`, `target`, `max_percent` (required) | Peak excursion past `target` ≤ `max_percent` of the initial step | Overshoot percentage |
| `no_oscillation` | `signal`, `max_frequency` (Hz, required), `min_amplitude`, `after` | No sustained oscillation above `max_frequency` | Estimated dominant frequency |
| `energy_budget` | `max_wh` or `max_j` (one required) | Total energy stayed within budget | Energy used |

Notes worth knowing:

- **`settling_time` is not "first time within tolerance".** It is
  *eventually always* inside the band, so a signal that enters the band and
  then leaves it fails. That is the case a first-crossing check misses.
- **`overshoot` is measured relative to the initial step size**, so a signal
  that starts at the target reports 0% rather than dividing by zero.
- **`no_oscillation` linearly detrends the signal first**, so a monotone ramp
  is not mistaken for an oscillation, and ignores residuals smaller than
  `min_amplitude` so sensor noise is not mistaken for hunting.
- **Derivatives use backward differences on the recorded trace**, and index 0
  is 0.0 rather than an extrapolation. Otherwise a jerk assertion fails on the
  spawn step every time.

---

## Temporal assertions

| `type` | Parameters | Meaning |
|---|---|---|
| `always` | `predicate` | The predicate holds at every step |
| `never` | `predicate` | The predicate holds at no step (i.e. `always not P`) |
| `eventually` | `predicate`, `within` (s) | The predicate holds at some step, optionally by a deadline |
| `eventually_always` | `predicate`, `settle_by` (s) | The predicate becomes true and never lapses again |
| `until` | `condition`, `release`, `within` (s) | `release` eventually holds, and `condition` holds at every step before it |

### Finite-trace semantics, stated plainly

A recorded run is finite, so these operators need a convention. Ours:

- `eventually P` with no deadline is satisfied by any step in the trace.
- `eventually_always P` is satisfied when there is a step after which P never
  becomes false again. On a finite trace that reduces to "P holds at the last
  step", which is why the result also reports the **stabilisation time** — the
  first step of that final unbroken run — and accepts `settle_by` to bound it.
  Without `settle_by` you are only testing the final step.
- `until` is **strong** until: the release condition must actually occur.
  `A until B` where B never happens is a failure, not a vacuous pass.
- Vacuous truth is reported rather than hidden: `always P` on an empty trace
  passes and says so in the message.

### Predicates

```yaml
# atom
{signal: clearance, op: ">=", value: 0.2}

# boolean combinations
{all: [{signal: z, op: ">=", value: 4.0}, {signal: z, op: "<=", value: 6.0}]}
{any: [...]}
{not: {...}}
```

Operators: `<` `<=` `>` `>=` `==` `!=`.

### Signals

Available to every predicate and to the `signal:` parameter of
`settling_time`, `overshoot` and `no_oscillation`:

| Signal | Units | Notes |
|---|---|---|
| `t` | s | |
| `x`, `y`, `z` | m | Ground truth |
| `yaw`, `yaw_rate` | rad, rad/s | |
| `vx`, `vy`, `vz`, `speed` | m/s | |
| `accel`, `jerk` | m/s², m/s³ | Backward differences |
| `cmd_speed` | m/s | Magnitude of the commanded setpoint |
| `distance_to_goal` | m | 3D |
| `clearance` | m | To the nearest obstacle hull; `inf` when there are none |
| `heading_error` | rad | Absolute, relative to the bearing to the goal |
| `geofence_margin` | m | Signed, from the world bounds; positive inside |
| `energy_j`, `energy_wh` | J, Wh | Cumulative |
| `estimate_error` | m | Distance between ground truth and the estimate the controller saw |
| `sensor_valid` | 0/1 | 0 during a dropout |
| `collided` | 0/1 | Latches once set |

`estimate_error` is the one people forget exists. It is what turns a sensor
dropout scenario from a formality into a real test: the controller only ever
sees the estimate, and this signal is how far that estimate has drifted from
the truth.

---

## Adding your own

The registry is open. See `examples/02_custom_assertion.py`:

```python
from simharness.assertions import AssertionResult, register_assertion

@register_assertion("max_cross_track_error")
def max_cross_track_error(spec, ctx):
    limit = float(spec.params["limit"])
    ...
    return AssertionResult(name=spec.label, type=spec.type, passed=worst <= limit, ...)
```

It is then usable from YAML as `{type: max_cross_track_error, limit: 0.5}`,
with the same reporting, JUnit output and HTML rendering as the built-ins.
