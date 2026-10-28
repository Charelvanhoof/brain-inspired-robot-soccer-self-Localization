# Brain Inspired Localization POC

This project is a proof of concept for robot self-localization using active
inference. It simulates a robot on a RoboCup-style field, compares the actual
robot state with the belief state, and lets you test landmark-only,
motion-only, and fused localization.

The simulator is intentionally basic. It is mainly used to feed observations
into the localization algorithm and to generate reports, not to be a
high-fidelity robot physics model.

## Project Files

- `brain_inspired_localization_simulator.py`: interactive POC simulator, experiments,
  convergence tests and reports; calls the shared core (POC V29).
- `brain_inspired_localization_core.py`: standalone localization core module
  exposing `step_core(request)` for integration without the GUI.
- `brain_inspired_localization_core.py`: contains the editable list of complete landmark profiles.
- `step_core_cpp_interface.txt`: C++ / pybind11 request and response notes.
- `Robot_Soccer_Self_Localization_with_Active_Inference.pdf`: latest paper.
- `reports/`: generated datasets and reports.

## Install

Use Python 3.10 or newer.

Install the required packages:

```powershell
python -m pip install pygame matplotlib
```

Required dependencies:

- `pygame` for the interactive GUI.
- `matplotlib` for plots and PDF reports.

## Run

Start the GUI with:

```powershell
python ".\brain_inspired_localization_simulator.py"
```

## GUI Basics

- Drag the green robot to move the actual pose.
- Drag the red robot to move the belief pose.
- Drag the ball to change where the actual robot looks.
- Hold `Q` or `E` to rotate the selected robot.
- Use `RESET` to copy the green actual robot state into the red belief robot
  state and clear the last experiment history.
- Use the algorithm button to switch between:
  - `landmark only`
  - `motion only`
  - `motion+landmark`
- Select recognition names through `LM PROFILE`: unique names match directly;
  repeated names use gated nearest matching. There is no association button.
- Use `PROFILE` to cycle masterdata settings:
  - `40Hz`, with `dt = 0.025 s`
  - `200Hz`, with `dt = 0.005 s`
- Use `DIST PREC` to select the distance precision modifier (YES/NO):
  - `YES`: reduce landmark precision as the predicted landmark distance grows.
  - `NO`: apply no distance weighting to landmark precision.
  Switching this option reapplies the selected frequency's preset, including
  its learning rates, replacing any manual adjustments to those settings.
- Use `LM PROFILE` to select a complete setup from core, e.g.
  `9 unique`, `9 ambiguous`, `2 ambiguous` or `9 unique +1 orientation far`.
  The list includes 1/2/3/5/9-point setups and 9+1 near/far setups for both modes.
  Near/far orientation cues are at (1200, 300) and (4900, 300) FCS cm.
  Names, types and coordinates are supplied directly by the selected profile;
  normal tracking still ignores orientation landmarks.
- Use `STEP 1`, `STEP 100`, or `STEP 2000` to run an inference experiment.
- For motion algorithms, the selected demo path plays for the chosen number of
  steps and the data is stored for the report.
- Use `PLAY DEMO` to play the selected demo path continuously.
- Use the demo-path button to cycle through available paths.
- Use `PLOT LAST EXPERIMENT` to create a detailed multipage PDF report and a
  publication-oriented PDF in the `reports` folder.
- Use `P-2hrs` to export a two-hour simulated tracking dataset for the selected
  profile/path, with a nearby initial belief. It writes CSV and JSON metadata
  under `reports`: 288,000 steps at 40Hz or 1,440,000 steps at 200Hz.
- Use `P-LM` to compare landmark profiles in repeated trials (see below).
- Use `1 STEP ANALYSIS` to show per-step diagnostic information in the side
  panel.
- Sensor/simulation toggles:
  - `NOISE` adds landmark and motion observation noise.
  - `FOV` limits simulated landmark observations to landmarks inside the actual
    robot camera cone.
  - `SLIP` simulates short odometry slip events by scaling motion velocity
    readings.
  - `DROP` randomly removes 10% of landmark observations.
  - `OBSTR` hides affected landmarks for about `2.0 +/- 0.5` seconds.
  - `BIAS` adds fixed odometry drift: `+18 cm/s` forward, `+8 cm/s` left,
    `+1.5 deg/s` rotation.

## Landmark Profile Experiment (P-LM)

Set the actual and belief starting poses, ball position, demo path, algorithm,
and sensor/inference settings in the GUI. Enter a numeric trial count N in
`CONV CASES`, then click `P-LM`.

The experiment runs N trials of 30 simulated seconds for each landmark profile
`1`, `2`, `3`, `5`, and `9` (5 x N trials in total). Each trial uses the saved
starting poses and ball position, starts with zero motion, and replays the same
selected path. The batch uses the captured settings and varies the landmark
profile rather than using only the currently selected `LM PROFILE`.

Outputs are saved under `reports`:

- CSV actual/belief poses, tracking errors, visible landmark counts, and free
  energy, plus JSON settings/completion metadata for all five profiles.
- A report PDF and a publication-oriented PDF comparing profiles `3`, `5`, and
  `9` only. Curves show mean absolute X, Y, and heading errors over time, with
  P1-P99 percentile shading when more than one completed trial is available.

Press `Esc` to stop the batch. Collected data is retained; the comparison plots
use completed trials only.

## Headless Dataset Export

Run a tracking dataset export without opening the GUI, for example:

```powershell
python ".\brain_inspired_localization_simulator.py" --export-paper-data --paper-data-profile 40Hz --paper-data-duration-s 7200 --paper-data-path "Circle" --paper-data-start nearby
```

Distance precision is enabled by default. Add
`--no-paper-data-distance-precision-modifier` to disable it, or use
`--paper-data-distance-precision-modifier` to enable it explicitly.
`--paper-data-steps` overrides the requested duration. Use `--help` to see all
export options. This command exports the tracking dataset; `P-LM` is a separate
GUI experiment.

## Convergence Tests

Convergence tests use the same condition throughout:

- `|dx| <= 3 cm`
- `|dy| <= 3 cm`
- `|dtheta| <= 3 deg`
- at least `5` consecutive hits

The `CONV CASES` input controls the number of cases. Enter a number, or enter
`reg` to use the fixed regression start set.

Static convergence buttons:

- `S-CONV`: landmark-only static convergence with random actual and belief
  starts.
- `S-NEAR`: landmark-only static convergence where belief starts near the
  actual pose (`+/-50 cm`, `+/-20 deg`).
- `S-CORN`: landmark-only static convergence with the actual pose outside a
  corner.

Dynamic convergence buttons:

- `D-CONV`: convergence while the actual robot follows the selected demo path
  from random starts.
- `D-NEAR`: convergence while the actual robot follows the selected demo path
  from nearby belief starts.

Convergence tests write a PDF report to the `reports` folder. Reports
include converged/non-converged counts plus steps-to-converge statistics:
mean, standard deviation, p95, p99, maximum, and the number of converged cases.
Failed cases include compact actual/belief path traces, visible landmark counts,
and start/end poses for review.

## Landmark profiles: one list in core

`LANDMARK_SCENARIO_PROFILES` in core is a list of self-contained profiles. Each profile holds its
name and the complete list of landmark records. There is no
master landmark table, membership list, legacy-ID table or inferred renaming.
It is intentional that a landmark can be repeated in several profiles.

For example, this entire profile can be edited in one place:

```json
{
  "name": "2 ambiguous",
  "landmarks": [
    {"name": "goal", "type": "pose", "x_cm": 900, "y_cm": 0},
    {"name": "goal", "type": "pose", "x_cm": -900, "y_cm": 0}
  ]
}
```

`9 unique` explicitly gives its nine landmarks individual recognition names
such as `corner_TR`; `9 ambiguous` explicitly repeats names such as `corner`.
Nothing in Python renames landmarks to switch between these scenarios.
A unique name matches directly; repeated names use nearest association and the
existing gate. Uniqueness comes from name counts in the selected complete map,
never from how many landmarks happen to be in the current FOV.

Landmark records contain `name`, `type`, `x_cm`, `y_cm`, with absolute FCS
coordinates in centimeters. Type is `pose` or `orientation`; normal tracking
uses only pose landmarks. Raw orientation measurements may remain in
reports, without contributing an update or free energy. No gPI is included yet.

The selected profile is used both for core calculations and to generate sensor
observations. Thus `2 ambiguous` gives the core exactly those two map points.
FOV, noise and other sensor toggles then act on observations independently.
The simulator uses row indices for ground-truth/report references; there are no
hand-maintained labels to align. To add a point or profile, edit the list in core
and restart the simulator. Profiles need not exist in pairs.

## Core profile selection and one-frame API

Landmark profiles are built into core as Python data. No JSON loading or landmark
initialization call is needed. The first profile is selected by default:

```python
import brain_inspired_localization_core as core

core.set_landmark_profile("9 unique")
response = core.step_core(request)
```

The GUI sets the currently selected profile in core and synchronizes the existing
core settings. Neither the profile nor settings are added to the step_core request.
Selecting a profile changes the map; matching follows its names. It does not reset
learning rates or retain robot belief. The caller supplies the current belief
and receives the next belief on every call.

Map-point diagnostic keys are string row indices (`"0"`, `"1"`, ...) in the
selected profile's landmark list, not sensor identities. Observations supply
recognition names. Settings remain the existing uppercase core variables.

`step_core` requires a UTC timestamp with exactly three fractional digits (`Z`
or `+00:00`) and a finite positive time step. Belief, landmark and motion values
must be finite. It shields inputs, delegates to inference and returns detached
JSON-friendly data. Existing diagnostics remain; `landmark_terms` additionally
reports actual associations, whether an observation was used, skip reasons,
precision, Jacobian, J-transpose/P/error terms, update contributions and per-point
free energy. `diagnostics.settings` records the core settings actually used.

There is one algorithm implementation: `brain_inspired_localization_core.py`.
It imports only standard-library modules and contains no GUI, sensor simulation,
or map-file reads. Its built-in profiles are shared with the simulator. The
simulator sets the selected profile and existing settings, and processes each
frame through `core.step_core` using `simulation_step_from_observations`.
The map is exactly the selected profile; FOV is a separate sensor setting.

Shared state/transform helpers are imported from core, without copied function
bodies. Map indices are local to each selected profile and are also used for
simulator ground truth and reports. One-step analysis makes one read-only step_core call and
presents the returned motion/landmark terms; it no longer recomputes inference
contributions. The prediction GC diagnostic keeps fused analysis aligned with
the actual post-motion landmark prediction. GUI layout and launch command are
unchanged; a further physical split of simulation/reports/UI files is optional.

Run `python -m unittest test_core_interface test_localization_regression test_convergence_symmetry -v`.

## Frozen regression baseline before the landmark refactor

Run the new numerical regression set and the existing convergence tests with:

```powershell
python -m unittest test_localization_regression test_convergence_symmetry -v
```

`localization_regression_baseline.json` records 16 deterministic scenarios from
`resolve-symmetry` commit `5402ded`, including the source hashes and explicit
40 Hz settings. It contains actual/belief generalized coordinates and saved
sensor observations, not just seeds. Stationary input frames use a repeat count;
moving cases store each observation frame separately. FOV is 110 degrees, with
noise, dropout, obstruction, slip and bias disabled.

Coverage includes Gated nearest and Unique ID, all seven landmark layouts and both recognition modes, landmark-only/motion-only/fused inference, stationary and
moving poses, a symmetric solution, no observations, one visible corner, and
an association rejected by the gate. Some cases deliberately do not converge
within their fixed 20/60/600-step budgets; the suite preserves these outcomes
rather than asserting that every scenario must succeed.

The tests replay saved inputs through the simulator adapter and the standalone
core (public stepping and low-level core unit tests). All cases, including perfect-recognition
Unique ID cases, also run through `step_core(request)`, checking that requests
are not mutated. Test adapters select the explicit unique/gated profile and
normalize historical input/output field names for comparison. The frozen measurements
and expected numeric results are not regenerated. Test-only adapters translate
old observation IDs/profile names and compare predictions only for points in
the selected profile; there is no legacy table in the application configuration.
The step-4 exclusion of orientation landmarks intentionally changes only the
two poster scenarios:
their tracking states and energies are now checked against the equivalent
already-frozen nine-landmark scenarios. The remaining 14 scenarios continue
to match their original numerical results, with prediction metadata projected
onto the selected profile's own map points. Original poster results remain in
the baseline for historical comparison, without silently overwriting them.

At steps 1, 5, 20, 100, 300 and the final step (when reached), compare generalized
coordinates, motion/landmark/total free energy, actual association decisions,
predicted observations, motion terms, and exported report rows. Numeric tolerance
is `rel_tol=1e-10`, `abs_tol=1e-9`; counts and labels must match exactly. A real
three-trial convergence report is also replayed against saved observations,
checking one correct, one symmetric and one failed case, step statistics and
failure details. This checks report data, not PDF layout or interactive rendering.
A separate perfect-recognition report checks two correct and one failed trial,
including the original 173/461-step convergence results and report statistics.

Ordinary test runs only read the baseline. To intentionally establish a new
reference after reviewing a behavior change, run:

```powershell
python test_localization_regression.py --capture-baseline
```

Do not regenerate it merely to make a failing regression pass. During the
planned landmark and file refactors, keep the current expected numeric results
and adapt only the input/output adapters when field names or interfaces change.
These are behavior-preservation tests, not an independent proof of the math.

## Reports

The last-experiment report contains:

- Position page: `x`, `y`, `theta`, and free energy.
- Velocity page: `x_dot`, `y_dot`, `theta_dot`.
- Acceleration page: `x_ddot`, `y_ddot`, `theta_ddot`.

Reports include the active algorithm, masterdata profile, demo path, precision
settings, distance precision modifier state, learning rates,
noise/slip/dropout/obstruction/FOV settings,
association mode, landmark association gate, and landmark scenario profile.

The optional masterdata profile comparison tooling remains in the code for
manual use, but it is no longer exposed as a primary GUI button.

## Notes

- The localization algorithm is implemented once, in the `CORE` block of
  `brain_inspired_localization_core.py`. Simulator/UI, reports and tests run
  on top of it through explicit initialization and per-frame stepping.
- The real robot / C++ integration boundary is `step_core(request)`. It
  processes one supplied observation frame at a time and returns the updated
  generalized-coordinate belief state plus diagnostics.
- Simulation landmark profiles affect simulated observation generation and
  reporting. The real robot `step_core(request)` path still accepts whatever
  landmark observations the caller supplies.
- Field coordinates use `(0, 0)` at the center of the field.
- `+X` points toward the enemy goal.
- `+Y` points to the robot's left side when facing the enemy goal.
- `theta = 0` points toward the enemy goal, with positive rotation to the left.
- Angles in the core algorithm are in radians.
