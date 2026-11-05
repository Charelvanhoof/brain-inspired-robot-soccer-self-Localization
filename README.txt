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
- `brain_inspired_localization_simulator.py`: contains the editable list of complete landmark profiles.
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
  Frequency selection sets the global timestep separately from learning-rate
  presets. Applying a preset does not reset PI_CAM, PI_ODO or PI_IMU.
  `LEARNING_RATE_PROFILES` contains a profile name and six rates per entry.
  `LEARNING_RATE_PROFILE_SELECTIONS` links frequency and distance modifier to
  the named profile, e.g. `40Hz - DPM enabled`. There is no separate GUI profile
  selector; the frequency and `DIST PREC` controls select the linked rates.
  The four default profiles use the archived paper's tuning. Separate
  `RobotSports test 40Hz DPM enabled` preserves the robot-test tuning. An entry in
  `LEARNING_RATE_PROFILE_OVERRIDES` selects `RobotSports test 40Hz DPM enabled`
  for `testRobotsports` at 40Hz with DPM enabled. Other combinations use the
  default mapping. Changing the landmark profile reapplies the matching tuning.
- Use `DIST PREC` to select the distance precision modifier (YES/NO):
  - `YES`: reduce landmark precision as the predicted landmark distance grows.
  - `NO`: apply no distance weighting to landmark precision.
  Switching this option reapplies the selected frequency's preset, including
  its learning rates, replacing any manual adjustments to those settings.
- Use `LM PROFILE` to select a complete setup from the simulator, e.g.
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

## Landmark profiles: one list in the simulator

`LANDMARK_SCENARIO_PROFILES` in the simulator is a list of self-contained profiles. Each profile holds its
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
hand-maintained labels to align. To add a point or profile, edit the list in the simulator
and restart the simulator. Profiles need not exist in pairs.

## Core profile selection and one-frame API

Core contains no profile catalog or default field map. The simulator owns its
profile catalog; a real robot wrapper loads the applicable field configuration.
Both supply the selected map through the same initialization boundary:

```python
import brain_inspired_localization_core as core

core.init_landmarks({
    "profile": "profile name",
    "landmarks": [
        {"name": "goal", "type": "pose", "x_cm": 900, "y_cm": 0},
        {"name": "goal", "type": "pose", "x_cm": -900, "y_cm": 0},
    ],
})
core.init_learning_rates({
    "profile": "profile name",
    "landmark_xy": 1.6, "landmark_theta": 1.700e-07,
    "motion_velocity_xy": 4.0, "motion_velocity_theta": 4.0,
    "motion_acceleration_xy": 40.0, "motion_acceleration_theta": 40.0,
})
response = core.step_core(request)
```

`profile` is a diagnostic label, not a catalog lookup key. `init_landmarks`
validates and stores its own copy of the map, returning profile and landmark_count.
Later caller edits have no effect until init_landmarks is explicitly called again.
Invalid initialization leaves the previous map intact. Landmark/fused inference
requires initialization; motion-only inference does not. An explicit empty map
is allowed. One active map is stored per imported core module.

A string `profile` name and all six learning rates must be initialized before inference; core has no default
rates. The values above are examples. Rates must be finite non-negative numbers;
zero disables a correction term while motion propagation remains active. Core
copies the supplied values; changes require another `init_learning_rates` call.
The learning-rate profile name is for reference only and does not affect inference.
Invalid input preserves the active rates. Map and belief remain unchanged.

The GUI initializes the selected map and learning rates at startup, and applies
changes before the next step. Neither the profile nor settings are
added to the step_core request.
Selecting a profile changes the map; matching follows its names. It does not reset
learning rates or retain robot belief. The caller supplies the current belief
and receives the next belief on every call.

Map-point diagnostic keys are string row indices (`"0"`, `"1"`, ...) in the
selected profile's landmark list, not sensor identities. Observations supply
recognition names. Learning-rate presets belong to the simulator or robot caller.
Sensor precisions and internal core settings retain their existing interfaces.

`step_core` requires a timestamp string (no format requirement) and a finite
positive time step. Belief, landmark and motion values
must be finite. It shields inputs, delegates to inference and returns detached
JSON-friendly data. Existing diagnostics remain; `landmark_terms` additionally
reports actual associations, whether an observation was used, skip reasons,
precision, Jacobian, J-transpose/P/error terms, update contributions and per-point
free energy. `diagnostics.settings.learning_rates` records the active six rates; the rest of
`diagnostics.settings` records current sensor precisions and internal settings.
`diagnostics.profile` records the active map label.

There is one algorithm implementation: `brain_inspired_localization_core.py`.
It imports only standard-library modules and contains no GUI, sensor simulation,
or map-file reads. The simulator supplies its selected profile through
`init_landmarks` and `init_learning_rates`, and processes each frame
through `core.step_core` using `simulation_step_from_observations`.
The map is exactly the selected profile; FOV is a separate sensor setting.

Shared state/transform helpers are imported from core, without copied function
bodies. Map indices are local to each selected profile and are also used for
simulator ground truth and reports. One-step analysis makes one read-only step_core call and
presents the returned motion/landmark terms; it no longer recomputes inference
contributions. The prediction GC diagnostic keeps fused analysis aligned with
the actual post-motion landmark prediction. GUI layout and launch command are
unchanged; a further physical split of simulation/reports/UI files is optional.

## Reports

The last-experiment report contains:

- Position page: `x`, `y`, `theta`, and free energy.
- Velocity page: `x_dot`, `y_dot`, `theta_dot`.
- Acceleration page: `x_ddot`, `y_ddot`, `theta_ddot`.

Reports include the active algorithm, masterdata profile, demo path, precision
settings, distance precision modifier state, learning rates,
noise/slip/dropout/obstruction/FOV settings,
profile recognition (Unique names or Repeated names), landmark association gate,
and landmark scenario profile. Core determines association from each observed
name; no association mode is configured.

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
