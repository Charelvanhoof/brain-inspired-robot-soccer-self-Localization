# Brain Inspired Localisation POC

This project is a proof of concept for robot self-localisation using active
inference. It simulates a robot on a RoboCup-style field, compares the actual
robot state with the belief state, and lets you test landmark-only,
motion-only, and fused localisation.

The simulator is intentionally basic. It is mainly used to feed observations
into the localisation algorithm and to generate reports, not to be a
high-fidelity robot physics model.

## Project Files

- `brain inspired localisation.py`: interactive POC simulator, inference core,
  experiments, convergence tests, and reports (POC V29).
- `brain_inspired_localisation_core.py`: standalone localisation core module
  exposing `step_core(request)` for integration without the GUI.
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
python ".\brain inspired localisation.py"
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
- Use the association button to choose how landmark observations are matched to
  known field landmarks before the update:
  - `Gated nearest`
  - `Unique ID`
- Use `PROFILE` to cycle masterdata settings:
  - `40Hz`, with `dt = 0.025 s`
  - `200Hz`, with `dt = 0.005 s`
- Use `DIST PREC` to select the distance precision modifier (YES/NO):
  - `YES`: reduce landmark precision as the predicted landmark distance grows.
  - `NO`: apply no distance weighting to landmark precision.
  Switching this option reapplies the selected frequency's preset, including
  its learning rates, replacing any manual adjustments to those settings.
- Use `LM PROFILE` to cycle simulated landmark sets:
  - `9+1far`: the nine field landmarks plus uniquely recognized `poster2` at
    FCS (4900, 300) cm, 40 m behind the enemy goal line with the same left
    offset as poster1. Only poster2 is observed in this profile.
  - `9+1clo`: the nine field landmarks plus a uniquely recognized `poster1` at
    FCS (1200, 300) cm, 3 m behind the enemy goal line and 3 m to the left of
    the goal centre (looking toward the enemy goal). The poster
    supplies asymmetric evidence when observed and lies outside the normal GUI
    field view. The usual FoV, noise, dropout, and obstruction settings apply.
  - `9`: all landmarks
  - `2`: both goals
  - `1`: center
  - `3`: center and both goals
  - `5`: both goals, center, left middle, right middle
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
python ".\brain inspired localisation.py" --export-paper-data --paper-data-profile 40Hz --paper-data-duration-s 7200 --paper-data-path "Circle" --paper-data-start nearby
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

- The heart of the localisation / active-inference algorithm is in the `CORE`
  block of the Python file. The UI, reports, tests, and demo simulator sit on
  top of that.
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
