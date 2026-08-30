# Brain Inspired Localisation POC

This project is a proof of concept for robot self-localisation using active
inference. It simulates a robot on a RoboCup-style field, compares the actual
robot state with the belief state, and lets you test landmark-only,
motion-only, and fused localisation.

The simulator is intentionally basic. It is mainly used to feed observations
into the localisation algorithm and to generate reports, not to be a
high-fidelity robot physics model.

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
- Use `PROFILE` to cycle masterdata settings. V27 includes:
  - `40Hz`, with `dt = 0.025 s`
  - `200Hz`, with `dt = 0.005 s`
- Use `LM PROFILE` to cycle simulated landmark sets:
  - `9`: all landmarks
  - `2`: both goals
  - `5`: both goals, center, left middle, right middle
- Use `STEP 1`, `STEP 100`, or `STEP 2000` to run an inference experiment.
- For motion algorithms, the selected demo path plays for the chosen number of
  steps and the data is stored for the report.
- Use `PLAY DEMO` to play the selected demo path continuously.
- Use the demo-path button to cycle through available paths.
- Use `PLOT LAST EXPERIMENT` to create a multipage PDF report in the `reports`
  folder.
- Use `PAPER DATA 288K` to export a paper-style tracking dataset for the
  selected profile/path. It writes CSV and JSON metadata under `reports`.
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

Convergence tests write a PDF report to the `reports` folder. V27 reports
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
settings, learning rates, noise/slip/dropout/obstruction/FOV settings,
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
