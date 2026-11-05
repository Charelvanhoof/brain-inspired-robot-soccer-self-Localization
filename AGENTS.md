# Project Structure Guide

The project has two main modules. Their separation defines architectural
ownership: the core implements localization, and the simulator runs on top of it.

## `brain_inspired_localization_core.py`

The single implementation of localization math, generalized-coordinate updates,
landmark association, and inference diagnostics.

### Responsibilities

- Own the localization / active-inference algorithm and its configuration.
- Provide the headless `step_core(request)` integration boundary.
- Process one caller-supplied observation frame at a time.
- Provide shared state and coordinate-transform helpers.

### Strict rules

- The primary algorithm source is the paper
  `Robot_Soccer_Self_Localization_with_Active_Inference/Robot_Soccer_Self_Localization_with_Active_Inference_preprint.pdf`.
- Keep the mathematical formulation, algorithm structure, and terminology
  aligned with the paper. Explain proposed deviations before implementing them.
- Keep review-sensitive math explicit, well documented, and reviewable in isolation.
- For non-trivial algorithm functions, include `Input`, `Output`, and `Function`
  comments, followed by short comments matching the algorithm steps.
- Generalized coordinates are the single source of truth for robot state.
  Derive pose views when needed; do not maintain parallel persistent pose fields.
- Use the project's FCS/RCS conventions and documented units consistently.
- Use only the Python standard library. Remain independent of the simulator,
  GUI, plotting, and external map files.
- Do not generate simulated observations, run experiments, or produce reports.
- Keep caller-owned timing, sensor acquisition, repeated stepping, and transport
  outside this module.
- Keep detailed API documentation in `step_core_cpp_interface.txt`.

## `brain_inspired_localization_simulator.py`

Owns simulation, experiment configuration, reports, convergence evaluation,
robustness scans, one-step analysis, and UI. All these responsibilities run
on top of the shared core.

### Architectural blocks

- `SIMULATOR / UI CONFIGURATION`: display and interaction configuration.
- `INFERENCE / EXPERIMENT SETTINGS`: simulator-controlled inference settings
  and experiment configuration.
- `MASTERDATA`: simulator access to shared landmark profiles.
- `SIMULATION EXPERIMENT`: simulated sensors, core adapters, and experiment runners.
- `REPORT`: plots, summaries, datasets, and presentation-oriented output.
- `CONVERGENCE TESTS`: trial generation, convergence evaluation, and robustness scans.
- `ONE-STEP ANALYSIS`: diagnostic inspection and debugging views.
- `UI`: interactive state, demo playback, rendering, and the pygame entrypoint.

### Strict rules

- Call the shared core for localization; do not duplicate algorithm implementations.
- Import reusable state and transform helpers from core.
- Keep simulated sensor generation and experiment stepping in
  `SIMULATION EXPERIMENT`.
- Build analysis views from core diagnostics rather than reimplementing inference.
- Keep reporting, evaluation, and UI responsibilities in their respective blocks.
- Pose-shaped data may be used as a derived view for reporting and rendering.
- Keep UI code self-contained so it can later move into a separate module.
- The simulator is a POC harness, not a high-fidelity robot dynamics model.

### Block organization

Use the existing top-level separators for architectural blocks:

```python
# =========================
# BLOCK NAME
# =========================
```

- Place functions according to ownership, not convenience.
- Use small subheaders for local grouping within a block.

## Debugging And Change Scope

- When investigating an error or unexpected behavior, first identify and explain the most likely root cause.
- If the issue is a clear bug and the fix is narrow, local, and behavior-preserving, it is fine to implement directly.
- If a potential fix would change functionality, add safeguards, impose limits, alter UX, or otherwise go beyond the narrow bug itself, propose it first before implementing.
- Do not introduce broad defensive changes automatically during debugging unless they are explicitly requested.

## Coordinate System Summary

- Use the industrial-standard / ROS-style axis conventions described below, consistent with the paper.
- `FCS` means `Field Coordinate System`.
- In `FCS`, `(0, 0)` is the center point of the field.
- In `FCS`, `+X` points toward the enemy goal.
- In `FCS`, `+Y` points to the left.
- In `FCS`, `theta = 0` rad / `0` deg points toward the enemy goal, with positive rotation to the left.
- `RCS` means `Robot Coordinate System`.
- In `RCS`, `(0, 0)` is the center of the robot.
- In `RCS`, `+X` points straight forward from the robot point of view.
- In `RCS`, `+Y` points to the robot's left.
- In `RCS`, `theta = 0` rad / `0` deg points straight forward, with positive rotation to the left.
- Unless explicitly stated otherwise, all `theta` / angle values in `CORE` and related math are in radians.
- Landmark observations are expressed in `RCS`, while robot pose and landmark map positions on the field are expressed in `FCS`, unless a function explicitly documents a different boundary format.
- When implementing sensing transforms, keep the conversion direction explicit: robot pose and landmarks in `FCS` are translated into expected landmark observations in `RCS`.

# Python Environment

- Use `C:\Users\chare\AppData\Local\Python\pythoncore-3.13-64\python.exe` for this project's scripts, tests, and GUI checks. This interpreter has `pygame` installed and was verified with the user's running environment.
- Invoke it explicitly in PowerShell, for example:
  ```powershell
  & 'C:\Users\chare\AppData\Local\Python\pythoncore-3.13-64\python.exe' 'brain_inspired_localization_simulator.py'
  & 'C:\Users\chare\AppData\Local\Python\pythoncore-3.13-64\python.exe' -m unittest discover -q
  ```
- Bare `python` currently resolves to KiCad's bundled Python, which lacks `pygame`. Do not conclude the GUI cannot be tested from that interpreter's missing packages.
- This instruction does not change PATH or VS Code settings. If the recorded interpreter no longer exists, locate the current VS Code interpreter before running checks.

## Regression Tests

- Run the regression suite after changes to core or simulator code, and report
  the result. For behavior-preserving refactors, run it before and after.
- Use the project interpreter:
  ```powershell
  & 'C:\Users\chare\AppData\Local\Python\pythoncore-3.13-64\python.exe' -m unittest discover -q
  ```
- The suite covers frozen numerical regression, the public core interface,
  and convergence behavior. See `tests/README.md` for details.
- Frozen numerical cases use explicit settings, including learning rates;
  they do not evaluate performance with the current default settings.
- Do not regenerate the frozen baseline merely to make a failing test pass.
  Investigate the cause and explain intentional behavior changes first.

# Programming guidelines


## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
