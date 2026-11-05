Automated checks and their frozen reference data live in `regression_tests/`. From the project root, run `python -m unittest discover -s regression_tests -v`.

## Frozen regression baseline before the landmark refactor

Run the new numerical regression set and the existing convergence tests with:

```powershell
python -m unittest regression_tests.test_localization_regression regression_tests.test_convergence_symmetry -v
```

`regression_tests/localization_regression_baseline.json` records 16 deterministic scenarios from
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
python -m regression_tests.test_localization_regression --capture-baseline
```

Do not regenerate it merely to make a failing regression pass. During the
planned landmark and file refactors, keep the current expected numeric results
and adapt only the input/output adapters when field names or interfaces change.
These are behavior-preservation tests, not an independent proof of the math.

