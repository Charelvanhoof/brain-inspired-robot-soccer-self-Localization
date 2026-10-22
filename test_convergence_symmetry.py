"""Run with: python -m unittest test_convergence_symmetry -v."""
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "localisation_simulator", Path(__file__).with_name("brain inspired localisation.py"),
)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


class ConvergenceSymmetryTests(unittest.TestCase):
    def setUp(self):
        self.actual = {"x_cm": 200.0, "y_cm": -100.0, "theta_rad": 0.25}
        self.symmetrical = {"x_cm": -200.0, "y_cm": 100.0, "theta_rad": 0.25 - math.pi}

    def history(self, beliefs):
        return [{"actual_pose": self.actual, "belief_pose": belief} for belief in beliefs]

    def test_symmetrical_requires_five_consecutive_hits(self):
        four_hits = [self.symmetrical] * 4
        self.assertFalse(sim.history_has_sequential_convergence(self.history(four_hits), symmetrical=True))
        self.assertFalse(sim.history_has_sequential_convergence(
            self.history(four_hits + [self.actual] + four_hits), symmetrical=True,
        ))
        five_hits = self.history([self.symmetrical] * 5)
        self.assertTrue(sim.history_has_sequential_convergence(five_hits, symmetrical=True))
        self.assertFalse(sim.history_has_sequential_convergence(five_hits))
        self.assertTrue(sim.history_has_sequential_convergence(self.history([self.actual] * 5)))

    def test_symmetrical_tolerances_and_angle_wrapping(self):
        # The equivalent angle above already crosses the +/-pi representation boundary.
        self.assertTrue(sim.pose_symmetrically_converged(self.actual, self.symmetrical))
        for key, tolerance in (
            ("x_cm", sim.CONV_TOL_X_CM), ("y_cm", sim.CONV_TOL_Y_CM),
            ("theta_rad", sim.CONV_TOL_ANGLE_RAD),
        ):
            inside = dict(self.symmetrical)
            inside[key] += tolerance - 1e-8
            self.assertTrue(sim.pose_symmetrically_converged(self.actual, inside))
            outside = dict(self.symmetrical)
            outside[key] += tolerance + 1e-8
            self.assertFalse(sim.pose_symmetrically_converged(self.actual, outside))

    def test_moving_symmetrical_target(self):
        history = []
        for step in range(5):
            actual = {"x_cm": 200.0 + step * 20, "y_cm": -100.0 + step, "theta_rad": 0.25 + step * 0.1}
            belief = {"x_cm": -actual["x_cm"], "y_cm": -actual["y_cm"], "theta_rad": actual["theta_rad"] + math.pi}
            history.append({"actual_pose": actual, "belief_pose": belief})
        self.assertTrue(sim.history_has_sequential_convergence(history, symmetrical=True))
        self.assertFalse(sim.history_has_sequential_convergence(history))

    def test_all_five_convergence_modes_have_exclusive_counts(self):
        actual = {"x": 200.0, "y": -100.0, "theta": 0.25}
        cases = [
            {"actual": actual, "belief": actual},
            {"actual": actual, "belief": {"x": -200.0, "y": 100.0, "theta": 0.25 - math.pi}},
            {"actual": actual, "belief": {"x": 0.0, "y": 0.0, "theta": 0.0}},
        ]
        with patch.object(sim, "INFER_STEPS_MAX", 360), patch.object(
            sim, "simulation_inference_step", return_value=({}, {}, {}, 0.0),
        ), patch.object(sim, "simulation_sense_landmarks", return_value=[]), patch.object(
            sim, "reset_demo_playback_state",
        ), patch.object(sim, "step_demo_playback"):
            for dynamic, mode in ((False, "random"), (False, "nearby"), (False, "outside_corner"), (True, "random"), (True, "nearby")):
                with self.subTest(dynamic=dynamic, mode=mode):
                    robots = {"actual": sim.make_robot_gc_state(), "belief": sim.make_robot_gc_state()}
                    progress = []
                    kwargs = dict(start_mode=mode, progress_callback=lambda *args: progress.append(args))
                    if dynamic:
                        report = sim.run_dynamic_converge_test(
                            robots, sim.ACTIVE_ALGORITHM_NAME, [{"name": "test"}], 0, start_cases=cases, **kwargs,
                            look_target_x_cm=300.0, look_target_y_cm=-50.0,
                        )
                        self.assertEqual((report["look_target_x_cm"], report["look_target_y_cm"]), (300.0, -50.0))
                    else:
                        report = sim.run_converge_test(
                            robots, sim.ACTIVE_ALGORITHM_NAME,
                            trial_specs=[{"actual_start": c["actual"], "belief_start": c["belief"]} for c in cases],
                            **kwargs,
                        )
                    self.assertEqual((report["converged"], report["non_converged_symmetrical"], report["non_converged"]), (1, 1, 1))
                    self.assertEqual(report["trial_steps"], [5, 5, 360])
                    self.assertEqual(report["converged_steps"], [5])
                    self.assertEqual([f["symmetrical"] for f in report["failures"]], [True, False])
                    if dynamic:
                        self.assertEqual(
                            [record["step"] for record in report["failures"][-1]["path_history"]],
                            list(range(61, 361)),
                        )
                    self.assertEqual(progress[-1], (3, 3, 1, 2))

    def test_ordinary_experiment_does_not_stop_at_symmetrical_pose(self):
        robots = {
            "actual": sim.make_robot_gc_state(200.0, -100.0, 0.25),
            "belief": sim.make_robot_gc_state(-200.0, 100.0, 0.25 - math.pi),
        }
        with patch.object(sim, "simulation_inference_step", return_value=({}, {}, {}, 0.0)):
            history = sim.run_experiment(robots, sim.ACTIVE_ALGORITHM_NAME, 6, until_convergence=True)
        self.assertEqual(len(history), 6)
        self.assertFalse(sim.history_has_sequential_convergence(history))


if __name__ == "__main__":
    unittest.main()
