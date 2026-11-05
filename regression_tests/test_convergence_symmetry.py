"""Run with: python -m unittest regression_tests.test_convergence_symmetry -v."""
import importlib.util
import math
from pathlib import Path
import unittest
import brain_inspired_localization_core as core
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "localization_simulator", Path(__file__).resolve().parent.parent / "brain_inspired_localization_simulator.py",
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


class PosterLandmarkTests(unittest.TestCase):
    def test_far_profile_uses_only_its_distinct_poster(self):
        profile = sim.landmark_scenario_profile_by_name("9 ambiguous +1 orientation far")
        self.assertEqual(len(profile["landmarks"]), 10)
        self.assertEqual(profile["landmarks"][-1], dict(name="poster2", type="orientation", x_cm=4900.0, y_cm=300.0))
        with patch.multiple(sim, ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME=profile["name"],
                            NOISE_ENABLED=False, VIEWING_ANGLE_ENABLED=False,
                            LANDMARK_DROPOUT_ENABLED=False, LANDMARK_OBSTRUCTION_ENABLED=False):
            robot = sim.make_robot_gc_state()
            observations = sim.simulation_sense_landmarks(robot, sim.active_simulation_landmarks())
            sim.ensure_simulation_core_configuration()
            predictions = core.obs_list_to_map(core.landmarks_fcs_2_rcs(robot, core.LANDMARKS))
            poster = next(obs for obs in observations if obs["name"] == "poster2")
            self.assertNotIn("landmark_id", poster)
            self.assertNotIn("poster1", {obs["name"] for obs in observations})
            association = core.associate_landmark_observation(sim.calculate_angle({"poster": poster})["poster"], predictions)
            self.assertTrue(association["accepted"])
            self.assertEqual(association["candidate_id"], "9")

    def test_poster_profile_and_standalone_map_agree(self):
        with patch.object(sim, "ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME", "9 ambiguous +1 orientation close"):
            sim.ensure_simulation_core_configuration()
            self.assertEqual(core.LANDMARK_MAP, sim.active_landmark_scenario_profile()["landmarks"])
            self.assertEqual(core.LANDMARKS["9"], (1200.0, 300.0))
            self.assertEqual(core.unique_landmark_candidate_id_for_name("poster1"), "9")

    def test_poster_is_asymmetric_and_respects_field_of_view(self):
        actual = sim.make_robot_gc_state(200, -100, 0.25)
        mirrored = sim.make_robot_gc_state(-200, 100, 0.25 - math.pi)
        with patch.multiple(sim, ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME="9 ambiguous +1 orientation close",
                            NOISE_ENABLED=False, VIEWING_ANGLE_ENABLED=False,
                            LANDMARK_DROPOUT_ENABLED=False, LANDMARK_OBSTRUCTION_ENABLED=False):
            observations = sim.simulation_sense_landmarks(actual, sim.active_simulation_landmarks())
            poster = next(obs for obs in observations if obs["name"] == "poster1")
            sim.ensure_simulation_core_configuration()
            predictions = core.obs_list_to_map(core.landmarks_fcs_2_rcs(mirrored, core.LANDMARKS))
            association = core.associate_landmark_observation(sim.calculate_angle({"poster": poster})["poster"], predictions)
            self.assertTrue(association["accepted"])
            self.assertAlmostEqual(association["position_error_cm"], 2 * math.hypot(1200, 300))
            with patch.object(sim, "VIEWING_ANGLE_ENABLED", True):
                heading = math.atan2(300, 1200)
                facing = sim.make_robot_gc_state(0, 0, heading)
                away = sim.make_robot_gc_state(0, 0, heading + math.pi)
                poster_map = {"9": (1200.0, 300.0)}
                self.assertEqual(len(sim.simulation_sense_landmarks(facing, poster_map)), 1)
                self.assertEqual(sim.simulation_sense_landmarks(away, poster_map), [])

    def test_orientation_posters_do_not_displace_symmetric_tracking_belief(self):
        with patch.multiple(sim, DT=0.025, LR_XY=1.6, LR_ANGLE=sim.DEFAULT_ANGLE_LR * 0.8,
                            NOISE_ENABLED=False, VIEWING_ANGLE_ENABLED=False,
                            LANDMARK_DROPOUT_ENABLED=False, LANDMARK_OBSTRUCTION_ENABLED=False,
                            PRECISION_DISTANCE_MODIFIER_ENABLED=True,
                            ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME="9 ambiguous"):
            for profile, stays_symmetrical in (("9 ambiguous", True), ("9 ambiguous +1 orientation close", True), ("9 ambiguous +1 orientation far", True)):
                with self.subTest(profile=profile), patch.object(sim, "ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME", profile):
                    robots = {
                        "actual": sim.make_robot_gc_state(200.0, -100.0, 0.25),
                        "belief": sim.make_robot_gc_state(-200.0, 100.0, 0.25 - math.pi),
                    }
                    sim.simulation_inference_step(robots, sim.LANDMARK_ONLY_ALGORITHM_NAME)
                    self.assertEqual(
                        sim.pose_symmetrically_converged(
                            sim.robot_pose_view(robots["actual"]), sim.robot_pose_view(robots["belief"]),
                        ), stays_symmetrical,
                    )


if __name__ == "__main__":
    unittest.main()
