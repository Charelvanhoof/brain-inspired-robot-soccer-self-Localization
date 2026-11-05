"""Initialize supplied maps and exercise the public one-frame API."""
from copy import deepcopy
from itertools import islice
import json
import math
import unittest
from unittest.mock import patch
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import brain_inspired_localization_core as core


class CoreInterfaceTests(unittest.TestCase):
    def setUp(self):
        # Each test restores the prior module configuration, including legacy views.
        from regression_tests.test_localization_regression import CORE_STATE_NAMES
        names = CORE_STATE_NAMES
        restore = patch.multiple(core, **deepcopy({name: getattr(core, name) for name in names}))
        restore.start()
        self.addCleanup(restore.stop)
        self.landmarks = [
            dict(name="corner", type="pose", x_cm=100, y_cm=50),
            dict(name="corner", type="pose", x_cm=-100, y_cm=-50),
            dict(name="center", type="pose", x_cm=0, y_cm=0),
            dict(name="bearing_cue", type="orientation", x_cm=3000, y_cm=200),
        ]

        self.learning_rates = {
            "landmark_xy": 1.6, "landmark_theta": 2.20325e-05 * 0.8,
            "motion_velocity_xy": 4.0, "motion_velocity_theta": 4.0,
            "motion_acceleration_xy": 40.0, "motion_acceleration_theta": 40.0,
        }
        core.init_learning_rates(self.learning_rates)
        core.init_landmarks({"profile": "test-map", "landmarks": self.landmarks})

    def request(self):
        actual = core.make_robot_gc_state(10, -20, 0.25)
        observations = [{"name": row["name"], "x_rcs_cm": row["x_rcs_cm"], "y_rcs_cm": row["y_rcs_cm"]}
                        for row in core.landmarks_fcs_2_rcs(actual, core.LANDMARKS)]
        return dict(timestamp="2026-10-27T12:00:00.025Z", step_dt_s=0.025,
                    belief_fcs_gc=core.make_robot_gc_state(15, -15, 0.3),
                    landmark_observations_rcs=observations,
                    motion_observation={key: 0 for key in ("x_vel_cm_s", "y_vel_cm_s", "theta_vel_rad_s",
                                                           "x_acc_cm_s2", "y_acc_cm_s2", "theta_acc_rad_s2")})

    def test_core_import_needs_no_landmark_file_or_gui(self):
        with tempfile.TemporaryDirectory(prefix="standalone-localization-core-") as directory:
            shutil.copyfile(core.__file__, Path(directory) / "brain_inspired_localization_core.py")
            code = (f"import sys; sys.path.insert(0, {directory!r}); "
                    "import brain_inspired_localization_core as core; "
                    "assert core.LANDMARK_MAP == [] and core.ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME is None; "
                    "assert not hasattr(core, 'LANDMARK_SCENARIO_PROFILES'); "
                    "assert core.LEARNING_RATES is None; "
                    "assert not hasattr(core, 'LR_XY') and not hasattr(core, 'MOTION_LR_VEL_XY'); "
                    "assert not hasattr(core, 'CORE_SETTING_DEFAULTS'); "
                    "assert 'pygame' not in sys.modules and 'matplotlib' not in sys.modules; "
                    "assert not hasattr(core, 'initialize_core'); "
                    "assert not hasattr(core, 'NOISE_ENABLED')")
            result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_profile_selection_replaces_map_and_association_without_changing_gains(self):
        core.init_landmarks({"profile": "two corners", "landmarks": self.landmarks[:2]})
        self.assertEqual(core.landmark_candidate_ids_for_name("corner"), ["0", "1"])
        core.init_landmarks({"profile": "center only", "landmarks": self.landmarks[2:3]})
        self.assertEqual(core.unique_landmark_candidate_id_for_name("center"), "0")
        self.assertFalse(hasattr(core, "ACTIVE_DATA_ASSOCIATION_NAME"))
        self.assertFalse(hasattr(core, "DATA_ASSOCIATION_UNIQUE_ID_NAME"))
        self.assertFalse(hasattr(core, "DATA_ASSOCIATION_UNIQUE_DIRECT_NAME"))
        self.assertEqual(core.LEARNING_RATES["landmark_xy"], 1.6)
        self.assertFalse(hasattr(core, "initialize_core"))
        self.assertFalse(hasattr(core, "get_core_configuration"))

    def test_request_response_and_diagnostics_are_detached_json_data(self):
        request = json.loads(json.dumps(self.request()))
        original = deepcopy(request)
        response = core.step_core(request)
        self.assertEqual(request, original)
        self.assertEqual(json.loads(json.dumps(response, allow_nan=False)), response)
        reference = deepcopy(response)
        response["belief_fcs_gc"]["x"][0] = 999
        response["diagnostics"]["settings"]["learning_rates"]["landmark_xy"] = 999
        response["diagnostics"]["landmark_terms"][0]["eps"][0] = 999
        self.assertEqual(core.step_core(request), reference)
        self.assertEqual(core.LEARNING_RATES["landmark_xy"], 1.6)

    def test_landmark_initialization_copies_data_and_requires_explicit_reload(self):
        profile = {"profile": "profile name", "landmarks": deepcopy(self.landmarks)}
        original = deepcopy(profile)
        result = core.init_landmarks(profile)
        request = self.request()
        reference = core.step_core(request)
        self.assertEqual(profile, original)
        self.assertEqual(result, {"profile": "profile name", "landmark_count": 4})
        self.assertEqual(reference["diagnostics"]["profile"], "profile name")
        profile["profile"] = "changed field"
        profile["landmarks"][0]["x_cm"] = 500
        profile["landmarks"][0]["name"] = "new beacon"
        result["profile"] = "changed response"
        self.assertEqual(core.step_core(request), reference)
        core.init_landmarks(profile)
        self.assertEqual(core.ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME, "changed field")
        self.assertEqual(core.LANDMARKS["0"], (500, 50))
        self.assertEqual(core.unique_landmark_candidate_id_for_name("new beacon"), "0")
        profile["landmarks"][0]["y_cm"] = 75
        core.init_landmarks(profile)  # Reloading also works with the same profile label.
        self.assertEqual(core.LANDMARKS["0"], (500, 75))

    def test_learning_rate_initialization_copies_data_and_preserves_map_and_belief(self):
        supplied = self.learning_rates.copy()
        request = self.request()
        original_request = deepcopy(request)
        original_map = deepcopy(core.LANDMARK_MAP)
        result = core.init_learning_rates(supplied)
        self.assertEqual(result, supplied)
        reference = core.step_core(request)
        supplied["landmark_xy"] = 0.25
        result["landmark_theta"] = 0
        self.assertEqual(core.step_core(request), reference)
        core.init_learning_rates(supplied)
        self.assertEqual(core.LEARNING_RATES, supplied)
        self.assertEqual(core.step_core(request)["diagnostics"]["settings"]["learning_rates"], supplied)
        self.assertNotEqual(core.step_core(request)["belief_fcs_gc"], reference["belief_fcs_gc"])
        self.assertEqual(core.LANDMARK_MAP, original_map)
        self.assertEqual(request, original_request)

    def test_invalid_learning_rates_preserve_active_configuration(self):
        reference = core.step_core(self.request())
        invalid = [None, {}, {**self.learning_rates, "unknown_rate": 1}]
        for name in self.learning_rates:
            missing = self.learning_rates.copy()
            del missing[name]
            invalid.append(missing)
            for value in (-1, math.nan, math.inf, None, True, "1.6"):
                invalid.append({**self.learning_rates, name: value})
        for supplied in invalid:
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                core.init_learning_rates(supplied)
            self.assertEqual(core.LEARNING_RATES, self.learning_rates)
            self.assertEqual(core.step_core(self.request()), reference)

    def test_all_inference_modes_require_learning_rate_initialization(self):
        request = self.request()
        with patch.object(core, "LEARNING_RATES", None):
            for algorithm in ("landmark only", "motion only", "motion+landmark"):
                request["algorithm_name"] = algorithm
                original = deepcopy(request["belief_fcs_gc"])
                with self.subTest(algorithm=algorithm), self.assertRaisesRegex(ValueError, "init_learning_rates"):
                    core.step_core(request)
                with self.assertRaisesRegex(ValueError, "init_learning_rates"):
                    core.inference_step_from_observations(
                        request["belief_fcs_gc"], motion_observation=request["motion_observation"],
                        algorithm_name=algorithm, step_dt_s=0.025)
                self.assertEqual(request["belief_fcs_gc"], original)
            with self.assertRaisesRegex(ValueError, "init_learning_rates"):
                core.motion_inference_step(request["belief_fcs_gc"], request["motion_observation"])
            with self.assertRaisesRegex(ValueError, "init_learning_rates"):
                core.landmark_inference_step(request["belief_fcs_gc"], {})

    def test_zero_learning_rates_disable_corrections_but_keep_generalized_motion(self):
        core.init_learning_rates(dict.fromkeys(self.learning_rates, 0))
        request = self.request()
        request["algorithm_name"] = "landmark only"
        request["belief_fcs_gc"]["theta"][0] = 0
        self.assertEqual(core.step_core(request)["belief_fcs_gc"], request["belief_fcs_gc"])
        request["algorithm_name"] = "motion+landmark"
        for axis in ("x", "y", "theta"):
            request["belief_fcs_gc"][axis][1:] = [2, 3]
        response = core.step_core(request)
        for axis in ("x", "y", "theta"):
            position, velocity, acceleration = request["belief_fcs_gc"][axis]
            self.assertAlmostEqual(response["belief_fcs_gc"][axis][0], position + 0.025 * velocity)
            self.assertAlmostEqual(response["belief_fcs_gc"][axis][1], velocity + 0.025 * acceleration)
            self.assertEqual(response["belief_fcs_gc"][axis][2], acceleration)

    def test_each_learning_rate_controls_its_own_correction(self):
        targets = {
            "landmark_xy": {("x", 0), ("y", 0)},
            "landmark_theta": {("theta", 0)},
            "motion_velocity_xy": {("x", 1), ("y", 1)},
            "motion_velocity_theta": {("theta", 1)},
            "motion_acceleration_xy": {("x", 2), ("y", 2)},
            "motion_acceleration_theta": {("theta", 2)},
        }
        for name, expected in targets.items():
            request = self.request()
            request["algorithm_name"] = "landmark only" if name.startswith("landmark") else "motion only"
            for axis in ("x", "y", "theta"):
                request["belief_fcs_gc"][axis][1:] = [2, 3]
            rates = dict.fromkeys(self.learning_rates, 0.0)
            core.init_learning_rates(rates)
            baseline = core.step_core(request)["belief_fcs_gc"]
            rates[name] = 0.5
            core.init_learning_rates(rates)
            updated = core.step_core(request)["belief_fcs_gc"]
            changed = {(axis, order) for axis in ("x", "y", "theta") for order in range(3)
                       if not math.isclose(updated[axis][order], baseline[axis][order], abs_tol=1e-12)}
            with self.subTest(rate=name):
                self.assertEqual(changed, expected)

    def test_invalid_landmark_initialization_preserves_active_configuration(self):
        reference = core.step_core(self.request())
        invalid_profiles = [None, {}, {"profile": 42, "landmarks": []},
                            {"profile": "field"}, {"profile": "field", "landmarks": {}},
                            {"profile": "field", "landmarks": [None]},
                            {"profile": "field", "landmarks": [{"name": "beacon"}]}]
        for key, value in (("name", None), ("type", "corner"), ("x_cm", math.inf),
                           ("y_cm", math.nan), ("x_cm", None), ("y_cm", "bad")):
            # A valid first row must not be applied before a later invalid row fails.
            rows = deepcopy(self.landmarks[:2])
            rows[1][key] = value
            invalid_profiles.append({"profile": "bad field", "landmarks": rows})
        for profile in invalid_profiles:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                core.init_landmarks(profile)
            self.assertEqual(core.step_core(self.request()), reference)

    def test_landmark_inference_requires_initialization_but_motion_only_does_not(self):
        request = self.request()
        with patch.multiple(core, ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME=None,
                            LANDMARK_MAP=[], LANDMARK_POINT_IDS=(), LANDMARKS={}, LANDMARK_METADATA={}):
            for algorithm in ("landmark only", "motion+landmark"):
                request["algorithm_name"] = algorithm
                original = deepcopy(request["belief_fcs_gc"])
                with self.subTest(algorithm=algorithm), self.assertRaisesRegex(ValueError, "init_landmarks"):
                    core.step_core(request)
                with self.assertRaisesRegex(ValueError, "init_landmarks"):
                    core.inference_step_from_observations(
                        request["belief_fcs_gc"], motion_observation=request["motion_observation"],
                        algorithm_name=algorithm, step_dt_s=0.025)
                self.assertEqual(request["belief_fcs_gc"], original)
            with self.assertRaisesRegex(ValueError, "init_landmarks"):
                core.landmark_inference_step(request["belief_fcs_gc"], {})
            request["algorithm_name"] = "motion only"
            self.assertEqual(core.step_core(request)["diagnostics"]["landmark_terms"], [])

    def test_explicitly_initialized_empty_map_replaces_previous_map(self):
        result = core.init_landmarks({"profile": "no landmarks", "landmarks": []})
        self.assertEqual(result, {"profile": "no landmarks", "landmark_count": 0})
        self.assertEqual(core.LANDMARKS, {})
        self.assertEqual(core.LANDMARK_METADATA, {})
        request = self.request()
        request["algorithm_name"] = "landmark only"
        self.assertEqual(core.step_core(request)["diagnostics"]["free_energy"], 0)

    def test_diagnostics_describe_actual_updates_and_ignored_orientation(self):
        request = self.request()
        response = core.step_core(request)
        diagnostics = response["diagnostics"]
        used = [term for term in diagnostics["landmark_terms"] if term["used_for_update"]]
        ignored = [term for term in diagnostics["landmark_terms"] if not term["used_for_update"]]
        self.assertEqual(len(used), 3)
        self.assertEqual({term["association_name"] for term in used}, {"direct", "gated nearest"})
        self.assertEqual(len(ignored), 1)
        self.assertEqual(ignored[0]["skip_reason"], "orientation_landmark")
        self.assertAlmostEqual(sum(term["free_energy_contribution"] for term in used), diagnostics["landmark_free_energy"])
        for index, axis in enumerate(("x", "y", "theta")):
            predicted = request["belief_fcs_gc"][axis][0] + request["step_dt_s"] * sum(term["mu_dot_contribution"][index] for term in used)
            if axis == "theta":
                predicted = core.wrap_rad(predicted)
            self.assertAlmostEqual(response["belief_fcs_gc"][axis][0], predicted)
        self.assertEqual(diagnostics["timestamp"], request["timestamp"])
        self.assertEqual(diagnostics["step_dt_s"], 0.025)

    def test_required_timestamp_positive_dt_and_finite_inputs(self):
        request = self.request()
        request["timestamp"] = None
        with self.assertRaises(ValueError):
            core.step_core(request)
        for dt in (0, -1, math.nan, math.inf):
            request = self.request()
            request["step_dt_s"] = dt
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                core.step_core(request)
        for target in ("belief", "landmark", "motion"):
            request = self.request()
            if target == "belief":
                request["belief_fcs_gc"]["x"][0] = math.nan
            elif target == "landmark":
                request["landmark_observations_rcs"][0]["x_rcs_cm"] = math.inf
            else:
                request["motion_observation"]["x_vel_cm_s"] = math.inf
            with self.subTest(target=target), self.assertRaises(ValueError):
                core.step_core(request)

    def test_timestamp_is_informational_and_echoed_unchanged(self):
        request = self.request()
        reference = core.step_core(request)
        for timestamp in ("", "frame 42", "2026-10-27T12:00:00Z", "2026-02-30T12:00:00.025Z",
                          "2026-10-27T12:00:00.025+01:00"):
            request["timestamp"] = timestamp
            with self.subTest(timestamp=timestamp):
                response = core.step_core(request)
                self.assertEqual(response["diagnostics"]["timestamp"], timestamp)
                response["diagnostics"]["timestamp"] = reference["diagnostics"]["timestamp"]
                self.assertEqual(response, reference)

    def test_motion_only_uses_no_landmark_updates(self):
        request = self.request()
        request["algorithm_name"] = "motion only"
        response = core.step_core(request)
        self.assertEqual(response["diagnostics"]["landmark_terms"], [])

    def test_selected_core_profile_replays_frozen_numeric_reference(self):
        from regression_tests import test_localization_regression as regression
        data = json.loads(regression.BASELINE.read_text(encoding="utf-8"))
        compare = regression.LocalizationRegressionTests().assert_close
        references = {"gated_near_poster": "gated_symmetric_9", "gated_far_poster": "gated_near_9"}
        for case in data["scenarios"]:
            with self.subTest(case=case["name"]):
                mapped = regression.profile_for_case(case)["landmarks"]
                core.init_learning_rates({name: regression.SETTINGS[setting]
                                          for name, setting in regression.LEARNING_RATE_SETTINGS.items()})
                settings = {key: value for key, value in regression.SETTINGS.items()
                            if key != "DT" and key not in regression.LEARNING_RATE_SETTINGS.values()}
                profile = regression.profile_for_case(case)
                core.init_landmarks({"profile": profile["name"], "landmarks": profile["landmarks"]})
                for name, value in settings.items():
                    setattr(core, name, value)
                reference = next(row for row in data["scenarios"] if row["name"] == references.get(case["name"], case["name"]))["expected"]
                checkpoints = {row["step"]: row for row in reference["trace"]}
                belief = deepcopy(case["initial_belief"])
                frames = (frame for frame in case["frames"] for _ in range(frame["repeat"]))
                for step, frame in enumerate(islice(frames, reference["steps"]), 1):
                    response = core.step_core(dict(timestamp="2026-10-27T12:00:00.025Z", step_dt_s=0.025,
                                                   belief_fcs_gc=belief, algorithm_name=case["algorithm"],
                                                   landmark_observations_rcs=regression.named_observations(case, frame["landmark_observations_rcs"]),
                                                   motion_observation=frame["motion_observation"]))
                    belief = response["belief_fcs_gc"]
                    if step in checkpoints:
                        compare(dict(step=step, belief=belief, **{key: response["diagnostics"][key]
                                     for key in ("free_energy", "motion_free_energy", "landmark_free_energy")}), checkpoints[step])


if __name__ == "__main__":
    unittest.main()
