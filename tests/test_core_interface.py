"""Select built-in profiles and exercise the public one-frame API."""
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
        names = (*core.CORE_SETTING_DEFAULTS, "LANDMARK_MAP", "LANDMARK_POINT_IDS", "LANDMARKS", "LANDMARK_METADATA", "ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME")
        restore = patch.multiple(core, **deepcopy({name: getattr(core, name) for name in names}))
        restore.start()
        self.addCleanup(restore.stop)
        self.landmarks = [
            dict(name="corner", type="pose", x_cm=100, y_cm=50),
            dict(name="corner", type="pose", x_cm=-100, y_cm=-50),
            dict(name="center", type="pose", x_cm=0, y_cm=0),
            dict(name="bearing_cue", type="orientation", x_cm=3000, y_cm=200),
        ]

        profiles = patch.object(core, "LANDMARK_SCENARIO_PROFILES", core.LANDMARK_SCENARIO_PROFILES + [
            dict(name="test-map", landmarks=self.landmarks)])
        profiles.start()
        self.addCleanup(profiles.stop)
        core.set_landmark_profile("test-map")

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
                    "assert len(core.LANDMARK_MAP) == 9; "
                    "assert 'pygame' not in sys.modules and 'matplotlib' not in sys.modules; "
                    "assert not hasattr(core, 'initialize_core'); "
                    "assert not hasattr(core, 'NOISE_ENABLED')")
            result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_profile_selection_replaces_map_and_association_without_changing_gains(self):
        core.LR_XY = 1.6
        core.set_landmark_profile("2 ambiguous")
        self.assertEqual(core.landmark_candidate_ids_for_name("goal"), ["0", "1"])
        core.set_landmark_profile("9 unique")
        self.assertEqual(core.unique_landmark_candidate_id_for_name("corner_TR"), "3")
        self.assertEqual(core.ACTIVE_DATA_ASSOCIATION_NAME, "Unique ID")
        self.assertEqual(core.LR_XY, 1.6)
        self.assertFalse(hasattr(core, "initialize_core"))
        self.assertFalse(hasattr(core, "get_core_configuration"))

    def test_request_response_and_diagnostics_are_detached_json_data(self):
        core.LR_XY = 1.6
        request = json.loads(json.dumps(self.request()))
        original = deepcopy(request)
        response = core.step_core(request)
        self.assertEqual(request, original)
        self.assertEqual(json.loads(json.dumps(response, allow_nan=False)), response)
        reference = deepcopy(response)
        response["belief_fcs_gc"]["x"][0] = 999
        response["diagnostics"]["settings"]["LR_XY"] = 999
        response["diagnostics"]["landmark_terms"][0]["eps"][0] = 999
        self.assertEqual(core.step_core(request), reference)
        self.assertEqual(core.LR_XY, 1.6)

    def test_diagnostics_describe_actual_updates_and_ignored_orientation(self):
        request = self.request()
        response = core.step_core(request)
        diagnostics = response["diagnostics"]
        used = [term for term in diagnostics["landmark_terms"] if term["used_for_update"]]
        ignored = [term for term in diagnostics["landmark_terms"] if not term["used_for_update"]]
        self.assertEqual(len(used), 3)
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
        from tests import test_localization_regression as regression
        data = json.loads(regression.BASELINE.read_text(encoding="utf-8"))
        compare = regression.LocalizationRegressionTests().assert_close
        references = {"gated_near_poster": "gated_symmetric_9", "gated_far_poster": "gated_near_9"}
        for case in data["scenarios"]:
            with self.subTest(case=case["name"]):
                mapped = regression.profile_for_case(case)["landmarks"]
                settings = {key: value for key, value in regression.SETTINGS.items() if key != "DT"}
                settings["ACTIVE_DATA_ASSOCIATION_NAME"] = case["association"]
                core.set_landmark_profile(regression.profile_for_case(case)["name"])
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
