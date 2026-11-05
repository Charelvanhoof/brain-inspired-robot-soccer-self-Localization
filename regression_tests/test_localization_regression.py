"""Frozen pre-refactor regression cases; run python -m unittest regression_tests.test_localization_regression -v.

Capture explicitly with python -m regression_tests.test_localization_regression --capture-baseline.
Never recapture merely to make a failed regression pass.
"""
from contextlib import ExitStack
from copy import deepcopy
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import brain_inspired_localization_core as core

ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).with_name("localization_regression_baseline.json")
spec = importlib.util.spec_from_file_location("regression_simulator", ROOT / "brain_inspired_localization_simulator.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

# Pin actual numeric settings, rather than inheriting import/GUI defaults.
SETTINGS = dict(DT=0.025, PI_CAM=1.0, PI_ODO=1.0, PI_IMU=1.0,
                LR_XY=1.6, LR_ANGLE=2.20325e-05 * 0.8, ANGLE_WEIGHT=1.0,
                MOTION_LR_VEL_XY=4.0, MOTION_LR_VEL_THETA=4.0,
                MOTION_LR_ACC_XY=40.0, MOTION_LR_ACC_THETA=40.0,
                PRECISION_DISTANCE_MODIFIER_ENABLED=True,
                DISTANCE_PRECISION_REF_CM=600.0, DISTANCE_PRECISION_MIN_FACTOR=0.15,
                LANDMARK_ASSOCIATION_GATE_CM=1000.0)
LEARNING_RATE_SETTINGS = {
    "landmark_xy": "LR_XY", "landmark_theta": "LR_ANGLE",
    "motion_velocity_xy": "MOTION_LR_VEL_XY", "motion_velocity_theta": "MOTION_LR_VEL_THETA",
    "motion_acceleration_xy": "MOTION_LR_ACC_XY", "motion_acceleration_theta": "MOTION_LR_ACC_THETA",
}
CORE_STATE_NAMES = ("LEARNING_RATES", "PI_CAM", "PI_ODO", "PI_IMU", "ANGLE_WEIGHT",
                    "PRECISION_DISTANCE_MODIFIER_ENABLED", "DISTANCE_PRECISION_REF_CM",
                    "DISTANCE_PRECISION_MIN_FACTOR", "LANDMARK_ASSOCIATION_GATE_CM",
                    "LANDMARK_MAP", "LANDMARK_POINT_IDS", "LANDMARKS", "LANDMARK_METADATA",
                    "ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME")


SENSORS = dict(NOISE_ENABLED=False, VIEWING_ANGLE_ENABLED=True,
               VIEWING_ANGLE_DEG=110.0, VIEWING_HALF_ANGLE_RAD=math.radians(55),
               LANDMARK_DROPOUT_ENABLED=False, LANDMARK_OBSTRUCTION_ENABLED=False,
               ODOMETRY_BIAS_ENABLED=False, MOTION_SLIP_ENABLED=False)
REPORT_KEYS = ("runs", "converged", "non_converged_symmetrical", "non_converged",
               "trial_steps", "converged_steps", "mean_steps_to_converge",
               "sd_steps_to_converge", "max_steps_to_converge", "p95_steps_to_converge",
               "p99_steps_to_converge", "failures", "start_mode", "stopped")


_REFERENCE_PROFILE_NAME = "9 ambiguous"


def profile_for_case(scenario):
    name = scenario["profile"]
    if name not in {profile["name"] for profile in sim.LANDMARK_SCENARIO_PROFILES}:
        recognition = "unique" if scenario["association"] == "Unique ID" else "ambiguous"
        if name in {"9+1clo", "9+1far"}:
            distance = "close" if name == "9+1clo" else "far"
            name = f"9 {recognition} +1 orientation {distance}"
        else:
            name += " " + recognition
    return sim.landmark_scenario_profile_by_name(name)


def configured(module, scenario):
    stack = ExitStack()
    profile = profile_for_case(scenario)
    stack.enter_context(patch(__name__ + "._REFERENCE_PROFILE_NAME", profile["name"]))
    if module is sim:
        stack.enter_context(patch.multiple(sim, **SETTINGS, **SENSORS,
                                          ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME=profile["name"]))
    else:
        names = CORE_STATE_NAMES
        stack.enter_context(patch.multiple(core, **deepcopy({name: getattr(core, name) for name in names})))
        landmarks = profile["landmarks"]
        core.init_learning_rates({"profile": "frozen regression", **{name: SETTINGS[setting] for name, setting in LEARNING_RATE_SETTINGS.items()}})
        settings = {name: value for name, value in SETTINGS.items()
                    if name != "DT" and name not in LEARNING_RATE_SETTINGS.values()}
        core.init_landmarks({"profile": profile["name"], "landmarks": profile["landmarks"]})
        for name, value in settings.items():
            setattr(core, name, value)
    return stack


def legacy_point_id(value):
    family = _REFERENCE_PROFILE_NAME.replace(" ambiguous", " unique")
    rows = sim.landmark_scenario_profile_by_name(family)["landmarks"]
    return {str(index): row["name"] for index, row in enumerate(rows)}.get(value, value)


def make_case(name, association="Gated nearest", algorithm="landmark only", profile="9",
              actual=(200, -100, 0.25), belief=(220, -115, 0.30), steps=600,
              moving=False, reject=False):
    scenario = dict(name=name, association=association, algorithm=algorithm,
                    profile=profile, max_steps=steps, initial_belief=sim.make_robot_gc_state(*belief))
    frames = []
    with configured(sim, scenario):
        for index in range(steps if moving else 1):
            t = (index + 1) * SETTINGS["DT"] if moving else 0.0
            state = sim.make_robot_gc_state(actual[0] + 30 * t, actual[1] - 10 * t,
                                            sim.wrap_rad(actual[2] + 0.15 * t)) if moving else sim.make_robot_gc_state(*actual)
            if moving:
                state["x"][1], state["y"][1], state["theta"][1] = 30.0, -10.0, 0.15
            observations, ground_truth = sim.simulation_sense_landmarks(state, sim.active_simulation_landmarks(), return_ground_truth=True)
            if algorithm == "motion only":
                observations = []
                ground_truth = {}
            if reject:
                observations = [{"obs_id": "outlier", "name": "corner", "x_rcs_cm": 100000.0, "y_rcs_cm": 100000.0}]
                ground_truth = {}
            frames.append(dict(actual_gc=state, landmark_observations_rcs=observations,
                               landmark_ground_truth=ground_truth,
                               motion_observation=sim.simulation_sense_motion(state),
                               repeat=1 if moving else steps))
    scenario["frames"] = frames
    return scenario


def named_observations(scenario, observations):
    """Adapt frozen legacy input fields; keep all numeric measurements unchanged."""
    result = []
    for obs in observations:
        name = obs.get("name", obs.get("type"))
        if "name" not in obs and scenario["association"] == "Unique ID":
            name = obs["landmark_id"]
        result.append({**{key: value for key, value in obs.items() if key not in {"type", "landmark_id", "name"}},
                       "name": name})
    return result


def legacy_observation_view(obs):
    """Project renamed fields back to the frozen report schema, without changing numbers."""
    family = _REFERENCE_PROFILE_NAME.replace(" ambiguous", " unique")
    unique = sim.landmark_scenario_profile_by_name(family)["landmarks"]
    gated = sim.landmark_scenario_profile_by_name(family.replace(" unique", " ambiguous"))["landmarks"]
    canonical = {a["name"]: b["name"] for a, b in zip(unique, gated)}
    result = {**{key: value for key, value in obs.items() if key not in {"name", "landmark_id"}},
              "type": canonical.get(obs["name"], obs["name"])}
    if "id" in result:
        result["id"] = legacy_point_id(result["id"])
    return result


def replay(module, scenario, public=False):
    belief = deepcopy(scenario["initial_belief"])
    trace, checkpoints, history = [], [], []
    associations = []
    associate = core.associate_landmark_observation

    def record_association(*args, **kwargs):
        result = associate(*args, **kwargs)
        # Translate only diagnostic wording for the unchanged historical baseline.
        if result["association_name"] == "direct":
            legacy_name = "Unique ID" if scenario["association"] == "Unique ID" else "Unique direct"
        else:
            legacy_name = "Gated nearest"
        associations.append(dict(deepcopy(result), candidate_id=legacy_point_id(result["candidate_id"]),
                                 association_name=legacy_name))
        return result

    with configured(module, scenario), patch.object(core, "associate_landmark_observation", side_effect=record_association):
        for frame in scenario["frames"]:
            for _ in range(frame["repeat"]):
                step = len(history) + 1
                associations.clear()
                if module is sim:
                    response = sim.simulation_step_from_observations(
                        belief, named_observations(scenario, frame["landmark_observations_rcs"]),
                        deepcopy(frame["motion_observation"]), scenario["algorithm"], SETTINGS["DT"],
                        timestamp="2026-10-27T00:00:00.025Z")
                    belief, diagnostics = response["belief_fcs_gc"], response["diagnostics"]
                elif public:
                    request = dict(belief_fcs_gc=belief, algorithm_name=scenario["algorithm"],
                                   step_dt_s=SETTINGS["DT"],
                                   timestamp=(datetime.datetime(2026, 10, 27, tzinfo=datetime.timezone.utc)
                                              + datetime.timedelta(seconds=step * SETTINGS["DT"])).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                                   landmark_observations_rcs=named_observations(scenario, frame["landmark_observations_rcs"]),
                                   motion_observation=deepcopy(frame["motion_observation"]))
                    original = deepcopy(request)
                    response = module.step_core(request)
                    assert request == original, "step_core mutated its request"
                    belief, diagnostics = response["belief_fcs_gc"], response["diagnostics"]
                else:
                    belief, diagnostics = module.inference_step_from_observations(
                        belief, named_observations(scenario, frame["landmark_observations_rcs"]),
                        deepcopy(frame["motion_observation"]), scenario["algorithm"], SETTINGS["DT"])
                # Report snapshots need simulator ground truth, separate from observations.
                robots = {"actual": deepcopy(frame["actual_gc"]), "belief": deepcopy(belief)}
                with configured(sim, scenario):
                    record = sim.snapshot_record(step, robots, diagnostics["y_obs"], diagnostics["y_hat"],
                                                 frame["motion_observation"], diagnostics["free_energy"],
                                                 sim.experiment_settings(scenario["algorithm"]))
                history.append(record)
                correct = sim.history_has_sequential_convergence(history[-5:])
                symmetric = sim.history_has_sequential_convergence(history[-5:], symmetrical=True)
                finished = correct or symmetric or step == scenario["max_steps"]
                if step in (1, 5, 20, 100, 300) or finished:
                    trace.append(dict(step=step, belief=deepcopy(belief), free_energy=diagnostics["free_energy"],
                                      motion_free_energy=diagnostics["motion_free_energy"],
                                      landmark_free_energy=diagnostics["landmark_free_energy"]))
                    # Ignore simulator-only IDs stripped by the public input boundary.
                    observed = {key: legacy_observation_view(obs) for key, obs in diagnostics["y_obs"].items()}
                    checkpoints.append(dict(step=step, associations=deepcopy(associations),
                                            y_obs=observed, y_hat={legacy_point_id(key): legacy_observation_view(obs) for key, obs in diagnostics["y_hat"].items()},
                                            motion_terms=deepcopy(diagnostics["motion_terms"]),
                                            export=sim.compact_tracking_export_row(record, step * SETTINGS["DT"])))
                if finished:
                    return dict(outcome="correct" if correct else "symmetric" if symmetric else "non-converged",
                                steps=step, trace=trace, checkpoints=checkpoints)


def report_replay(cases, unique=False):
    names = ("unique_near_9", "unique_symmetric_9", "gated_no_observations") if unique else ("gated_near_9", "gated_symmetric_9", "gated_no_observations")
    selected = [next(case for case in cases if case["name"] == name)
                for name in names]
    frames = {tuple(case["frames"][0]["actual_gc"][axis][0] for axis in ("x", "y", "theta")): case["frames"][0]
              for case in selected}

    def recorded_frame(actual):
        return frames[tuple(actual[axis][0] for axis in ("x", "y", "theta"))]

    def recorded_landmarks(actual, landmarks, return_ground_truth=False):
        frame = recorded_frame(actual)
        observations = named_observations(selected[0], frame["landmark_observations_rcs"])
        ground_truth = frame.get("landmark_ground_truth", {obs["obs_id"]: obs["landmark_id"]
                                 for obs in frame["landmark_observations_rcs"] if "landmark_id" in obs})
        return (observations, ground_truth) if return_ground_truth else observations

    starts = [dict(actual_start={axis: case["frames"][0]["actual_gc"][axis][0] for axis in ("x", "y", "theta")},
                   belief_start={axis: case["initial_belief"][axis][0] for axis in ("x", "y", "theta")})
              for case in selected]
    with configured(sim, selected[0]), patch.object(sim, "INFER_STEPS_MAX", 600), \
         patch.object(sim, "simulation_sense_landmarks", side_effect=recorded_landmarks), \
         patch.object(sim, "simulation_sense_motion", side_effect=lambda actual: deepcopy(recorded_frame(actual)["motion_observation"])):
        report = sim.run_converge_test({"actual": sim.make_robot_gc_state(), "belief": sim.make_robot_gc_state()},
                                       "landmark only", trial_specs=starts)
    return {key: report[key] for key in REPORT_KEYS}


def capture_baseline():
    cases = [
        make_case("gated_near_9"),
        make_case("unique_near_9", association="Unique ID"),
        make_case("gated_symmetric_9", belief=(-200, 100, sim.wrap_rad(0.25 + math.pi))),
        make_case("unique_symmetric_9", association="Unique ID", belief=(-200, 100, sim.wrap_rad(0.25 + math.pi))),
        make_case("gated_profile_1", profile="1"),
        make_case("unique_profile_2", association="Unique ID", profile="2"),
        make_case("gated_profile_3", profile="3"),
        make_case("unique_profile_5", association="Unique ID", profile="5"),
        make_case("gated_near_poster", profile="9+1clo", belief=(-200, 100, sim.wrap_rad(0.25 + math.pi))),
        make_case("gated_far_poster", profile="9+1far"),
        make_case("motion_only_moving", algorithm="motion only", moving=True, steps=60),
        make_case("fused_moving", algorithm="motion+landmark", moving=True, steps=60),
        make_case("fused_unique_moving", algorithm="motion+landmark", association="Unique ID", moving=True, steps=60),
        make_case("gated_no_observations", actual=(1120, 800, 0), belief=(0, 0, 0)),
        make_case("gated_one_corner", actual=(-953, 30, 1.654), belief=(-1000, 100, 1.571)),
        make_case("gated_rejected_outlier", reject=True, steps=20),
    ]
    for case in cases:
        case["expected"] = replay(sim, case)
    data = dict(reference_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("brain_inspired_localization_simulator.py", "brain_inspired_localization_core.py")},
                settings=SETTINGS, sensors=SENSORS, scenarios=cases, expected_report=report_replay(cases))
    BASELINE.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for case in cases:
        print(case["name"], case["expected"]["outcome"], case["expected"]["steps"])


class LandmarkMapTests(unittest.TestCase):
    def test_profiles_are_self_contained_json_records(self):
        profiles = json.loads(json.dumps(sim.LANDMARK_SCENARIO_PROFILES))
        self.assertEqual(profiles, sim.LANDMARK_SCENARIO_PROFILES)
        self.assertEqual(len({profile["name"] for profile in profiles}), len(profiles))
        for profile in profiles:
            self.assertEqual(set(profile), {"name", "landmarks"})
            for row in profile["landmarks"]:
                self.assertEqual(set(row), {"name", "type", "x_cm", "y_cm"})
                self.assertIn(row["type"], {"pose", "orientation"})
        self.assertFalse(hasattr(sim, "LEGACY_LANDMARK_IDS"))
        self.assertFalse(hasattr(sim, "LANDMARK_MAP"))
        self.assertFalse(hasattr(sim, "ACTIVE_DATA_ASSOCIATION_NAME"))
        self.assertFalse(hasattr(core, "LANDMARK_SCENARIO_PROFILES"))

    def test_unique_and_gated_profiles_hold_their_own_names(self):
        unique = sim.landmark_scenario_profile_by_name("9 unique")
        gated = sim.landmark_scenario_profile_by_name("9 ambiguous")
        self.assertEqual(len(unique["landmarks"]), 9)
        self.assertEqual(len({row["name"] for row in unique["landmarks"]}), 9)
        self.assertEqual(sum(row["name"] == "corner" for row in gated["landmarks"]), 4)
        self.assertEqual(sim.landmark_profile_association_name(unique), "Unique names")
        self.assertEqual(sim.landmark_profile_association_name(gated), "Repeated names")
        # The records determine uniqueness, even when the profile name says otherwise.
        self.assertEqual(sim.landmark_profile_association_name(
            sim.landmark_scenario_profile_by_name("1 ambiguous")), "Unique names")
        self.assertEqual([(r["x_cm"], r["y_cm"]) for r in unique["landmarks"]],
                         [(r["x_cm"], r["y_cm"]) for r in gated["landmarks"]])

    def test_selected_profile_is_exact_core_map(self):
        with configured(sim, dict(association="Gated nearest", profile="2")):
            sim.ensure_simulation_core_configuration()
            self.assertEqual(core.LANDMARK_MAP, sim.active_landmark_scenario_profile()["landmarks"])
            self.assertEqual(len(core.LANDMARK_MAP), 2)
            self.assertEqual(core.landmark_candidate_ids_for_name("goal"), ["0", "1"])

    def test_association_shortcut_selects_an_existing_profile(self):
        with configured(sim, dict(association="Gated nearest", profile="2")):
            sim.select_association_profile("Unique names")
            self.assertEqual(sim.ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME, "2 unique")
            self.assertEqual(sim.active_association_name(), "Unique names")
            self.assertEqual([row["name"] for row in sim.active_landmark_scenario_profile()["landmarks"]],
                             ["goal_opponent", "goal_own"])

    def test_single_custom_profile_has_no_master_landmark_dependencies(self):
        profile = {"name": "my-two-beacons", "landmarks": [
            dict(name="beacon", type="pose", x_cm=100, y_cm=50),
            dict(name="beacon", type="pose", x_cm=200, y_cm=-50),
        ]}
        with patch.object(sim, "LANDMARK_SCENARIO_PROFILES", [profile]), \
             patch.multiple(sim, **SETTINGS, **SENSORS, ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME=profile["name"]):
            robots = {"actual": sim.make_robot_gc_state(), "belief": sim.make_robot_gc_state(5, 5, 0.1)}
            observed, predicted, _, _ = sim.simulation_inference_step(robots)
            self.assertEqual(core.LANDMARK_MAP, profile["landmarks"])
            self.assertEqual(len(observed), 2)
            self.assertEqual(len(predicted), 2)
            self.assertEqual({row["name"] for row in observed.values()}, {"beacon"})
            self.assertEqual(sim.configured_association_names(), ["Repeated names"])

    def test_profiles_do_not_require_an_equivalent_other_mode(self):
        profiles = [sim.landmark_scenario_profile_by_name("9 unique"),
                    sim.landmark_scenario_profile_by_name("2 ambiguous")]
        with patch.object(sim, "LANDMARK_SCENARIO_PROFILES", profiles), \
             patch.object(sim, "ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME", "9 unique"):
            sim.select_association_profile("Repeated names")
            self.assertEqual(sim.active_landmark_scenario_profile()["name"], "2 ambiguous")

    def test_profile_batch_export_keeps_complete_profile_snapshots(self):
        import csv
        import tempfile
        for mode in ("Gated nearest", "Unique ID"):
            with self.subTest(mode=mode), configured(sim, dict(profile="9", association=mode)), \
                 tempfile.TemporaryDirectory(prefix="landmark-profile-export-") as directory, \
                 patch("os.getcwd", return_value=directory):
                settings = sim.experiment_settings("motion+landmark")
                settings["demo_path_name"] = "Stationary test"
                setup = dict(robots={"actual": sim.make_robot_gc_state(200, -100, 0.25),
                                     "belief": sim.make_robot_gc_state(220, -115, 0.3)},
                             settings=settings, runs=1, algorithm_name="motion+landmark",
                             ball={"x_cm": 0, "y_cm": 0}, demo_state={}, path_random_state=None)
                result = sim.export_paper_lm_dataset(setup, lambda experiment: None, duration_s=0.05)
                metadata = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
                self.assertEqual(len(metadata["landmark_profiles"]), 6 if mode == "Unique ID" else 5)
                self.assertTrue(all(sim.landmark_profile_association_name(profile) == ("Unique names" if mode == "Unique ID" else "Repeated names") and "landmarks" in profile
                                    for profile in metadata["landmark_profiles"]))
                with open(result["csv_path"], newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 3 * len(metadata["landmark_profiles"]))
                expected_names = {profile["name"] for profile in metadata["landmark_profiles"]}
                self.assertEqual({row["landmark_profile"] for row in rows}, expected_names)
                curves = sim.paper_lm_mean_errors(result["csv_path"], metadata)
                self.assertEqual(len(curves), 3)
                self.assertTrue(all(len(curve["time_s"]) == 3 for curve in curves.values()))


class LocalizationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(BASELINE.read_text(encoding="utf-8"))

    def assert_close(self, actual, expected, path="result"):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.assert_close(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), path)
            for index, value in enumerate(expected):
                self.assert_close(actual[index], value, f"{path}[{index}]")
        elif isinstance(expected, float):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
                            f"{path}: {actual!r} != {expected!r}")
        else:
            self.assertEqual(actual, expected, path)

    def assert_scenario_result(self, case, result):
        position_only_references = {"gated_near_poster": "gated_symmetric_9", "gated_far_poster": "gated_near_9"}
        if case["name"] in position_only_references:
            # Intentional step-4 change: posters no longer contribute to tracking.
            # Compare numbers to the already frozen equivalent nine-landmark case.
            reference = next(item for item in self.data["scenarios"]
                             if item["name"] == position_only_references[case["name"]])["expected"]
            for key in ("outcome", "steps", "trace"):
                self.assert_close(result[key], reference[key], f"{case['name']}.{key}")
        else:
            expected = deepcopy(case["expected"])
            family = profile_for_case(case)["name"].replace(" ambiguous", " unique")
            labels = {row["name"] for row in sim.landmark_scenario_profile_by_name(family)["landmarks"]}
            for checkpoint in expected["checkpoints"]:
                checkpoint["y_hat"] = {key: row for key, row in checkpoint["y_hat"].items() if key in labels}
            self.assert_close(result, expected)

    def test_saved_observations_match_current_simulator(self):
        self.assertEqual(self.data["settings"], SETTINGS)
        self.assertEqual(self.data["sensors"], SENSORS)
        for case in self.data["scenarios"]:
            if case["name"] == "gated_rejected_outlier":
                continue  # Deliberately synthetic outlier, not a sensor sample.
            with self.subTest(case=case["name"]), configured(sim, case):
                for frame in case["frames"]:
                    observations = sim.simulation_sense_landmarks(frame["actual_gc"], sim.active_simulation_landmarks())
                    if case["algorithm"] == "motion only":
                        observations = []
                    self.assertEqual(len(observations), len(frame["landmark_observations_rcs"]))
                    observations = [dict(obs, obs_id=old["obs_id"]) for obs, old in zip(observations, frame["landmark_observations_rcs"])]
                    self.assert_close(observations, named_observations(case, frame["landmark_observations_rcs"]))
                    self.assert_close(sim.simulation_sense_motion(frame["actual_gc"]), frame["motion_observation"])

    def test_simulator_updates_associations_and_report_rows(self):
        for case in self.data["scenarios"]:
            with self.subTest(case=case["name"]):
                self.assert_scenario_result(case, replay(sim, case))

    def test_standalone_core_matches_reference(self):
        for case in self.data["scenarios"]:
            with self.subTest(case=case["name"]):
                self.assert_scenario_result(case, replay(core, case))

    def test_public_step_core_matches_reference_including_unique_names(self):
        for case in self.data["scenarios"]:
            with self.subTest(case=case["name"]):
                self.assert_scenario_result(case, replay(core, case, public=True))

    def test_convergence_report_counts_steps_and_failures(self):
        self.assert_close(report_replay(self.data["scenarios"]), self.data["expected_report"])
        report = self.data["expected_report"]
        self.assertEqual((report["converged"], report["non_converged_symmetrical"], report["non_converged"]), (1, 1, 1))

    def test_unique_name_report_preserves_perfect_association_results(self):
        report = report_replay(self.data["scenarios"], unique=True)
        self.assertEqual((report["converged"], report["non_converged_symmetrical"], report["non_converged"]), (2, 0, 1))
        # Existing frozen Unique ID cases converge in 173 and 461 steps.
        self.assertEqual(report["trial_steps"], [173, 461, 600])
        self.assertEqual(report["converged_steps"], [173, 461])
        self.assertEqual(report["mean_steps_to_converge"], 317)
        self.assertEqual(report["sd_steps_to_converge"], 144)
        self.assertEqual(report["p95_steps_to_converge"], 461)
        self.assertEqual(report["p99_steps_to_converge"], 461)
        self.assert_close(report["failures"], self.data["expected_report"]["failures"][-1:])


class OrientationTrackingTests(unittest.TestCase):
    def test_orientation_observations_do_not_change_state_or_free_energy(self):
        for module in (core,):
            for mode in ("Gated nearest", "Unique ID"):
                with self.subTest(module=module.__name__, mode=mode), configured(module, dict(association=mode, profile="9+1clo")):
                    actual = module.make_robot_gc_state(200, -100, 0.25)
                    belief = module.make_robot_gc_state(220, -115, 0.3)
                    positions, orientations = [], []
                    for prediction in module.landmarks_fcs_2_rcs(actual, module.LANDMARKS):
                        obs = {key: value for key, value in prediction.items() if key not in {"id", "angle_rad"}}
                        obs["obs_id"] = prediction["id"]
                        target = positions if module.LANDMARK_METADATA[prediction["id"]]["type"] == "pose" else orientations
                        target.append(obs)

                    def step(observations):
                        return module.step_core(dict(timestamp="2026-10-27T00:00:00.025Z",
                                                     belief_fcs_gc=belief, landmark_observations_rcs=observations,
                                                     motion_observation=None, algorithm_name="landmark only", step_dt_s=0.025))

                    orientation_only = step(orientations)
                    self.assertEqual(orientation_only["belief_fcs_gc"], step([])["belief_fcs_gc"])
                    self.assertEqual(orientation_only["diagnostics"]["free_energy"], 0)
                    position_only, mixed = step(positions), step(positions + orientations)
                    self.assertEqual(mixed["belief_fcs_gc"], position_only["belief_fcs_gc"])
                    self.assertEqual(mixed["diagnostics"]["free_energy"], position_only["diagnostics"]["free_energy"])

    def test_type_controls_filtering_instead_of_poster_names(self):
        for module in (core,):
            landmarks = deepcopy(sim.landmark_scenario_profile_by_name("9 ambiguous")["landmarks"])
            landmarks[0]["type"] = "orientation"  # Center is normally a pose landmark.
            with self.subTest(module=module.__name__), configured(module, dict(association="Gated nearest", profile="9")):
                module.init_landmarks({"profile": "test-type", "landmarks": landmarks})
                belief = module.make_robot_gc_state(200, -100, 0.25)
                original = deepcopy(belief)
                observations = module.calculate_angle({"center_obs": {"name": "center", "x_rcs_cm": 50, "y_rcs_cm": 20}})
                _, energy = module.landmark_inference_step(belief, observations)
                self.assertEqual(belief, original)
                self.assertEqual(energy, 0)

    def test_one_step_analysis_excludes_orientation_contributions(self):
        for mode in ("Gated nearest", "Unique ID"):
            with self.subTest(mode=mode), configured(sim, dict(association=mode, profile="9+1clo")):
                actual, belief = sim.make_robot_gc_state(200, -100, 0.25), sim.make_robot_gc_state(220, -115, 0.3)
                analysis = sim.landmark_analysis_data(actual, belief)
                self.assertTrue(any(obs["name"] == "poster1" for obs in analysis["y_obs"].values()))
                self.assertTrue(analysis["details"])
                self.assertTrue(all(sim.active_landmark_scenario_profile()["landmarks"][int(detail["id"])]["type"] == "pose"
                                    for detail in analysis["details"]))
                diagnostics = sim.simulation_step_from_observations(
                    belief, list(analysis["y_obs"].values()), None, "landmark only", sim.DT)["diagnostics"]
                self.assertAlmostEqual(sum(detail["precision"] * (detail["ex"] ** 2 + detail["ey"] ** 2
                                                                  + (sim.ANGLE_WEIGHT * detail["ea"]) ** 2)
                                           for detail in analysis["details"]), diagnostics["free_energy"])


class RecognitionNameTests(unittest.TestCase):
    def test_association_uses_names_without_reading_simulator_identity(self):
        class NamedObservation(dict):
            def __getitem__(self, key):
                if key in {"landmark_id", "type"}:
                    raise AssertionError(f"Association read legacy field {key}")
                return super().__getitem__(key)

            def get(self, key, *args):
                if key in {"landmark_id", "type"}:
                    raise AssertionError(f"Association read legacy field {key}")
                return super().get(key, *args)

        for module in (core,):
            for mode in ("Gated nearest", "Unique ID"):
                scenario = dict(association=mode, profile="9")
                with self.subTest(module=module.__name__, mode=mode), configured(module, scenario):
                    state = module.make_robot_gc_state(200, -100, 0.25)
                    predictions = module.obs_list_to_map(module.landmarks_fcs_2_rcs(state, module.LANDMARKS))
                    obs = NamedObservation({key: value for key, value in predictions["3"].items() if key != "id"})
                    result = module.associate_landmark_observation(obs, predictions)
                    self.assertTrue(result["accepted"])
                    self.assertEqual(result["candidate_id"], "3")
                    self.assertEqual(result["association_name"], "direct" if mode == "Unique ID" else "gated nearest")
                    self.assertFalse(module.associate_landmark_observation(
                        obs, predictions, {"3"})["accepted"])
                    unknown = NamedObservation(obs, name="not_in_map")
                    self.assertFalse(module.associate_landmark_observation(unknown, predictions)["accepted"])

    def test_simulator_passes_only_names_and_keeps_report_identity(self):
        for mode in ("Gated nearest", "Unique ID"):
            with self.subTest(mode=mode), configured(sim, dict(association=mode, profile="9+1clo")):
                robots = {"actual": sim.make_robot_gc_state(200, -100, 0.25),
                          "belief": sim.make_robot_gc_state(-200, 100, sim.wrap_rad(0.25 + math.pi))}
                with patch.object(core, "step_core", wraps=core.step_core) as inference:
                    observed, _, _, _ = sim.simulation_inference_step(robots)
                supplied = inference.call_args.args[0]["landmark_observations_rcs"]
                for obs in supplied:
                    self.assertIn("name", obs)
                    self.assertNotIn("landmark_id", obs)
                    self.assertNotIn("type", obs)
                    if mode == "Unique ID":
                        self.assertIn(obs["name"], {row["name"] for row in sim.active_landmark_scenario_profile()["landmarks"]})
                self.assertTrue(all("landmark_id" in obs for obs in observed.values()))
                analysis = sim.landmark_analysis_data(robots["actual"], robots["belief"])
                self.assertTrue(analysis["details"])
                if mode == "Unique ID":
                    self.assertTrue(all(detail["actual_landmark_id"] == detail["associated_landmark_id"]
                                        for detail in analysis["details"]))

    def test_gui_sets_profile_and_settings_without_reselecting_each_frame(self):
        with configured(sim, dict(association="Gated nearest", profile="9")):
            profile = sim.landmark_scenario_profile_by_name("2 ambiguous")
            core.init_landmarks({"profile": profile["name"], "landmarks": profile["landmarks"]})
            core.init_learning_rates({"profile": "zero rates", **{name: 0 for name in LEARNING_RATE_SETTINGS}})
            robots = {"actual": sim.make_robot_gc_state(200, -100, 0.25),
                      "belief": sim.make_robot_gc_state(220, -115, 0.3)}
            with patch.object(core, "init_landmarks", wraps=core.init_landmarks) as select, \
                 patch.object(core, "init_learning_rates", wraps=core.init_learning_rates) as rates:
                sim.simulation_inference_step(robots)
                sim.simulation_inference_step(robots)
                self.assertEqual(select.call_count, 1)
                self.assertEqual(rates.call_count, 1)
                with patch.object(sim, "ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME", "1 ambiguous"):
                    sim.simulation_inference_step(robots)
                    self.assertEqual(select.call_count, 2)
                    self.assertEqual(len(core.LANDMARK_MAP), 1)
                    self.assertEqual(rates.call_count, 1)
                with patch.object(sim, "LR_XY", 1.7):
                    sim.simulation_inference_step(robots)
                    self.assertEqual(select.call_count, 3)
                    self.assertEqual(core.LEARNING_RATES["landmark_xy"], 1.7)
                    self.assertEqual(rates.call_count, 2)

    def test_one_step_analysis_uses_one_core_call_without_mutating_robots(self):
        for mode in ("Gated nearest", "Unique ID"):
            with self.subTest(mode=mode), configured(sim, dict(association=mode, profile="9+1clo")):
                robots = {"actual": sim.make_robot_gc_state(200, -100, 0.25),
                          "belief": sim.make_robot_gc_state(220, -115, 0.3)}
                robots["actual"]["x"][1], robots["actual"]["theta"][1] = 30, 0.15
                original_robots = deepcopy(robots)
                responses = []
                step = core.step_core

                def record(request):
                    response = step(request)
                    responses.append(deepcopy(response))
                    return response

                with patch.object(core, "step_core", side_effect=record) as called:
                    analysis = sim.one_step_analysis(robots, sim.FUSED_MOTION_LANDMARK_ALGORITHM_NAME)
                self.assertEqual(called.call_count, 1)
                self.assertEqual(robots, original_robots)
                used = {term["obs_id"]: term for term in responses[0]["diagnostics"]["landmark_terms"] if term["used_for_update"]}
                self.assertEqual(len(analysis["details"]), len(used))
                for detail in analysis["details"]:
                    self.assertEqual(detail["contrib"], used[detail["obs_id"]]["jt_p_eps"])
                for detail in analysis["motion_details"]:
                    self.assertEqual(detail["vel_error"], responses[0]["diagnostics"]["motion_terms"][detail["id"]]["vel_error"])

    def test_gui_has_no_copied_core_function_definitions(self):
        import ast
        gui = ast.parse((ROOT / "brain_inspired_localization_simulator.py").read_text(encoding="utf-8"))
        algorithm = ast.parse(Path(core.__file__).read_text(encoding="utf-8"))
        names = lambda tree: {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(names(gui) & names(algorithm), set())


if __name__ == "__main__":
    if sys.argv[1:] == ["--capture-baseline"]:
        capture_baseline()
    else:
        unittest.main()
