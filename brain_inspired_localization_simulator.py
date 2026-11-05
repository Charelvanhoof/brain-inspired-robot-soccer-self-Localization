# Code for the Robot_Soccer_Self_Localization_with_Active_Inference
# Based on version POC V29
import json
import datetime
import brain_inspired_localization_core as core
import math
from pathlib import Path
from copy import deepcopy
import random
import sys
import textwrap
import types

from brain_inspired_localization_core import (
    wrap_rad, make_robot_gc_state, scalar_2_gc, gc_2_scalar, obs_list_to_map,
    landmark_fcs_2_rcs, vel_fcs_2_rcs, acc_fcs_2_rcs, vel_rcs_2_fcs, acc_rcs_2_fcs,
    landmark_is_within_viewing_angle, calculate_angle, algorithm_uses_motion, algorithm_uses_landmarks,
)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib_stub = types.ModuleType("matplotlib")
    plt = types.ModuleType("matplotlib.pyplot")
    plt._is_stub = True

    def _missing_matplotlib(*args, **kwargs):
        raise ModuleNotFoundError("matplotlib is required for plotting and UI mode")

    plt.pause = lambda *args, **kwargs: None
    plt.__getattr__ = lambda name: _missing_matplotlib
    matplotlib_stub.pyplot = plt
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules["matplotlib.pyplot"] = plt

try:
    import pygame
except ModuleNotFoundError:
    pygame = types.ModuleType("pygame")
    pygame._is_stub = True

    def _missing_pygame(*args, **kwargs):
        raise ModuleNotFoundError("pygame is required for interactive UI mode")

    pygame.__getattr__ = lambda name: _missing_pygame
    sys.modules["pygame"] = pygame

# =========================
# SIMULATOR / UI CONFIGURATION
# =========================
# --- Field drawing in GUI and simulation settings (eg starting pose) (cm) ---
# Used to draw the field, generate test starting poses, and bound demo/UI movement.
# These dimensions do not configure CORE's landmark map; each core profile lists
FIELD_LENGTH_CM = 1800
FIELD_WIDTH_CM = 1200
OUTSIDE_MARGIN_CM = 250
GOAL_WIDTH_CM = 240 
GOAL_DEPTH_CM = 80  
CENTER_CIRCLE_RADIUS_CM = 200
GOAL_AREA_WIDTH_CM = 390
GOAL_AREA_DEPTH_CM = 75
PENALTY_AREA_WIDTH_CM = 600
PENALTY_AREA_DEPTH_CM = 225

# --- Display and interaction settings (px unless indicated otherwise) ---
FIELD_MARKING_WIDTH_PX = 2

PANEL_W_PX = 400
MARGIN_PX = 24
ROBOT_RADIUS_PX = 8
BALL_RADIUS_PX = 6
HEADING_LEN_PX = 30
ROTATE_STEP_DEG = 3.0
ROTATE_STEP_RAD = math.radians(ROTATE_STEP_DEG)
ROUND_CM = True
MAX_SENSE_LINES = 12

# =========================
# INFERENCE / EXPERIMENT SETTINGS
# =========================
DT = 0.020
PI_CAM = 1.0
PI_ODO = 1.0
PI_IMU = 1.0
INFER_STEPS_SHORT = 100
INFER_STEPS_LONG = 2000
INFER_STEPS_MAX = 5000
PAPER_VISIBILITY_STRESS_DURATION_S = 2.0 * 60.0 * 60.0
PAPER_LM_TRIAL_DURATION_S = 30.0
CONVERGE_TEST_RUNS = 100
CONVERGE_TEST_USE_REGRESSION_SET = False
# FCS starting-belief limits for S-NEAR / D-NEAR and future fPI evaluation.
NEAR_START_LIMITS = {"x_cm": 250.0, "y_cm": 250.0, "angle_deg": 30.0}
CONV_TOL_X_CM = 3.0
CONV_TOL_Y_CM = 3.0
CONV_TOL_ANGLE_DEG = 3.0
CONV_TOL_ANGLE_RAD = math.radians(CONV_TOL_ANGLE_DEG)
CONV_REQUIRED_SEQUENTIAL_HITS = 5
DEFAULT_ANGLE_LR = 2.20325e-05
LR_XY = 2.0
LR_ANGLE = DEFAULT_ANGLE_LR
ANGLE_WEIGHT = 1.0
LANDMARK_ONLY_ALGORITHM_NAME = "landmark only"
MOTION_ONLY_ALGORITHM_NAME = "motion only"
FUSED_MOTION_LANDMARK_ALGORITHM_NAME = "motion+landmark"
ACTIVE_ALGORITHM_NAME = FUSED_MOTION_LANDMARK_ALGORITHM_NAME
DATA_ASSOCIATION_GATED_NEAREST_NAME = "Gated nearest"
DATA_ASSOCIATION_UNIQUE_ID_NAME = "Unique ID"
LANDMARK_ASSOCIATION_GATE_CM = 1000.0
DISTANCE_PRECISION_REF_CM = 600.0
DISTANCE_PRECISION_MIN_FACTOR = 0.15
PRECISION_DISTANCE_MODIFIER_ENABLED = True
REALTIME_UI_FPS = 60
MAX_REALTIME_SIM_STEPS_PER_FRAME = 120
DEMO_TRACE_MAX_POINTS = 1200
DEMO_TURN_MAX_SPEED_RAD_S = 2.6
DEMO_TURN_MAX_ACCEL_RAD_S2 = 7.5
DEMO_TURN_NEAR_ZERO_ERROR_RAD = math.radians(5.0)
DEMO_TURN_NEAR_ZERO_ACCEL_RAD_S2 = 0.6
DEFAULT_VIEWING_ANGLE_DEG = 110.0
MIN_VIEWING_ANGLE_DEG = 0.0
MAX_VIEWING_ANGLE_DEG = 360.0
VIEWING_ANGLE_ENABLED = False
VIEWING_ANGLE_DEG = DEFAULT_VIEWING_ANGLE_DEG
VIEWING_HALF_ANGLE_RAD = math.radians(VIEWING_ANGLE_DEG / 2.0)
MOTION_LR_VEL_XY = 5.0
MOTION_LR_ACC_XY = 50.0
MOTION_LR_VEL_THETA = 5.0
MOTION_LR_ACC_THETA = 50.0
MASTERDATA_SETTING_PROFILES = [
    {
        "name": "40Hz",
        "precision_distance_modifier_enabled": True,
        "dt": 0.025,
        "pi_cam": 1.0,
        "pi_odo": 1.0,
        "pi_imu": 1.0,
        "lr_xy": 1.6,
        "lr_angle": 1.700e-07,
        "motion_lr_vel_xy": 4.0,
        "motion_lr_vel_theta": 4.0,
        "motion_lr_acc_xy": 40.0,
        "motion_lr_acc_theta": 40.0,
    },
    {
        "name": "40Hz",
        "precision_distance_modifier_enabled": False,
        "dt": 0.025,
        "pi_cam": 1.0,
        "pi_odo": 1.0,
        "pi_imu": 1.0,
        "lr_xy": 1.6,
        "lr_angle": 1.700e-07,
        "motion_lr_vel_xy": 4.0,
        "motion_lr_vel_theta": 4.0,
        "motion_lr_acc_xy": 40.0,
        "motion_lr_acc_theta": 40.0,
    },
    {
        "name": "200Hz",
        "precision_distance_modifier_enabled": True,
        "dt": 0.005,
        "pi_cam": 1.0,
        "pi_odo": 1.0,
        "pi_imu": 1.0,
        "lr_xy": 10.0,
        "lr_angle": 8.800e-07,
        "motion_lr_vel_xy": 6.0,
        "motion_lr_vel_theta": 5.0,
        "motion_lr_acc_xy": 4.0,
        "motion_lr_acc_theta": 3.0,
    },
    {
        "name": "200Hz",
        "precision_distance_modifier_enabled": False,
        "dt": 0.005,
        "pi_cam": 1.0,
        "pi_odo": 1.0,
        "pi_imu": 1.0,
        "lr_xy": 10.0,
        "lr_angle": 8.800e-07,
        "motion_lr_vel_xy": 6.0,
        "motion_lr_vel_theta": 5.0,
        "motion_lr_acc_xy": 4.0,
        "motion_lr_acc_theta": 3.0,
    },
]
MASTERDATA_FREQUENCY_NAMES = ["40Hz", "200Hz"]
MOTION_NOISE_RATIO = 0.10
# NOISE SETTINGS (shared Gaussian noise magnitude)
NOISE_STD_X_CM = 10.0
NOISE_STD_Y_CM = 10.0
NOISE_STD_ANGLE_DEG = 10.0
NOISE_ENABLED = False
ODOMETRY_BIAS_ENABLED = False
ODOMETRY_BIAS_X_CM_S = 18.0
ODOMETRY_BIAS_Y_CM_S = 8.0
ODOMETRY_BIAS_THETA_RAD_S = math.radians(1.5)
MOTION_SLIP_ENABLED = False
MOTION_SLIP_EVENT_PROBABILITY = 0.03
MOTION_SLIP_EVENT_INTERVAL_S = 1.0
MOTION_SLIP_MEAN_S = 2.0
MOTION_SLIP_STD_S = 0.5
MOTION_SLIP_VELOCITY_FACTOR = 1.5
MOTION_SLIP_ROTATION_LEVER_CM = 45.0
MOTION_SLIP_REMAINING_S = 0.0
MOTION_SLIP_AXIS = "x"
MOTION_SLIP_TURN_SIGN = 1.0
LANDMARK_DROPOUT_ENABLED = False
LANDMARK_DROPOUT_PROBABILITY = 0.10
LANDMARK_OBSTRUCTION_ENABLED = False
LANDMARK_OBSTRUCTION_EVENT_PROBABILITY = 0.03
LANDMARK_OBSTRUCTION_EVENT_INTERVAL_S = 1.0
LANDMARK_OBSTRUCTION_MEAN_S = 2.0
LANDMARK_OBSTRUCTION_STD_S = 0.5
LANDMARK_OBSTRUCTION_REMAINING_S = {}

# =========================
# MASTERDATA: Landmarks in FCS (cm)
# =========================
# Each profile contains its complete map and recognition names.
LANDMARK_SCENARIO_PROFILES = [{'name': '9 ambiguous',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 600.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': 900.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': -900.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 600.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0}]},
 {'name': '9 ambiguous +1 orientation close',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 600.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': 900.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': -900.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 600.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0},
                {'name': 'poster1', 'type': 'orientation', 'x_cm': 1200.0, 'y_cm': 300.0}]},
 {'name': '9 ambiguous +1 orientation far',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 600.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': 900.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': -900.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 600.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0},
                {'name': 'poster2', 'type': 'orientation', 'x_cm': 4900.0, 'y_cm': 300.0}]},
 {'name': '2 ambiguous',
  'landmarks': [{'name': 'goal', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0}]},
 {'name': '1 ambiguous',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0}]},
 {'name': '3 ambiguous',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0}]},
 {'name': '5 ambiguous',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 600.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': -600.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0}]},
 {'name': '9 unique',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'left_middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 600.0},
                {'name': 'right_middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': -600.0},
                {'name': 'corner_TR', 'type': 'pose', 'x_cm': 900.0, 'y_cm': -600.0},
                {'name': 'corner_TL', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 600.0},
                {'name': 'corner_BR', 'type': 'pose', 'x_cm': -900.0, 'y_cm': -600.0},
                {'name': 'corner_BL', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 600.0},
                {'name': 'goal_opponent', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal_own', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0}]},
 {'name': '9 unique +1 orientation close',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'left_middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 600.0},
                {'name': 'right_middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': -600.0},
                {'name': 'corner_TR', 'type': 'pose', 'x_cm': 900.0, 'y_cm': -600.0},
                {'name': 'corner_TL', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 600.0},
                {'name': 'corner_BR', 'type': 'pose', 'x_cm': -900.0, 'y_cm': -600.0},
                {'name': 'corner_BL', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 600.0},
                {'name': 'goal_opponent', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal_own', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0},
                {'name': 'poster1', 'type': 'orientation', 'x_cm': 1200.0, 'y_cm': 300.0}]},
 {'name': '9 unique +1 orientation far',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'left_middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 600.0},
                {'name': 'right_middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': -600.0},
                {'name': 'corner_TR', 'type': 'pose', 'x_cm': 900.0, 'y_cm': -600.0},
                {'name': 'corner_TL', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 600.0},
                {'name': 'corner_BR', 'type': 'pose', 'x_cm': -900.0, 'y_cm': -600.0},
                {'name': 'corner_BL', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 600.0},
                {'name': 'goal_opponent', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal_own', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0},
                {'name': 'poster2', 'type': 'orientation', 'x_cm': 4900.0, 'y_cm': 300.0}]},
 {'name': '2 unique',
  'landmarks': [{'name': 'goal_opponent', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal_own', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0}]},
 {'name': '1 unique',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0}]},
 {'name': '3 unique',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'goal_opponent', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal_own', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0}]},
 {'name': '5 unique',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'left_middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 600.0},
                {'name': 'right_middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': -600.0},
                {'name': 'goal_opponent', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal_own', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0}]},
 {'name': 'testRobotsports',
  'landmarks': [{'name': 'center', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 0.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': 600.0},
                {'name': 'middle', 'type': 'pose', 'x_cm': 0.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': 900.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': -900.0, 'y_cm': -600.0},
                {'name': 'corner', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 600.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 0.0},
                {'name': 'goal', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 0.0},
                {'name': 'goalpost', 'type': 'pose', 'x_cm': 900.0, 'y_cm': -120.0},
                {'name': 'goalpost', 'type': 'pose', 'x_cm': 900.0, 'y_cm': 120.0},
                {'name': 'goalpost', 'type': 'pose', 'x_cm': -900.0, 'y_cm': -120.0},
                {'name': 'goalpost', 'type': 'pose', 'x_cm': -900.0, 'y_cm': 120.0},
                {'name': 'penaltyintersection', 'type': 'pose', 'x_cm': 675.0, 'y_cm': -600.0},
                {'name': 'penaltyintersection', 'type': 'pose', 'x_cm': 675.0, 'y_cm': 600.0},
                {'name': 'penaltyintersection', 'type': 'pose', 'x_cm': -675.0, 'y_cm': -600.0},
                {'name': 'penaltyintersection', 'type': 'pose', 'x_cm': -675.0, 'y_cm': 600.0}]}]
ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME = LANDMARK_SCENARIO_PROFILES[0]["name"]


def landmark_scenario_profile_by_name(profile_name):
    for profile in LANDMARK_SCENARIO_PROFILES:
        if profile["name"] == profile_name:
            return profile
    raise ValueError(f"Unknown landmark profile: {profile_name}")


def active_landmark_scenario_profile():
    return landmark_scenario_profile_by_name(ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME)


def active_association_name():
    return core.landmark_profile_association_name(active_landmark_scenario_profile())


def configured_association_names():
    return list(dict.fromkeys(core.landmark_profile_association_name(profile) for profile in LANDMARK_SCENARIO_PROFILES))


def active_simulation_landmarks():
    return {str(index): (row["x_cm"], row["y_cm"])
            for index, row in enumerate(active_landmark_scenario_profile()["landmarks"])}


def select_association_profile(association_name):
    """Legacy headless-export option: select a profile with the requested name ambiguity."""
    global ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME
    shape = lambda profile: [(row["type"], row["x_cm"], row["y_cm"]) for row in profile["landmarks"]]
    current = active_landmark_scenario_profile()
    candidates = [profile for profile in LANDMARK_SCENARIO_PROFILES if core.landmark_profile_association_name(profile) == association_name]
    if not candidates:
        raise ValueError(f"No {association_name!r} landmark profile configured")
    selected = next((profile for profile in candidates if shape(profile) == shape(current)), candidates[0])
    ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME = selected["name"]


def paper_landmark_profiles():
    return sorted((profile for profile in LANDMARK_SCENARIO_PROFILES
                   if core.landmark_profile_association_name(profile) == active_association_name()
                   and all(row["type"] == "pose" for row in profile["landmarks"])),
                  key=lambda profile: len(profile["landmarks"]))


def step_count_for_duration(duration_s):
    return max(1, int(round(float(duration_s) / DT)))


def set_noise_cm(noise_std: float):
    global NOISE_STD_X_CM, NOISE_STD_Y_CM, NOISE_STD_ANGLE_DEG
    noise_std = max(0.0, float(noise_std))
    NOISE_STD_X_CM = noise_std
    NOISE_STD_Y_CM = noise_std
    NOISE_STD_ANGLE_DEG = noise_std


def set_motion_noise_percent(noise_percent: float):
    global MOTION_NOISE_RATIO
    MOTION_NOISE_RATIO = max(0.0, float(noise_percent)) / 100.0


def set_converge_test_runs(value: float):
    global CONVERGE_TEST_RUNS, CONVERGE_TEST_USE_REGRESSION_SET
    CONVERGE_TEST_RUNS = max(1, int(round(float(value))))
    CONVERGE_TEST_USE_REGRESSION_SET = False


def set_converge_test_regression_set():
    global CONVERGE_TEST_USE_REGRESSION_SET
    CONVERGE_TEST_USE_REGRESSION_SET = True


def set_dt(value: float):
    global DT
    DT = max(0.0, float(value))


def set_precision(value: float):
    global PI_CAM
    PI_CAM = max(0.0, float(value))


def set_precision_odo(value: float):
    global PI_ODO
    PI_ODO = max(0.0, float(value))


def set_precision_imu(value: float):
    global PI_IMU
    PI_IMU = max(0.0, float(value))


def set_learning_rate_xy(value: float):
    global LR_XY
    value = max(0.0, float(value))
    LR_XY = value


def set_learning_rate_angle(value: float):
    global LR_ANGLE
    value = max(0.0, float(value))
    LR_ANGLE = value


def set_motion_lr_vel_xy(value: float):
    global MOTION_LR_VEL_XY
    MOTION_LR_VEL_XY = max(0.0, float(value))


def set_motion_lr_acc_xy(value: float):
    global MOTION_LR_ACC_XY
    MOTION_LR_ACC_XY = max(0.0, float(value))


def set_motion_lr_vel_theta(value: float):
    global MOTION_LR_VEL_THETA
    MOTION_LR_VEL_THETA = max(0.0, float(value))


def set_motion_lr_acc_theta(value: float):
    global MOTION_LR_ACC_THETA
    MOTION_LR_ACC_THETA = max(0.0, float(value))


def apply_masterdata_setting_profile(profile):
    global PRECISION_DISTANCE_MODIFIER_ENABLED
    PRECISION_DISTANCE_MODIFIER_ENABLED = bool(profile["precision_distance_modifier_enabled"])
    set_dt(profile["dt"])
    set_precision(profile["pi_cam"])
    set_precision_odo(profile["pi_odo"])
    set_precision_imu(profile["pi_imu"])
    set_learning_rate_xy(profile["lr_xy"])
    set_learning_rate_angle(profile["lr_angle"])
    set_motion_lr_vel_xy(profile["motion_lr_vel_xy"])
    set_motion_lr_vel_theta(profile["motion_lr_vel_theta"])
    set_motion_lr_acc_xy(profile["motion_lr_acc_xy"])
    set_motion_lr_acc_theta(profile["motion_lr_acc_theta"])


def set_viewing_angle_deg(value: float):
    global VIEWING_ANGLE_DEG, VIEWING_HALF_ANGLE_RAD
    viewing_angle_deg = float(value)
    viewing_angle_deg = max(MIN_VIEWING_ANGLE_DEG, min(MAX_VIEWING_ANGLE_DEG, viewing_angle_deg))
    VIEWING_ANGLE_DEG = viewing_angle_deg
    VIEWING_HALF_ANGLE_RAD = math.radians(VIEWING_ANGLE_DEG / 2.0)


def apply_visibility_stress_paper_conditions(association_name=DATA_ASSOCIATION_UNIQUE_ID_NAME):
    global NOISE_ENABLED, VIEWING_ANGLE_ENABLED, LANDMARK_DROPOUT_ENABLED, LANDMARK_OBSTRUCTION_ENABLED
    NOISE_ENABLED = True
    VIEWING_ANGLE_ENABLED = True
    LANDMARK_DROPOUT_ENABLED = True
    LANDMARK_OBSTRUCTION_ENABLED = True
    select_association_profile(association_name)
    set_viewing_angle_deg(DEFAULT_VIEWING_ANGLE_DEG)
    reset_landmark_obstruction_state()
    reset_motion_slip_state()


def reset_landmark_obstruction_state():
    LANDMARK_OBSTRUCTION_REMAINING_S.clear()


def landmark_obstruction_active_count():
    return sum(1 for remaining_s in LANDMARK_OBSTRUCTION_REMAINING_S.values() if remaining_s > 0.0)


def reset_motion_slip_state():
    global MOTION_SLIP_REMAINING_S, MOTION_SLIP_AXIS, MOTION_SLIP_TURN_SIGN
    MOTION_SLIP_REMAINING_S = 0.0
    MOTION_SLIP_AXIS = "x"
    MOTION_SLIP_TURN_SIGN = 1.0


def motion_slip_is_active():
    return MOTION_SLIP_REMAINING_S > 0.0


def motion_slip_label():
    if not motion_slip_is_active():
        return "none"
    turn_label = "left" if MOTION_SLIP_TURN_SIGN > 0.0 else "right"
    return f"{MOTION_SLIP_AXIS}/{turn_label}"


# =========================
# SIMULATION EXPERIMENT
# =========================
# POC simulation layer on top of CORE.
# Builds simulated sensor observations and runs repeated simulated inference steps.

# --- Shared core adapter ---
def ensure_simulation_core_configuration():
    profile = active_landmark_scenario_profile()
    if core.ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME != profile["name"]:
        core.init_landmarks({"profile": profile["name"], "landmarks": profile["landmarks"]})
    for name in core.CORE_SETTING_DEFAULTS:
        if name != "ACTIVE_DATA_ASSOCIATION_NAME":
            setattr(core, name, globals()[name])


def simulation_step_from_observations(belief, observations, motion, algorithm_name, step_dt_s, timestamp=None):
    """Simulator adapter: one supplied frame through the robot's public core API."""
    ensure_simulation_core_configuration()
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    response = core.step_core(dict(timestamp=timestamp, belief_fcs_gc=belief,
                                   landmark_observations_rcs=observations, motion_observation=motion,
                                   algorithm_name=algorithm_name, step_dt_s=step_dt_s))
    return response


# --- Non-core state views ---
def robot_pose_view(robot_fcs_gc):
    x_cm, y_cm, theta_rad = gc_2_scalar(robot_fcs_gc)
    return {
        "x_cm": x_cm,
        "y_cm": y_cm,
        "theta_rad": theta_rad,
    }


def robot_gc_view(robot_fcs_gc):
    return {
        "x": list(robot_fcs_gc["x"]),
        "y": list(robot_fcs_gc["y"]),
        "theta": list(robot_fcs_gc["theta"]),
    }


# --- Motion sensing simulation ---
def sample_motion_slip_duration_s():
    duration_s = random.gauss(MOTION_SLIP_MEAN_S, MOTION_SLIP_STD_S)
    return max(DT, duration_s)


def motion_slips_for_this_step(step_dt_s=DT):
    global MOTION_SLIP_REMAINING_S, MOTION_SLIP_AXIS, MOTION_SLIP_TURN_SIGN
    if not MOTION_SLIP_ENABLED:
        return False

    if MOTION_SLIP_REMAINING_S > 0.0:
        MOTION_SLIP_REMAINING_S = max(0.0, MOTION_SLIP_REMAINING_S - step_dt_s)
        return True

    event_probability = MOTION_SLIP_EVENT_PROBABILITY * max(0.0, step_dt_s) / MOTION_SLIP_EVENT_INTERVAL_S
    if random.random() < event_probability:
        MOTION_SLIP_REMAINING_S = max(0.0, sample_motion_slip_duration_s() - step_dt_s)
        MOTION_SLIP_AXIS = random.choice(("x", "y"))
        MOTION_SLIP_TURN_SIGN = random.choice((-1.0, 1.0))
        return True

    return False


def apply_motion_slip_to_observation(motion_obs):
    if MOTION_SLIP_AXIS == "y":
        velocity_key = "y_vel_cm_s"
    else:
        velocity_key = "x_vel_cm_s"

    original_velocity = motion_obs[velocity_key]
    slipped_velocity = original_velocity * MOTION_SLIP_VELOCITY_FACTOR
    slip_delta_cm_s = slipped_velocity - original_velocity

    motion_obs[velocity_key] = slipped_velocity
    if MOTION_SLIP_ROTATION_LEVER_CM > 0.0:
        motion_obs["theta_vel_rad_s"] += MOTION_SLIP_TURN_SIGN * abs(slip_delta_cm_s) / MOTION_SLIP_ROTATION_LEVER_CM


def simulation_sense_motion(robot_fcs_gc):
    # Input:
    #   robot state in generalized coordinates
    # Output:
    #   motion observations with velocity/acceleration in RCS:
    #   x_vel_cm_s, y_vel_cm_s, theta_vel_rad_s, x_acc_cm_s2, y_acc_cm_s2, theta_acc_rad_s2
    # Function:
    #   builds simulated motion observations from generalized coordinates
    #   rotates translational velocity and acceleration from FCS into RCS
    #   optionally simulates directional wheel slip by scaling one velocity reading
    #   upward and adding a coupled yaw-rate reading
    #   adds motion noise when NOISE_ENABLED is active

    def motion_noise_std(motion_obs):
        # Scale the Gaussian noise standard deviation with the magnitude of each motion signal.
        # This keeps fast motion noisier than near-stationary motion.
        return {
            key: abs(float(value)) * MOTION_NOISE_RATIO
            for key, value in motion_obs.items()
        }

    x_vel_rcs, y_vel_rcs, theta_vel = vel_fcs_2_rcs(
        robot_fcs_gc,
        robot_fcs_gc["x"][1],
        robot_fcs_gc["y"][1],
        robot_fcs_gc["theta"][1],
    )
    x_acc_rcs, y_acc_rcs, theta_acc = acc_fcs_2_rcs(
        robot_fcs_gc,
        robot_fcs_gc["x"][2],
        robot_fcs_gc["y"][2],
        robot_fcs_gc["theta"][2],
    )

    motion_obs = {
        "x_vel_cm_s": x_vel_rcs,
        "y_vel_cm_s": y_vel_rcs,
        "theta_vel_rad_s": theta_vel,
        "x_acc_cm_s2": x_acc_rcs,
        "y_acc_cm_s2": y_acc_rcs,
        "theta_acc_rad_s2": theta_acc,
    }

    if ODOMETRY_BIAS_ENABLED:
        motion_obs["x_vel_cm_s"] += ODOMETRY_BIAS_X_CM_S
        motion_obs["y_vel_cm_s"] += ODOMETRY_BIAS_Y_CM_S
        motion_obs["theta_vel_rad_s"] += ODOMETRY_BIAS_THETA_RAD_S

    if motion_slips_for_this_step(DT):
        apply_motion_slip_to_observation(motion_obs)

    if NOISE_ENABLED:
        # Build per-signal noise levels from the clean motion observation before perturbing it.
        noise_std = motion_noise_std(motion_obs)
        motion_obs["x_vel_cm_s"] += random.gauss(0, noise_std["x_vel_cm_s"])
        motion_obs["y_vel_cm_s"] += random.gauss(0, noise_std["y_vel_cm_s"])
        motion_obs["theta_vel_rad_s"] += random.gauss(0, noise_std["theta_vel_rad_s"])
        motion_obs["x_acc_cm_s2"] += random.gauss(0, noise_std["x_acc_cm_s2"])
        motion_obs["y_acc_cm_s2"] += random.gauss(0, noise_std["y_acc_cm_s2"])
        motion_obs["theta_acc_rad_s2"] += random.gauss(0, noise_std["theta_acc_rad_s2"])

    return motion_obs


# --- Landmark visibility simulation ---
def landmark_drops_out_for_this_reading():
    if not LANDMARK_DROPOUT_ENABLED:
        return False
    return random.random() < LANDMARK_DROPOUT_PROBABILITY


def sample_landmark_obstruction_duration_s():
    duration_s = random.gauss(LANDMARK_OBSTRUCTION_MEAN_S, LANDMARK_OBSTRUCTION_STD_S)
    return max(DT, duration_s)


def landmark_is_obstructed_for_this_step(landmark_id, step_dt_s=DT):
    if not LANDMARK_OBSTRUCTION_ENABLED:
        return False

    remaining_s = LANDMARK_OBSTRUCTION_REMAINING_S.get(landmark_id, 0.0)
    if remaining_s > 0.0:
        LANDMARK_OBSTRUCTION_REMAINING_S[landmark_id] = max(0.0, remaining_s - step_dt_s)
        return True

    event_probability = LANDMARK_OBSTRUCTION_EVENT_PROBABILITY * max(0.0, step_dt_s) / LANDMARK_OBSTRUCTION_EVENT_INTERVAL_S
    if random.random() < event_probability:
        LANDMARK_OBSTRUCTION_REMAINING_S[landmark_id] = max(
            0.0,
            sample_landmark_obstruction_duration_s() - step_dt_s,
        )
        return True

    return False


def simulation_sense_landmarks(robot_fcs_gc, landmarks_fcs, return_ground_truth=False):
    # Input:
    #   robot state in generalized coordinates
    #   landmark positions in FCS: {id: (x_cm, y_cm)}
    # Output:
    #   named RCS observations; optional separate {obs_id: actual map point} for reports
    # Function:
    #   builds simulated sensory observations
    #   emits x/y landmark observations only; angle_rad is calculated by the core preparation step
    #   adds landmark observation noise when NOISE_ENABLED is active
    #   optionally drops individual readings
    #   optionally hides landmarks for short obstruction windows
    #   only visible landmarks if viewing angle is enabled
    obs = []
    ground_truth = {}
    # Preserve report observation labels while recognition names change by scenario.
    landmarks = active_landmark_scenario_profile()["landmarks"]
    for landmark_id, (landmark_x_cm, landmark_y_cm) in landmarks_fcs.items():
        x_rcs, y_rcs, angle = landmark_fcs_2_rcs(robot_fcs_gc, landmark_x_cm, landmark_y_cm)
        landmark_obs = {"id": landmark_id,
                        "name": landmarks[int(landmark_id)]["name"],
                        "x_rcs_cm": x_rcs, "y_rcs_cm": y_rcs}
        if VIEWING_ANGLE_ENABLED and not landmark_is_within_viewing_angle(
            robot_fcs_gc,
            landmark_x_cm,
            landmark_y_cm,
            VIEWING_HALF_ANGLE_RAD,
        ):
            continue

        if landmark_is_obstructed_for_this_step(landmark_obs["id"], DT):
            continue

        if landmark_drops_out_for_this_reading():
            continue

        x_rcs_cm = landmark_obs["x_rcs_cm"]
        y_rcs_cm = landmark_obs["y_rcs_cm"]

        if NOISE_ENABLED:
            x_rcs_cm += random.gauss(0, NOISE_STD_X_CM)
            y_rcs_cm += random.gauss(0, NOISE_STD_Y_CM)

        obs_id = f"obs_{len(obs)}"
        ground_truth[obs_id] = landmark_obs["id"]
        obs.append({
            "obs_id": obs_id,
            "name": landmark_obs["name"],
            "x_rcs_cm": x_rcs_cm,
            "y_rcs_cm": y_rcs_cm,
        })
    return (obs, ground_truth) if return_ground_truth else obs


def simulation_inference_step(robots, algorithm_name=ACTIVE_ALGORITHM_NAME):
    # Input:
    #   robots: dict with simulated actual and belief robot states
    #   algorithm_name: selects the active inference mode
    # Output:
    #   y_obs_map, yhat_map, y_motion_obs, free_energy for POC reporting and visualisation
    # Function:
    #   builds simulated sensor observations from the actual robot state
    #   passes x/y landmark observations into the core, where angle_rad is calculated
    #   delegates the belief update to the headless core inference step
    y_obs_list, ground_truth = simulation_sense_landmarks(robots["actual"], active_simulation_landmarks(), return_ground_truth=True)
    y_motion_obs = simulation_sense_motion(robots["actual"])
    response = simulation_step_from_observations(
        robots["belief"], y_obs_list, y_motion_obs, algorithm_name, DT)
    robots["belief"] = response["belief_fcs_gc"]
    diagnostics = response["diagnostics"]
    # Add simulator identity only to the report view, after the core call.
    report_observations = {key: dict(obs, landmark_id=ground_truth[key]) for key, obs in diagnostics["y_obs"].items()}
    return report_observations, diagnostics["y_hat"], y_motion_obs, diagnostics["free_energy"]


# --- Simulated experiment runner ---
def run_robustness_trial(actual_start, offset, algorithm_name, initialize_path, step_path,
                         progress_callback=None):
    """Execute one supplied start/offset using the selected demo adapter and sensors."""
    robots = {"actual": make_robot_gc_state(**actual_start)}
    path = initialize_path(robots["actual"])
    actual_pose = robot_pose_view(robots["actual"])
    robots["belief"] = make_robot_gc_state(
        actual_pose["x_cm"] + offset[0], actual_pose["y_cm"] + offset[1],
        wrap_rad(actual_pose["theta_rad"] + math.radians(offset[2])),
    )
    start_belief = robot_pose_view(robots["belief"])
    reset_landmark_obstruction_state()
    reset_motion_slip_state()
    correct_hits = symmetrical_hits = 0
    outcome = "timeout"
    steps = 0
    for step in range(1, INFER_STEPS_MAX + 1):
        if progress_callback is not None and (step == 1 or step % 50 == 0):
            if progress_callback(step, INFER_STEPS_MAX) is False:
                outcome = "cancelled"
                break
        step_path(robots["actual"], path)
        simulation_inference_step(robots, algorithm_name)
        steps = step
        actual_end = robot_pose_view(robots["actual"])
        belief_end = robot_pose_view(robots["belief"])
        correct = pose_converged_ros(
            actual_end, belief_end, CONV_TOL_X_CM, CONV_TOL_Y_CM, CONV_TOL_ANGLE_RAD,
        )
        correct_hits = correct_hits + 1 if correct else 0
        symmetrical_hits = symmetrical_hits + 1 if pose_symmetrically_converged(actual_end, belief_end) else 0
        if correct_hits >= CONV_REQUIRED_SEQUENTIAL_HITS:
            outcome = "correct"
            break
        if symmetrical_hits >= CONV_REQUIRED_SEQUENTIAL_HITS:
            outcome = "symmetric"
            break
    return {
        "actual_start": actual_pose, "belief_start": start_belief,
        "actual_end": robot_pose_view(robots["actual"]),
        "belief_end": robot_pose_view(robots["belief"]),
        "outcome": outcome, "steps": steps, "elapsed_s": steps * DT,
    }


def reset_paper_lm_experiment(saved_setup):
    """Create an independent trial at the saved actual/belief poses and path start."""
    from copy import deepcopy

    experiment = deepcopy(saved_setup)
    reset_landmark_obstruction_state()
    reset_motion_slip_state()
    return experiment


def run_paper_lm_batch(saved_setup, path_step, duration_s=PAPER_LM_TRIAL_DURATION_S, progress_callback=None, record_callback=None):
    """Run N fixed-start trials per landmark profile; retain error series and the last history."""
    from array import array

    global ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME
    global MOTION_SLIP_REMAINING_S, MOTION_SLIP_AXIS, MOTION_SLIP_TURN_SIGN
    max_steps = max(1, int(round(duration_s / saved_setup["settings"]["dt"])))
    previous_profile = ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME
    previous_obstructions = dict(LANDMARK_OBSTRUCTION_REMAINING_S)
    previous_slip = (MOTION_SLIP_REMAINING_S, MOTION_SLIP_AXIS, MOTION_SLIP_TURN_SIGN)
    result = {"trials": [], "last_history": [], "stopped": False, "runs_per_profile": saved_setup["runs"]}
    try:
        for profile in paper_landmark_profiles():
            ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME = profile["name"]
            profile_name = profile["name"]
            for trial_number in range(1, saved_setup["runs"] + 1):
                experiment = reset_paper_lm_experiment(saved_setup)
                settings = experiment["settings"]
                settings["landmark_scenario_profile"] = profile["name"]
                settings["landmark_scenario_count"] = len(active_simulation_landmarks())
                settings["landmark_profile"] = deepcopy(profile)
                history = []
                trial = {
                    "landmark_profile": profile_name,
                    "trial": trial_number,
                    "dt": settings["dt"],
                    "actual_start": robot_pose_view(experiment["robots"]["actual"]),
                    "belief_start": robot_pose_view(experiment["robots"]["belief"]),
                    "position_error_cm": array("d"),
                    "heading_error_deg": array("d"),
                    "steps": 0,
                    "completed": False,
                    "target_activation_steps": [],
                }
                result["trials"].append(trial)
                result["last_history"] = history
                if record_callback is not None:
                    record_callback(profile_name, trial_number, {
                        "step": 0,
                        "actual_pose": trial["actual_start"],
                        "belief_pose": trial["belief_start"],
                        "visible_landmark_count": None,
                        "free_energy": None,
                        "settings": settings,
                    })
                for step in range(1, max_steps + 1):
                    if progress_callback is not None and (step == 1 or (step - 1) % 100 == 0):
                        if progress_callback(profile_name, trial_number, saved_setup["runs"], step - 1, max_steps) is False:
                            result["stopped"] = True
                            return result
                    path_state = experiment["demo_state"].get("path_state", {})
                    previous_target = (path_state.get("target_x_cm"), path_state.get("target_y_cm"))
                    path_step(experiment)
                    path_state = experiment["demo_state"].get("path_state", {})
                    next_target = (path_state.get("target_x_cm"), path_state.get("target_y_cm"))
                    if next_target != previous_target:
                        trial["target_activation_steps"].append(step)
                    y_obs, y_hat, motion_obs, free_energy = simulation_inference_step(
                        experiment["robots"], saved_setup["algorithm_name"],
                    )
                    record = snapshot_record(step, experiment["robots"], y_obs, y_hat, motion_obs, free_energy, settings)
                    history.append(record)
                    if record_callback is not None:
                        record_callback(profile_name, trial_number, record)
                    actual = record["actual_pose"]
                    belief = record["belief_pose"]
                    trial["position_error_cm"].append(math.hypot(belief["x_cm"] - actual["x_cm"], belief["y_cm"] - actual["y_cm"]))
                    trial["heading_error_deg"].append(abs(math.degrees(wrap_rad(belief["theta_rad"] - actual["theta_rad"]))))
                    trial["steps"] = step
                trial["completed"] = True
    finally:
        ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME = previous_profile
        LANDMARK_OBSTRUCTION_REMAINING_S.clear()
        LANDMARK_OBSTRUCTION_REMAINING_S.update(previous_obstructions)
        MOTION_SLIP_REMAINING_S, MOTION_SLIP_AXIS, MOTION_SLIP_TURN_SIGN = previous_slip
    return result


def run_experiment(
    robots,
    algorithm_name,
    max_steps,
    until_convergence=False,
    step_callback=None,
    callback_interval=20,
    stop_on_symmetrical_convergence=False,
):
    # Input:
    #   robots: dict with actual and belief robot states
    #   algorithm_name: selects the active inference mode
    #   max_steps: maximum number of inference steps to run
    #   until_convergence: whether to stop early once the belief converges to the actual pose
    #   step_callback: optional callback invoked during the run for UI refresh / interruption handling
    #   callback_interval: callback cadence in steps
    #   stop_on_symmetrical_convergence: convergence tests may also stop at the rotated solution
    # Output:
    #   history: list of per-step experiment snapshots for reporting and plotting
    # Function:
    #   runs repeated inference steps under the selected algorithm
    #   records a snapshot after each step
    #   optionally updates an external callback during the run
    #   optionally stops early when the belief pose converges
    history = []
    settings = experiment_settings(algorithm_name)
    convergence_hits = 0
    symmetrical_hits = 0

    for k in range(1, max_steps + 1):
        y_obs_map, yhat_map, y_motion_obs, F = simulation_inference_step(robots, algorithm_name)
        history.append(snapshot_record(k, robots, y_obs_map, yhat_map, y_motion_obs, F, settings))

        if step_callback is not None and (k % callback_interval == 0 or k == 1 or k == max_steps):
            keep_running = step_callback()
            plt.pause(0.001)
            if keep_running is False:
                break

        if until_convergence:
            if pose_converged_ros(
                robot_pose_view(robots["actual"]),
                robot_pose_view(robots["belief"]),
                CONV_TOL_X_CM,
                CONV_TOL_Y_CM,
                CONV_TOL_ANGLE_RAD,
            ):
                convergence_hits += 1
                if convergence_hits >= CONV_REQUIRED_SEQUENTIAL_HITS:
                    if step_callback is not None:
                        step_callback()
                        plt.pause(0.001)
                    break
            else:
                convergence_hits = 0

            if stop_on_symmetrical_convergence:
                if pose_symmetrically_converged(history[-1]["actual_pose"], history[-1]["belief_pose"]):
                    symmetrical_hits += 1
                    if symmetrical_hits >= CONV_REQUIRED_SEQUENTIAL_HITS:
                        break
                else:
                    symmetrical_hits = 0

    return history




# =========================
# REPORT
# =========================
PAPER_LM_REPORT_COUNTS = (3, 5, 9)

def robot_generalized_coordinate_view(robot_fcs_gc):
    return robot_gc_view(robot_fcs_gc)


def near_start_condition_text(limits=None):
    """Describe the configured near bounds or a saved report's bounds."""
    limits = NEAR_START_LIMITS if limits is None else limits
    return (f"belief dx uniform +/-{limits['x_cm']:g} cm; dy uniform +/-{limits['y_cm']:g} cm; "
            f"dtheta uniform +/-{limits['angle_deg']:g} deg")


def precision_label_for_algorithm(algorithm_name, base_precision=None):
    if base_precision is None:
        base_precision = PI_CAM

    if algorithm_name == FUSED_MOTION_LANDMARK_ALGORITHM_NAME:
        return (
            f"fused cam={base_precision:.4f} odo={PI_ODO:.4f} imu={PI_IMU:.4f}"
        )
    if algorithm_name == MOTION_ONLY_ALGORITHM_NAME:
        return (
            f"motion odo={PI_ODO:.4f} imu={PI_IMU:.4f}"
        )
    return (
        f"distance-weighted cam={base_precision:.4f} odo={PI_ODO:.4f} imu={PI_IMU:.4f}"
    )


def gain_for_algorithm(algorithm_name):
    if algorithm_name == MOTION_ONLY_ALGORITHM_NAME:
        return {
            "xy": MOTION_LR_VEL_XY,
            "angle": MOTION_LR_VEL_THETA,
        }
    if algorithm_name == FUSED_MOTION_LANDMARK_ALGORITHM_NAME:
        return {"xy": LR_XY, "angle": LR_ANGLE}
    return {"xy": LR_XY, "angle": LR_ANGLE}


def experiment_settings(algorithm_name):
    return {
        "near_start_limits": dict(NEAR_START_LIMITS),
        "noise_enabled": NOISE_ENABLED,
        "noise_x_cm": NOISE_STD_X_CM,
        "noise_y_cm": NOISE_STD_Y_CM,
        "noise_angle_rad": math.radians(NOISE_STD_ANGLE_DEG),
        "motion_noise_ratio": MOTION_NOISE_RATIO,
        "odometry_bias_enabled": ODOMETRY_BIAS_ENABLED,
        "odometry_bias_x_cm_s": ODOMETRY_BIAS_X_CM_S,
        "odometry_bias_y_cm_s": ODOMETRY_BIAS_Y_CM_S,
        "odometry_bias_theta_rad_s": ODOMETRY_BIAS_THETA_RAD_S,
        "motion_slip_enabled": MOTION_SLIP_ENABLED,
        "motion_slip_event_probability": MOTION_SLIP_EVENT_PROBABILITY,
        "motion_slip_event_interval_s": MOTION_SLIP_EVENT_INTERVAL_S,
        "motion_slip_mean_s": MOTION_SLIP_MEAN_S,
        "motion_slip_std_s": MOTION_SLIP_STD_S,
        "motion_slip_velocity_factor": MOTION_SLIP_VELOCITY_FACTOR,
        "motion_slip_rotation_lever_cm": MOTION_SLIP_ROTATION_LEVER_CM,
        "landmark_dropout_enabled": LANDMARK_DROPOUT_ENABLED,
        "landmark_dropout_probability": LANDMARK_DROPOUT_PROBABILITY,
        "landmark_obstruction_enabled": LANDMARK_OBSTRUCTION_ENABLED,
        "landmark_obstruction_event_probability": LANDMARK_OBSTRUCTION_EVENT_PROBABILITY,
        "landmark_obstruction_event_interval_s": LANDMARK_OBSTRUCTION_EVENT_INTERVAL_S,
        "landmark_obstruction_mean_s": LANDMARK_OBSTRUCTION_MEAN_S,
        "landmark_obstruction_std_s": LANDMARK_OBSTRUCTION_STD_S,
        "viewing_angle_enabled": VIEWING_ANGLE_ENABLED,
        "viewing_angle_deg": VIEWING_ANGLE_DEG,
        "precision_cam": PI_CAM,
        "precision_odo": PI_ODO,
        "precision_imu": PI_IMU,
        "precision_distance_modifier_enabled": PRECISION_DISTANCE_MODIFIER_ENABLED,
        "precision_label": precision_label_for_algorithm(algorithm_name, PI_CAM),
        "dt": DT,
        "learning_rate": gain_for_algorithm(algorithm_name),
        "lr_cam_xy": LR_XY,
        "lr_cam_angle": LR_ANGLE,
        "lr_motion_vel_xy": MOTION_LR_VEL_XY,
        "lr_motion_vel_theta": MOTION_LR_VEL_THETA,
        "lr_motion_acc_xy": MOTION_LR_ACC_XY,
        "lr_motion_acc_theta": MOTION_LR_ACC_THETA,
        "data_association": active_association_name(),
        "landmark_association_gate_cm": LANDMARK_ASSOCIATION_GATE_CM,
        "landmark_scenario_profile": ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME,
        "landmark_profile": deepcopy(active_landmark_scenario_profile()),
        "landmark_scenario_count": len(active_landmark_scenario_profile()["landmarks"]),
    }


def sensory_pose_observation(actual_fcs_gc):
    pose = robot_pose_view(actual_fcs_gc)
    observed_x = pose["x_cm"]
    observed_y = pose["y_cm"]
    observed_theta = pose["theta_rad"]

    if NOISE_ENABLED:
        observed_x += random.gauss(0, NOISE_STD_X_CM)
        observed_y += random.gauss(0, NOISE_STD_Y_CM)
        observed_theta = wrap_rad(observed_theta + math.radians(random.gauss(0, NOISE_STD_ANGLE_DEG)))

    return {
        "x_cm": observed_x,
        "y_cm": observed_y,
        "theta_rad": observed_theta,
    }


def snapshot_record(step, robots, y_obs_map, yhat_map, y_motion_obs, free_energy, settings):
    observed_pose = sensory_pose_observation(robots["actual"])
    return {
        "step": step,
        "actual_pose": robot_pose_view(robots["actual"]),
        "observed_pose": observed_pose,
        "belief_pose": robot_pose_view(robots["belief"]),
        "actual_gc": robot_generalized_coordinate_view(robots["actual"]),
        "belief_gc": robot_generalized_coordinate_view(robots["belief"]),
        "y_obs": y_obs_map,
        "y_hat": yhat_map,
        "motion_obs": dict(y_motion_obs) if y_motion_obs is not None else None,
        "visible_landmark_ids": sorted(y_obs_map.keys()),
        "visible_landmark_count": len(y_obs_map),
        "obstruction_active_count": landmark_obstruction_active_count(),
        "free_energy": free_energy,
        "settings": settings,
    }


def compact_tracking_export_row(record, time_s):
    actual_pose = record["actual_pose"]
    belief_pose = record["belief_pose"]
    dx_cm = belief_pose["x_cm"] - actual_pose["x_cm"]
    dy_cm = belief_pose["y_cm"] - actual_pose["y_cm"]
    heading_error_rad = wrap_rad(belief_pose["theta_rad"] - actual_pose["theta_rad"])
    visible_landmark_ids = record.get("visible_landmark_ids", [])
    return {
        "step": record["step"],
        "time_s": time_s,
        "actual_x_cm": actual_pose["x_cm"],
        "actual_y_cm": actual_pose["y_cm"],
        "actual_theta_rad": actual_pose["theta_rad"],
        "belief_x_cm": belief_pose["x_cm"],
        "belief_y_cm": belief_pose["y_cm"],
        "belief_theta_rad": belief_pose["theta_rad"],
        "position_error_cm": math.hypot(dx_cm, dy_cm),
        "heading_error_deg": abs(math.degrees(heading_error_rad)),
        "visible_landmark_count": record.get("visible_landmark_count", len(visible_landmark_ids)),
        "visible_landmark_ids": ";".join(visible_landmark_ids),
        "obstruction_active_count": record.get("obstruction_active_count", 0),
        "precision_distance_modifier_enabled": record["settings"]["precision_distance_modifier_enabled"],
        "free_energy": record["free_energy"],
    }


def export_paper_lm_dataset(saved_setup, path_step, duration_s=PAPER_LM_TRIAL_DURATION_S, progress_callback=None):
    """Stream all P-LM trials to one CSV, with settings and completion metadata."""
    import csv
    import datetime
    import json
    import os

    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_path = os.path.join(reports_dir, f"paper_lm_dataset_{timestamp}")
    csv_path = base_path + ".csv"
    json_path = base_path + ".json"
    fieldnames = [
        "landmark_profile", "trial", "step", "time_s",
        "actual_x_cm", "actual_y_cm", "actual_theta_rad",
        "belief_x_cm", "belief_y_cm", "belief_theta_rad",
        "position_error_cm", "heading_error_deg",
        "visible_landmark_count", "free_energy",
    ]
    rows_written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        def write_record(profile, trial, record):
            nonlocal rows_written
            values = compact_tracking_export_row(record, record["step"] * record["settings"]["dt"])
            row = {key: values[key] for key in fieldnames if key not in {"landmark_profile", "trial"}}
            row.update(landmark_profile=profile, trial=trial)
            writer.writerow(row)
            rows_written += 1

        result = run_paper_lm_batch(
            saved_setup, path_step, duration_s=duration_s,
            progress_callback=progress_callback, record_callback=write_record,
        )

    metadata = {
        "csv_file": os.path.basename(csv_path),
        "rows": rows_written,
        "runs_per_profile": saved_setup["runs"],
        "requested_trial_duration_s": duration_s,
        "steps_per_trial": max(1, int(round(duration_s / saved_setup["settings"]["dt"]))),
        "algorithm_name": saved_setup["algorithm_name"],
        "settings": saved_setup["settings"],
        "landmark_profiles": paper_landmark_profiles(),
        "actual_start": robot_pose_view(saved_setup["robots"]["actual"]),
        "belief_start": robot_pose_view(saved_setup["robots"]["belief"]),
        "ball": saved_setup["ball"],
        "path_start": {key: value for key, value in saved_setup["demo_state"].items() if key != "paths"},
        "path_random_state": saved_setup["path_random_state"],
        "stopped": result["stopped"],
        "trials": [
            {key: trial[key] for key in ("landmark_profile", "trial", "steps", "completed", "target_activation_steps")}
            for trial in result["trials"]
        ],
        "data_notes": {
            "time_s": "Elapsed simulation time within each trial; step 0 is the saved initial state.",
            "position_error_cm": "Euclidean distance between belief and actual position.",
            "heading_error_deg": "Absolute wrapped belief-minus-actual heading error, in degrees.",
            "visible_landmark_count": "Number of observations supplied for this step; blank at step 0.",
            "free_energy": "Inference diagnostic for the update producing this row; blank at step 0.",
        },
    }
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)
    result.update(csv_path=csv_path, json_path=json_path, rows=rows_written)
    return result


def simulation_settings_report_lines(settings):
    odometry_bias_text = (
        f"Odom bias: {'YES' if settings.get('odometry_bias_enabled') else 'NO'}"
    )
    if settings.get("odometry_bias_enabled"):
        odometry_bias_text += (
            f"   x={settings.get('odometry_bias_x_cm_s', 0.0):.1f} cm/s"
            f"   y={settings.get('odometry_bias_y_cm_s', 0.0):.1f} cm/s"
            f"   theta={settings.get('odometry_bias_theta_rad_s', 0.0):.4f} rad/s"
        )
    slip_text = (
        f"Slip: {'YES' if settings.get('motion_slip_enabled') else 'NO'}"
        f"   p={settings.get('motion_slip_event_probability', 0.0):.0%}/s"
        f"   factor={settings.get('motion_slip_velocity_factor', 0.0):.1f}x"
    )
    dropout_text = (
        f"Dropout: {'YES' if settings.get('landmark_dropout_enabled') else 'NO'}"
        f"   p={settings.get('landmark_dropout_probability', 0.0):.0%}/reading"
    )
    obstruction_text = (
        f"Obstruction: {'YES' if settings.get('landmark_obstruction_enabled') else 'NO'}"
        f"   p={settings.get('landmark_obstruction_event_probability', 0.0):.0%}/s"
        f"   win={settings.get('landmark_obstruction_mean_s', 0.0):.1f}+/-{settings.get('landmark_obstruction_std_s', 0.0):.1f}s"
    )
    viewing_text = (
        f"FoV: {'YES' if settings.get('viewing_angle_enabled') else 'NO'}"
        f"   angle={settings.get('viewing_angle_deg', 0.0):.1f} deg"
    )
    association_text = (
        f"Association={settings.get('data_association', active_association_name())}"
        f"   gate={settings.get('landmark_association_gate_cm', LANDMARK_ASSOCIATION_GATE_CM):.0f} cm"
    )
    landmark_profile_text = (
        f"LM profile={settings.get('landmark_scenario_profile', ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME)}"
    )
    return odometry_bias_text, slip_text, dropout_text, obstruction_text, viewing_text, association_text, landmark_profile_text


def report_value(source, primary_key, fallback_key=None, default=0.0):
    if primary_key in source:
        return source[primary_key]
    if fallback_key is not None and fallback_key in source:
        return source[fallback_key]
    return default


def precision_report_row(settings):
    return (
        f"PI: cam={format_display_value(report_value(settings, 'precision_cam', 'pi_cam', PI_CAM))}   "
        f"odo={format_display_value(report_value(settings, 'precision_odo', 'pi_odo', PI_ODO))}   "
        f"imu={format_display_value(report_value(settings, 'precision_imu', 'pi_imu', PI_IMU))}"
    )


def precision_distance_modifier_report_row(settings):
    enabled = settings.get("precision_distance_modifier_enabled", PRECISION_DISTANCE_MODIFIER_ENABLED)
    return f"Distance precision modifier: {'YES' if enabled else 'NO'}"


def learning_rate_report_row(settings):
    return (
        f"LR: cam xy={format_display_value(report_value(settings, 'lr_cam_xy', 'lr_xy', LR_XY))}   "
        f"cam a={format_display_value(report_value(settings, 'lr_cam_angle', 'lr_angle', LR_ANGLE))}   "
        f"vel xy={format_display_value(report_value(settings, 'lr_motion_vel_xy', 'motion_lr_vel_xy', MOTION_LR_VEL_XY))}   "
        f"vel a={format_display_value(report_value(settings, 'lr_motion_vel_theta', 'motion_lr_vel_theta', MOTION_LR_VEL_THETA))}   "
        f"acc xy={format_display_value(report_value(settings, 'lr_motion_acc_xy', 'motion_lr_acc_xy', MOTION_LR_ACC_XY))}   "
        f"acc a={format_display_value(report_value(settings, 'lr_motion_acc_theta', 'motion_lr_acc_theta', MOTION_LR_ACC_THETA))}"
    )


def convergence_condition_report_row(x_cm, y_cm, angle_rad, sequential_hits):
    return (
        f"Convergence condition: |dx|<={x_cm:.1f} cm   |dy|<={y_cm:.1f} cm   "
        f"|dtheta|<={angle_rad:.4f} rad   sequential hits>={sequential_hits}"
    )


def convergence_report_condition_lines(settings):
    """Format the saved test conditions shared by convergence and scan reports."""
    if settings["noise_enabled"]:
        noise_text = (
            f"Noise: YES   landmarks x={settings['noise_x_cm']:.2f} cm   "
            f"y={settings['noise_y_cm']:.2f} cm   angle={settings['noise_angle_rad']:.4f} rad   "
            f"motion={settings.get('motion_noise_ratio', MOTION_NOISE_RATIO) * 100.0:.1f}%"
        )
    else:
        noise_text = "Noise: NO"
    (odometry_bias_text, slip_text, dropout_text, obstruction_text,
     viewing_text, association_text, landmark_profile_text) = simulation_settings_report_lines(settings)
    return [
        noise_text,
        odometry_bias_text,
        slip_text,
        f"{dropout_text}   |   {obstruction_text}",
        f"{viewing_text}   |   {association_text}   |   {landmark_profile_text}",
        f"DT={settings['dt']:.4f}",
        precision_report_row(settings),
        precision_distance_modifier_report_row(settings),
        learning_rate_report_row(settings),
    ]


def paper_lm_profile_counts(metadata):
    # Read either self-contained profile snapshots or historical report metadata.
    return {profile["name"]: len(profile.get("landmarks", profile.get("landmark_ids", [])))
            for profile in metadata["landmark_profiles"]}


def paper_lm_mean_errors(csv_path, metadata):
    """Compute mean, sample SD and P1/P99 of absolute errors at each time."""
    import csv

    profile_counts = paper_lm_profile_counts(metadata)
    selected_profiles = [name for name, count in profile_counts.items() if count in PAPER_LM_REPORT_COUNTS]
    completed = {
        (trial["landmark_profile"], trial["trial"])
        for trial in metadata["trials"]
        if trial["completed"] and trial["landmark_profile"] in selected_profiles
    }
    sums = {}
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            profile = row["landmark_profile"]
            if (profile, int(row["trial"])) not in completed:
                continue
            step = int(row["step"])
            # Per timepoint: time, count, means, M2 values, and per-axis trial samples.
            bucket = sums.setdefault(profile, {}).setdefault(step, [float(row["time_s"]), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], [], []])
            bucket[1] += 1
            errors = (
                abs(float(row["belief_x_cm"]) - float(row["actual_x_cm"])),
                abs(float(row["belief_y_cm"]) - float(row["actual_y_cm"])),
                abs(math.degrees(wrap_rad(float(row["belief_theta_rad"]) - float(row["actual_theta_rad"])))),
            )
            # Welford accumulation avoids cancellation when trials are nearly equal.
            for idx, error in enumerate(errors):
                delta = error - bucket[2 + idx]
                bucket[2 + idx] += delta / bucket[1]
                bucket[5 + idx] += delta * (error - bucket[2 + idx])
                bucket[8 + idx].append(error)

    curves = {}
    for profile in selected_profiles:
        if profile not in sums:
            continue
        n = sum(p == profile for p, _ in completed)
        steps = sorted(sums[profile])
        if steps != list(range(metadata["steps_per_trial"] + 1)) or any(sums[profile][step][1] != n for step in steps):
            raise ValueError(f"P-LM CSV has missing or duplicate samples for LM {profile}")
        buckets = [sums[profile][step] for step in steps]
        curves[profile] = {
            "n": n,
            "time_s": [values[0] for values in buckets],
            "x_cm": [values[2] for values in buckets],
            "y_cm": [values[3] for values in buckets],
            "angle_deg": [values[4] for values in buckets],
        }
        for idx, key in enumerate(("x_cm", "y_cm", "angle_deg")):
            curves[profile][key + "_sd"] = [math.sqrt(max(0.0, values[5 + idx]) / (n - 1)) for values in buckets] if n > 1 else None
            # Linear interpolation between ranked observations, separately at each step.
            samples = [sorted(values[8 + idx]) for values in buckets]
            for percentile in (1, 99):
                rank = (n - 1) * percentile / 100.0
                lower, upper = math.floor(rank), math.ceil(rank)
                fraction = rank - lower
                curves[profile][key + f"_p{percentile}"] = [
                    values[lower] + fraction * (values[upper] - values[lower])
                    for values in samples
                ]
    return curves


def plot_paper_lm_report(csv_path, json_path, open_pdf=True):
    """Make a settings report and a paper figure from the saved P-LM dataset."""
    import json
    import os
    from matplotlib.backends.backend_pdf import PdfPages

    with open(json_path, encoding="utf-8") as json_file:
        metadata = json.load(json_file)
    curves = paper_lm_mean_errors(csv_path, metadata)
    if not curves:
        return {}
    # Older datasets may be longer; show their first thirty seconds without editing the CSV.
    for curve in curves.values():
        sample_count = sum(time_s <= PAPER_LM_TRIAL_DURATION_S + 1e-9 for time_s in curve["time_s"])
        for key, values in curve.items():
            if isinstance(values, list):
                curve[key] = values[:sample_count]
    settings = metadata["settings"]
    profile_counts = paper_lm_profile_counts(metadata)
    target_times = sorted({
        step * settings["dt"]
        for trial in metadata["trials"]
        if trial["completed"] and trial["landmark_profile"] in curves
        for step in trial.get("target_activation_steps", [])
        if step * settings["dt"] <= PAPER_LM_TRIAL_DURATION_S + 1e-9
    }) if settings["demo_path_name"] == "Random targets" else []
    counts = ", ".join(f"LM {profile}: n={curves[profile]['n'] if profile in curves else 0}" for profile in curves)
    simulation_lines = simulation_settings_report_lines(settings)[:-1]
    header_lines = [
        "P-LM Experiment Report - Mean absolute tracking error",
        f"Source: {os.path.basename(csv_path)}",
        f"Algorithm: {metadata['algorithm_name']}   Path: {settings['demo_path_name']}   Profile: {settings['profile_name']}",
        f"Trial duration: {metadata['requested_trial_duration_s']:g} s   DT={settings['dt']:g} s   Requested N={metadata['runs_per_profile']} per LM profile",
        f"Displayed time window: 0-{max(curve['time_s'][-1] for curve in curves.values()):g} s",
        f"Completed trials used: {counts}   Batch {'STOPPED' if metadata['stopped'] else 'COMPLETE'}",
        format_ros_pose_line("Start actual", metadata["actual_start"]) + "   " + format_ros_pose_line("Start belief", metadata["belief_start"]),
        f"Ball (FCS): x={metadata['ball']['x_cm']:g} cm, y={metadata['ball']['y_cm']:g} cm; identical starts and path in every trial",
        f"Noise: {'YES' if settings['noise_enabled'] else 'NO'}   landmark sigma x={settings['noise_x_cm']:g}, y={settings['noise_y_cm']:g} cm   motion={100 * settings['motion_noise_ratio']:g}%",
        "   |   ".join(simulation_lines[:2]),
        "   |   ".join(simulation_lines[2:4]),
        "   |   ".join(simulation_lines[4:]),
        precision_report_row(settings) + "   " + precision_distance_modifier_report_row(settings),
        learning_rate_report_row(settings),
        "Profiles: " + "; ".join(f"{p['name']} = {', '.join(row['name'] for row in p['landmarks']) if 'landmarks' in p else ', '.join(p['landmark_ids'])}" for p in metadata["landmark_profiles"] if p["name"] in curves),
        "Mean of |belief - actual| at each time; heading difference wrapped to [-pi, pi). No smoothing. Time zero included.",
        "Shading: P1-P99 across trials at each timestep (middle 98%; linear percentile interpolation). No clipping; no band for n<2.",
    ]
    header = "\n".join(textwrap.fill(line, width=130, break_long_words=False, break_on_hyphens=False) for line in header_lines)
    if target_times:
        header += "\nThin red vertical lines: next random target activated (recorded simulation step)."
    base_path = os.path.splitext(csv_path)[0]
    paths = {"report_path": base_path + "_report.pdf", "paper_path": base_path + "_graph.pdf"}
    colors = {"1": "tab:blue", "2": "tab:orange", "3": "tab:red", "5": "tab:green", "9": "tab:purple"}
    styles = {"1": "-", "2": "--", "3": (0, (3, 1, 1, 1)), "5": "-.", "9": ":"}
    panels = (("x_cm", "X error (cm)"), ("y_cm", "Y error (cm)"), ("angle_deg", "Heading error (deg)"))
    for paper_ready, output_path in ((False, paths["report_path"]), (True, paths["paper_path"])):
        pages = [[panel] for panel in panels] if paper_ready else [panels]
        with PdfPages(output_path) as pdf:
            for page_panels in pages:
                fig = plt.figure(figsize=(15, 3) if paper_ready else (15, 10))
                grid = fig.add_gridspec(1 if paper_ready else 4, 1, height_ratios=[1] if paper_ready else [2.5, 1, 1, 1])
                # Match each standalone graph to the report panel's physical dimensions.
                fig.subplots_adjust(left=0.08, right=0.98,
                                    top=(0.55 + report_panel_height) / 3 if paper_ready else 0.97,
                                    bottom=0.55 / 3 if paper_ready else 0.09, hspace=0.35)
                offset = 0 if paper_ready else 1
                if paper_ready:
                    component = {"x_cm": "X", "y_cm": "Y", "angle_deg": "heading"}[page_panels[0][0]]
                    counts = {curve["n"] for curve in curves.values()}
                    counts_text = f"N={next(iter(counts))}" if len(counts) == 1 else ", ".join(f"{profile} LM: N={curve['n']}" for profile, curve in curves.items())
                    fig.suptitle(f"Absolute {component} error over time by landmark count\nLine: mean | Shading: P1-P99 | " + counts_text, fontsize=10, y=0.99)
                else:
                    header_axis = fig.add_subplot(grid[0, 0])
                    header_axis.axis("off")
                    header_axis.text(0, 1, header, fontsize=9, family="monospace", va="top", linespacing=1.4)
                first_axis = None
                for idx, (key, label) in enumerate(page_panels):
                    axis = fig.add_subplot(grid[idx + offset, 0], sharex=first_axis)
                    if first_axis is None:
                        first_axis = axis
                    for profile, curve in curves.items():
                        if curve["n"] > 1:
                            axis.fill_between(curve["time_s"], curve[key + "_p1"], curve[key + "_p99"],
                                              color=colors[str(profile_counts[profile])], alpha=0.16, linewidth=0)
                        legend_label = f"{profile} LM" if paper_ready else f"LM {profile} (n={curve['n']})"
                        axis.plot(curve["time_s"], curve[key], label=legend_label, color=colors[str(profile_counts[profile])], linestyle=styles[str(profile_counts[profile])], linewidth=1.3)
                    axis.set_ylabel(label, fontsize=10)
                    for event_idx, time_s in enumerate(target_times):
                        axis.axvline(time_s, color="red", linewidth=0.6, alpha=0.65,
                                     label=("Target" if paper_ready else "Next target") if event_idx == 0 else "_nolegend_")
                    axis.set_ylim(bottom=0)
                    axis.grid(True, alpha=0.25)
                    if paper_ready:
                        axis.tick_params(labelsize=10)
                    if idx == 0:
                        axis.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4 if target_times else 3,
                                    fontsize=8, borderaxespad=0,
                                    columnspacing=2.0, handlelength=2.0)
                    if paper_ready or idx == 2:
                        axis.set_xlabel("Simulation time (s)", fontsize=10)
                if not paper_ready:
                    report_panel_height = axis.get_position().height * fig.get_figheight()
                pdf.savefig(fig)
                plt.close(fig)
    if open_pdf:
        os.startfile(paths["report_path"])
    return paths


def plot_experiment(history, algorithm_name="Unknown", paper_ready=False, open_pdf=True):
    if not history:
        return

    import datetime
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec

    steps = [h["step"] for h in history]

    def gc_value(record, state_name, axis_name, order_idx, pose_key=None):
        gc_key = f"{state_name}_gc"
        if gc_key in record:
            axis_series = record[gc_key].get(axis_name, [])
            if len(axis_series) > order_idx:
                return axis_series[order_idx]
        if order_idx == 0 and pose_key is not None:
            return record[f"{state_name}_pose"][pose_key]
        return 0.0

    def observed_motion_fcs(record):
        motion_obs = record.get("motion_obs")
        belief_gc = record.get("belief_gc")
        if motion_obs is None or belief_gc is None:
            return {
                "x_vel": 0.0,
                "y_vel": 0.0,
                "theta_vel": 0.0,
                "x_acc": 0.0,
                "y_acc": 0.0,
                "theta_acc": 0.0,
            }
        obs_x_vel, obs_y_vel, obs_theta_vel = vel_rcs_2_fcs(
            belief_gc,
            motion_obs["x_vel_cm_s"],
            motion_obs["y_vel_cm_s"],
            motion_obs["theta_vel_rad_s"],
        )
        obs_x_acc, obs_y_acc, obs_theta_acc = acc_rcs_2_fcs(
            belief_gc,
            motion_obs["x_acc_cm_s2"],
            motion_obs["y_acc_cm_s2"],
            motion_obs["theta_acc_rad_s2"],
        )
        return {
            "x_vel": obs_x_vel,
            "y_vel": obs_y_vel,
            "theta_vel": obs_theta_vel,
            "x_acc": obs_x_acc,
            "y_acc": obs_y_acc,
            "theta_acc": obs_theta_acc,
        }

    actual_x = [gc_value(h, "actual", "x", 0, "x_cm") for h in history]
    actual_y = [gc_value(h, "actual", "y", 0, "y_cm") for h in history]
    actual_theta = [gc_value(h, "actual", "theta", 0, "theta_rad") for h in history]
    actual_x_vel = [gc_value(h, "actual", "x", 1) for h in history]
    actual_y_vel = [gc_value(h, "actual", "y", 1) for h in history]
    actual_theta_vel = [gc_value(h, "actual", "theta", 1) for h in history]
    actual_x_acc = [gc_value(h, "actual", "x", 2) for h in history]
    actual_y_acc = [gc_value(h, "actual", "y", 2) for h in history]
    actual_theta_acc = [gc_value(h, "actual", "theta", 2) for h in history]

    observed_motion = [observed_motion_fcs(h) for h in history]
    observed_x_vel = [obs["x_vel"] for obs in observed_motion]
    observed_y_vel = [obs["y_vel"] for obs in observed_motion]
    observed_theta_vel = [obs["theta_vel"] for obs in observed_motion]
    observed_x_acc = [obs["x_acc"] for obs in observed_motion]
    observed_y_acc = [obs["y_acc"] for obs in observed_motion]
    observed_theta_acc = [obs["theta_acc"] for obs in observed_motion]

    belief_x = [gc_value(h, "belief", "x", 0, "x_cm") for h in history]
    belief_y = [gc_value(h, "belief", "y", 0, "y_cm") for h in history]
    belief_theta = [gc_value(h, "belief", "theta", 0, "theta_rad") for h in history]
    belief_x_vel = [gc_value(h, "belief", "x", 1) for h in history]
    belief_y_vel = [gc_value(h, "belief", "y", 1) for h in history]
    belief_theta_vel = [gc_value(h, "belief", "theta", 1) for h in history]
    belief_x_acc = [gc_value(h, "belief", "x", 2) for h in history]
    belief_y_acc = [gc_value(h, "belief", "y", 2) for h in history]
    belief_theta_acc = [gc_value(h, "belief", "theta", 2) for h in history]

    free_energy = [h["free_energy"] for h in history]
    free_energy_steps = steps[1:] if len(steps) > 1 else steps
    free_energy_plot = free_energy[1:] if len(free_energy) > 1 else free_energy

    start_actual = history[0]["actual_pose"]
    start_belief = history[0]["belief_pose"]
    end_actual = history[-1]["actual_pose"]
    end_belief = history[-1]["belief_pose"]
    settings = history[0].get("settings", experiment_settings(algorithm_name))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_prefix = "paper_experiment" if paper_ready else "last_experiment_report"
    pdf_path = os.path.join(reports_dir, f"{report_prefix}_{timestamp_file}.pdf")

    if settings["noise_enabled"]:
        noise_text = (
            f"Noise: YES   landmarks x={settings['noise_x_cm']:.2f} cm   "
            f"y={settings['noise_y_cm']:.2f} cm   angle={settings['noise_angle_rad']:.4f} rad   "
            f"motion={settings.get('motion_noise_ratio', MOTION_NOISE_RATIO) * 100.0:.1f}%"
        )
    else:
        noise_text = "Noise: NO"

    profile_text = settings.get("profile_name", "manual/unknown")
    path_text = settings.get("demo_path_name", "n/a")
    association_text = (
        f"Association={settings.get('data_association', active_association_name())} "
        f"gate={settings.get('landmark_association_gate_cm', LANDMARK_ASSOCIATION_GATE_CM):.0f}cm"
    )
    landmark_profile_text = (
        f"LM profile={settings.get('landmark_scenario_profile', ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME)}"
    )
    if settings["viewing_angle_enabled"]:
        viewing_text = f"Viewing angle: ON   fov={settings['viewing_angle_deg']:.1f} deg"
    else:
        viewing_text = f"Viewing angle: OFF   fov={settings['viewing_angle_deg']:.1f} deg"
    noise_text = textwrap.fill(
        noise_text,
        width=82,
        break_long_words=False,
        break_on_hyphens=False,
    )
    viewing_text = textwrap.fill(
        viewing_text,
        width=82,
        break_long_words=False,
        break_on_hyphens=False,
    )
    hyper_text = (
        f"Profile={profile_text}   Path={path_text}   DT={settings['dt']:.4f}\n"
        f"{precision_report_row(settings)}\n"
        f"{precision_distance_modifier_report_row(settings)}\n"
        f"{learning_rate_report_row(settings)}\n"
        f"{association_text}\n"
        f"{landmark_profile_text}\n"
        f"{viewing_text}\n"
    )

    start_text = (
        "Start (ROS/FCS)\n"
        f"{format_ros_pose_line('Actual', start_actual)}\n"
        f"{format_ros_pose_line('Belief', start_belief)}"
    )

    final_text = (
        "Final (ROS/FCS)\n"
        f"{format_ros_pose_line('Actual', end_actual)}\n"
        f"{format_ros_pose_line('Belief', end_belief)}"
    )

    if paper_ready:
        fig_pos = plt.figure(figsize=(7.2, 5.8))
        gs_pos = GridSpec(4, 1, figure=fig_pos)
        fig_pos.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.11, hspace=0.35)
        pos_row_offset = 0
    else:
        fig_pos = plt.figure(figsize=(15, 10))
        gs_pos = GridSpec(5, 1, figure=fig_pos, height_ratios=[2.9, 1.0, 1.0, 1.0, 1.0])
        fig_pos.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.45)
        pos_row_offset = 1

        ax0 = fig_pos.add_subplot(gs_pos[0, 0])
        ax0.axis("off")
        ax0.text(0.01, 0.97, f"Experiment time: {timestamp}", fontsize=10, family="monospace", va="top")
        ax0.text(0.52, 0.97, f"Algorithm: {algorithm_name}", fontsize=10, family="monospace", va="top")
        ax0.text(0.01, 0.81, start_text, fontsize=9, family="monospace", va="top", linespacing=1.4)
        ax0.text(0.52, 0.81, final_text, fontsize=9, family="monospace", va="top", linespacing=1.4)
        ax0.text(0.01, 0.43, noise_text, fontsize=9, family="monospace", va="top", linespacing=1.4)
        ax0.text(0.01, 0.25, hyper_text, fontsize=9, family="monospace", va="top", linespacing=1.35)

    ax1 = fig_pos.add_subplot(gs_pos[pos_row_offset, 0])
    ax1.plot(steps, belief_x, label="Belief x", color="tab:orange")
    ax1.plot(steps, actual_x, label="Actual x", color="tab:green")
    ax1.set_ylabel("x forward (cm)")
    ax1.legend(loc="center right")

    ax2 = fig_pos.add_subplot(gs_pos[pos_row_offset + 1, 0], sharex=ax1)
    ax2.plot(steps, belief_y, label="Belief y", color="tab:orange")
    ax2.plot(steps, actual_y, label="Actual y", color="tab:green")
    ax2.set_ylabel("y left (cm)")
    ax2.legend(loc="center right")

    ax3 = fig_pos.add_subplot(gs_pos[pos_row_offset + 2, 0], sharex=ax1)
    ax3.plot(steps, belief_theta, label="Belief theta", color="tab:orange")
    ax3.plot(steps, actual_theta, label="Actual theta", color="tab:green")
    ax3.set_ylabel("theta (rad)")
    ax3.legend(loc="center right")

    ax4 = fig_pos.add_subplot(gs_pos[pos_row_offset + 3, 0], sharex=ax1)
    ax4.plot(free_energy_steps, free_energy_plot, label="Free Energy")
    ax4.set_ylabel("F")
    ax4.set_xlabel("step")
    ax4.legend(loc="center right")

    fig_vel = plt.figure(figsize=(7.2, 4.6) if paper_ready else (15, 8))
    gs_vel = GridSpec(3, 1, figure=fig_vel)
    fig_vel.subplots_adjust(
        left=0.12 if paper_ready else 0.08,
        right=0.98,
        top=0.98 if paper_ready else 0.94,
        bottom=0.12 if paper_ready else 0.08,
        hspace=0.35 if paper_ready else 0.4,
    )
    if not paper_ready:
        fig_vel.suptitle(f"Velocity - {algorithm_name}", fontsize=12)

    ax5 = fig_vel.add_subplot(gs_vel[0, 0])
    if settings["noise_enabled"]:
        ax5.plot(steps, observed_x_vel, label="Observations transformed into FCS\nusing belief heading", color="tab:blue")
    ax5.plot(steps, belief_x_vel, label="Belief x dot", color="tab:orange")
    ax5.plot(steps, actual_x_vel, label="Actual x dot", color="tab:green")
    ax5.set_ylabel("x vel (cm/s)")
    ax5.legend(loc="center right")

    ax6 = fig_vel.add_subplot(gs_vel[1, 0], sharex=ax5)
    if settings["noise_enabled"]:
        ax6.plot(steps, observed_y_vel, label="Observations transformed into FCS\nusing belief heading", color="tab:blue")
    ax6.plot(steps, belief_y_vel, label="Belief y dot", color="tab:orange")
    ax6.plot(steps, actual_y_vel, label="Actual y dot", color="tab:green")
    ax6.set_ylabel("y vel (cm/s)")
    ax6.legend(loc="center right")

    ax7 = fig_vel.add_subplot(gs_vel[2, 0], sharex=ax5)
    if settings["noise_enabled"]:
        ax7.plot(steps, observed_theta_vel, label="Observation", color="tab:blue")
    ax7.plot(steps, belief_theta_vel, label="Belief theta dot", color="tab:orange")
    ax7.plot(steps, actual_theta_vel, label="Actual theta dot", color="tab:green")
    ax7.set_ylabel("theta vel (rad/s)")
    ax7.set_xlabel("step")
    ax7.legend(loc="center right")

    fig_acc = plt.figure(figsize=(7.2, 4.6) if paper_ready else (15, 8))
    gs_acc = GridSpec(3, 1, figure=fig_acc)
    fig_acc.subplots_adjust(
        left=0.12 if paper_ready else 0.08,
        right=0.98,
        top=0.98 if paper_ready else 0.94,
        bottom=0.12 if paper_ready else 0.08,
        hspace=0.35 if paper_ready else 0.4,
    )
    if not paper_ready:
        fig_acc.suptitle(f"Acceleration - {algorithm_name}", fontsize=12)

    ax8 = fig_acc.add_subplot(gs_acc[0, 0])
    if settings["noise_enabled"]:
        ax8.plot(steps, observed_x_acc, label="Observations transformed into FCS\nusing belief heading", color="tab:blue")
    ax8.plot(steps, belief_x_acc, label="Belief x ddot", color="tab:orange")
    ax8.plot(steps, actual_x_acc, label="Actual x ddot", color="tab:green")
    ax8.set_ylabel("x acc (cm/s^2)")
    ax8.legend(loc="center right")

    ax9 = fig_acc.add_subplot(gs_acc[1, 0], sharex=ax8)
    if settings["noise_enabled"]:
        ax9.plot(steps, observed_y_acc, label="Observations transformed into FCS\nusing belief heading", color="tab:blue")
    ax9.plot(steps, belief_y_acc, label="Belief y ddot", color="tab:orange")
    ax9.plot(steps, actual_y_acc, label="Actual y ddot", color="tab:green")
    ax9.set_ylabel("y acc (cm/s^2)")
    ax9.legend(loc="center right")

    ax10 = fig_acc.add_subplot(gs_acc[2, 0], sharex=ax8)
    if settings["noise_enabled"]:
        ax10.plot(steps, observed_theta_acc, label="Observation", color="tab:blue")
    ax10.plot(steps, belief_theta_acc, label="Belief theta ddot", color="tab:orange")
    ax10.plot(steps, actual_theta_acc, label="Actual theta ddot", color="tab:green")
    ax10.set_ylabel("theta acc (rad/s^2)")
    ax10.set_xlabel("step")
    ax10.legend(loc="center right")

    for axis in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10):
        axis.grid(True, alpha=0.25)

    with PdfPages(pdf_path) as pdf:
        for fig in (fig_pos, fig_vel, fig_acc):
            pdf.savefig(fig, bbox_inches="tight" if paper_ready else None)
            plt.close(fig)

    if open_pdf:
        os.startfile(pdf_path)
    return pdf_path


def export_robustness_dataset(config, algorithm_name, initialize_path, step_path,
                              settings, progress_callback=None, output_dir="reports"):
    """Stream trials/points to disk; settings_id links every row to its run metadata."""
    import csv

    robustness_axes(config)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    base = Path(output_dir).resolve() / f"robustness_{timestamp}"
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: str(base) + suffix for name, suffix in (
        ("trials_path", "_trials.csv"), ("summary_path", "_points.csv"),
        ("metadata_path", ".json"), ("report_path", "_coarse.pdf"),
    )}
    metadata = {
        "settings_id": base.name, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "algorithm_name": algorithm_name, "settings": deepcopy(settings), "sweep": dict(config),
        "start_domain_fcs_cm": {"half_x": FIELD_LENGTH_CM / 2 + 80.0, "half_y": FIELD_WIDTH_CM / 2 + 80.0},
        "convergence": {"x_cm": CONV_TOL_X_CM, "y_cm": CONV_TOL_Y_CM,
                        "angle_deg": CONV_TOL_ANGLE_DEG, "sequential_hits": CONV_REQUIRED_SEQUENTIAL_HITS,
                        "max_steps": INFER_STEPS_MAX, "symmetric_is_failure": True},
        "sampling": "Uniform actual starts in the intersection where actual and exact-offset belief both fit; shared trial quantiles across offsets",
        "search_strategy": "angle_spiral_first_failure",
        "search": "For each heading offset, visit the XY grid in widening square spiral rings from zero. A cell is tested only if its neighbor one index toward zero in each nonzero coordinate passed. Otherwise it is skipped, propagating failed/skipped inner cells outward. Stop a tested cell at its first failed trial.",
        "status": "running", **paths,
    }
    # Demo specs contain Python sampler functions; save their names at the
    # reporting boundary while the playback adapter keeps the callable spec.
    metadata["settings"]["demo_path"] = {
        key: value.__name__ if callable(value) else value
        for key, value in settings["demo_path"].items()
    }

    def save_metadata():
        Path(paths["metadata_path"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_metadata()
    trial_fields = ["settings_id", "point_id", "phase", "trial_id", "seed", "dx_cm", "dy_cm", "da_deg",
                    "outcome", "steps", "elapsed_s"]
    for pose_name in ("actual_start", "belief_start", "actual_end", "belief_end"):
        trial_fields.extend(f"{pose_name}_{axis}" for axis in ("x_cm", "y_cm", "theta_rad"))
    point_fields = ["settings_id", "point_id", "phase", "dx_cm", "dy_cm", "da_deg", "completed",
                    "correct", "symmetric", "timeout", "convergence_pct", "status",
                    "blocked_by_dx_cm", "blocked_by_dy_cm"]
    saved_random_state = random.getstate()
    try:
        with open(paths["trials_path"], "w", newline="", encoding="utf-8") as trials_file, open(
            paths["summary_path"], "w", newline="", encoding="utf-8",
        ) as points_file:
            trials_writer = csv.DictWriter(trials_file, fieldnames=trial_fields)
            points_writer = csv.DictWriter(points_file, fieldnames=point_fields)
            trials_writer.writeheader()
            points_writer.writeheader()

            def record_trial(result):
                row = {key: result[key] for key in trial_fields if key in result}
                row["settings_id"] = base.name
                for pose_name in ("actual_start", "belief_start", "actual_end", "belief_end"):
                    row.update({f"{pose_name}_{key}": value for key, value in result[pose_name].items()})
                trials_writer.writerow(row)
                trials_file.flush()

            def record_point(point):
                points_writer.writerow(dict(point, settings_id=base.name))
                metadata["completed_points"] = point["point_id"]
                # Skips run quickly and have no sensor trials to preserve.
                # Batch their flushes; measured cells still persist immediately.
                if point["status"] != "skipped" or point["point_id"] % 100 == 0:
                    points_file.flush()
                    save_metadata()

            def trial_runner(actual_start, offset, seed, progress):
                random.seed(seed)
                return run_robustness_trial(actual_start, offset, algorithm_name,
                                            initialize_path, step_path, progress)

            result = run_robustness_sweep(config, trial_runner, record_trial, record_point, progress_callback)
        metadata.update({key: value for key, value in result.items() if key != "points"})
        metadata["status"] = "cancelled" if result["stopped"] else "complete"
        save_metadata()
    except Exception:
        metadata["status"] = "error"
        save_metadata()
        raise
    finally:
        random.setstate(saved_random_state)
        reset_landmark_obstruction_state()
        reset_motion_slip_state()
    return dict(result, **paths)


def scan_report_condition_lines(metadata):
    """Describe the test conditions saved with this scan, not current GUI state."""
    settings = metadata["settings"]
    convergence = metadata["convergence"]
    return [
        f"Algorithm: {metadata['algorithm_name']} | {settings['profile_name']} | Path: {settings['demo_path']['name']}",
        convergence_condition_report_row(convergence["x_cm"], convergence["y_cm"],
                                         math.radians(convergence["angle_deg"]), convergence["sequential_hits"]),
        f"Ball position (FCS): x={settings['ball_fcs']['x_cm']:.1f} cm   y={settings['ball_fcs']['y_cm']:.1f} cm",
        *convergence_report_condition_lines(settings),
    ]


def export_robustness_angle_view(metadata_path):
    """Save an offline angle slider over measured coarse dx/dy cells only."""
    import csv

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    config = metadata["sweep"]
    multiplier = config.get("coarse_multiplier", 1)
    with open(metadata["summary_path"], newline="", encoding="utf-8") as stream:
        points = [
            [float(row["dx_cm"]), float(row["dy_cm"]), float(row["da_deg"]),
             int(row["correct"]), int(row["completed"]), int(row["symmetric"]), int(row["timeout"]), row["status"]]
            for row in csv.DictReader(stream)
            if row.get("phase", "coarse") == "coarse" and row["status"] in {"pass", "fail"}
        ]
    data = {
        "points": points, "stepX": config["step_x_cm"] * multiplier,
        "stepY": config["step_y_cm"] * multiplier, "runs": config["runs"],
        "run": metadata["settings_id"], "algorithm": metadata["algorithm_name"],
        "landmarks": metadata["settings"]["landmark_scenario_profile"],
        "conditions": scan_report_condition_lines(metadata),
    }
    report_path = Path(metadata["report_path"])
    stem = report_path.stem.removesuffix("_coarse")
    output_path = report_path.with_name(stem + "_coarse_interactive.html")
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robustness: angle slices</title>
<style>
:root { color-scheme: light dark; --bg:light-dark(#fff,#171b21); --ink:light-dark(#192633,#edf2f7);
 --muted:light-dark(#536171,#b4c0cc); --grid:light-dark(#d6dce2,#414b58); --unknown:light-dark(#f0f3f6,#252d37);
 --pass:light-dark(#65bb7d,#4eaa6a); --fail:light-dark(#a3453f,#f08e86); }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--ink); font:16px system-ui,sans-serif; }
main { max-width:1040px; margin:24px auto; padding:0 20px 24px; } h1 { font-size:24px; font-weight:600; margin:0 0 8px; }
.secondary { color:var(--muted); font-size:14px; } .control { margin:24px 0 12px; } .row { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
label { font-size:19px; } output { font-variant-numeric:tabular-nums; font-weight:600; } input { width:100%; margin:14px 0 0; accent-color:var(--pass); min-height:32px; }
.ends { display:flex; justify-content:space-between; } .legend { display:flex; gap:20px; flex-wrap:wrap; margin:14px 0 8px; font-size:14px; }
.key { display:inline-block; width:14px; height:14px; vertical-align:-2px; margin-right:6px; } .green { background:var(--pass); }
.gray { background:var(--unknown); border:1px solid var(--grid); } .cross { color:var(--fail); font-weight:600; }
svg { display:block; width:100%; touch-action:pan-y; } svg text { fill:var(--ink); font:13px system-ui,sans-serif; }
.detail { min-height:44px; margin:8px 0; font-variant-numeric:tabular-nums; } .note { max-width:78ch; line-height:1.5; }
</style></head><body><main id="robustness-angle-view">
<h1>Where does every trial converge?</h1>
<div id="settings" class="secondary"></div>
<div class="secondary" style="margin-top:12px;line-height:1.6"><strong>Test conditions</strong><div id="test-conditions"></div></div>
<div class="control"><div class="row"><label for="angle">FCS heading offset Δa: <output id="angle-value">0°</output></label>
<span id="slice-count" class="secondary" aria-live="polite"></span></div>
<input id="angle" type="range" min="0" step="1" aria-label="FCS heading offset in degrees">
<div class="ends secondary"><span id="angle-min"></span><span>0°</span><span id="angle-max"></span></div></div>
<div class="legend"><span><i class="key green"></i>100% correct</span><span><b class="cross">×</b> Below 100% (symmetric = fail)</span>
<span><i class="key gray"></i>Skipped / untested</span></div>
<div id="plot"><svg id="map" role="img" aria-label="Convergence map: FCS delta x horizontal and delta y vertical, zero at the center"></svg></div>
<div id="detail" class="detail" aria-live="polite">Hover or tap a tested cell for its result.</div>
<div class="secondary note">The center is zero position error: Δx = Δy = 0. Solid green cells passed all N trials; failed cells stop at the first failure.
Gray means not tested: once an inner cell fails, cells farther out behind it are skipped rather than tested and found to fail.
Offsets are relative to each trial’s actual start, rather than absolute positions on the soccer field.</div>
<div id="run" class="secondary" style="margin-top:16px"></div>
</main><script id="robustness-data" type="application/json">__ROBUSTNESS_DATA__</script>
<script>
(() => {
 const root=document.getElementById('robustness-angle-view');
 const data=JSON.parse(document.getElementById('robustness-data').textContent);
 const get=id=>root.querySelector('#'+id), svg=get('map'), slider=get('angle');
 const angles=[...new Set(data.points.map(p=>p[2]))].sort((a,b)=>a-b);
 const xs=[...new Set(data.points.map(p=>p[0]))].sort((a,b)=>a-b);
 const ys=[...new Set(data.points.map(p=>p[1]))].sort((a,b)=>a-b);
 const halfX=Math.max(data.stepX,...xs.map(Math.abs))+data.stepX;
 const halfY=Math.max(data.stepY,...ys.map(Math.abs))+data.stepY;
 const bounds=(values,step)=>new Map(values.map((v,i)=>[v,[
   v-Math.min(step/2,i ? (v-values[i-1])/2 : step/2),
   v+Math.min(step/2,i<values.length-1 ? (values[i+1]-v)/2 : step/2)]]));
 const xb=bounds(xs,data.stepX), yb=bounds(ys,data.stepY);
 const format=value=>(value>0?'+':'')+value;
 get('settings').textContent=`Up to ${data.runs} random-start trials per offset · ${data.stepX} × ${data.stepY} cm cells`;
 for(const line of data.conditions) {
   const row=document.createElement('div');row.textContent=line;get('test-conditions').append(row);
 }
 get('run').textContent=data.run;
 slider.max=Math.max(0,angles.length-1); slider.value=Math.max(0,angles.indexOf(0));
 slider.disabled=angles.length<2;
 get('angle-min').textContent=angles.length?format(angles[0])+'°':'No completed samples';
 get('angle-max').textContent=angles.length?format(angles[angles.length-1])+'°':'';
 const ns='http://www.w3.org/2000/svg';
 function add(tag,attrs,text,parent=svg) {
   const el=document.createElementNS(ns,tag);
   for(const [key,value] of Object.entries(attrs))el.setAttribute(key,value);
   if(text!==undefined)el.textContent=text;parent.append(el);return el;
 }
 function draw() {
   const angle=angles[Number(slider.value)], selected=data.points.filter(p=>p[2]===angle);
   get('angle-value').textContent=angle===undefined?'—':format(angle)+'°';
   slider.setAttribute('aria-valuetext',angle===undefined?'No samples':`${angle} degrees`);
   get('slice-count').textContent=`${selected.filter(p=>p[7]==='pass').length} of ${selected.length} tested offsets at 100%`;
   get('detail').textContent='Hover or tap a tested cell for its result.';
   const width=Math.max(300,get('plot').clientWidth), height=width<500?510:650;
   const margin={left:66,right:22,top:24,bottom:52};
   const left=margin.left,top=margin.top,w=width-left-margin.right,h=height-top-margin.bottom;
   const cx=left+w/2,cy=top+h/2,scale=Math.min(w/(2*halfX),h/(2*halfY));
   const X=x=>cx+x*scale,Y=y=>cy-y*scale;
   svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.replaceChildren();
   add('title',{},`Correct convergence at heading offset ${angle} degrees`);
   add('rect',{x:left,y:top,width:w,height:h,fill:'var(--unknown)',stroke:'var(--grid)'});
   // Render only measured cells. Missing conditions retain the unknown fill.
   for(const p of selected) {
     const [loX,hiX]=xb.get(p[0]),[loY,hiY]=yb.get(p[1]),pass=p[7]==='pass';
     const rect=add('rect',{x:X(loX),y:Y(hiY),width:(hiX-loX)*scale,height:(hiY-loY)*scale,
       fill:pass?'var(--pass)':'var(--bg)','data-dx':p[0],'data-dy':p[1],'data-pass':pass});
     const verdict=pass?`100% (${p[3]}/${p[4]} correct)`:`Below 100%; ${p[4]} trials attempted; ${p[5]} symmetric; ${p[6]} fail`;
     const detail=`Δx ${format(p[0])} cm · Δy ${format(p[1])} cm · Δa ${format(p[2])}°: ${verdict}.`;
     add('title',{},detail,rect);
     rect.addEventListener('pointerenter',()=>get('detail').textContent=detail);
     rect.addEventListener('click',()=>get('detail').textContent=detail);
     if(!pass) {
       const r=Math.max(1,Math.min(3,data.stepX*scale/3));
       add('path',{d:`M${X(p[0])-r},${Y(p[1])-r}l${2*r},${2*r}m${-2*r},0l${2*r},${-2*r}`,
         stroke:'var(--fail)','stroke-width':1.2,'pointer-events':'none'});
     }
   }
   const tick=width<500?1000:500;
   for(let x=Math.ceil(-halfX/tick)*tick;x<=halfX;x+=tick) {
     add('line',{x1:X(x),y1:top,x2:X(x),y2:top+h,stroke:'var(--grid)','stroke-width':0.5,'pointer-events':'none'});
     add('text',{x:X(x),y:top+h+22,'text-anchor':'middle'},String(x));
   }
   for(let y=Math.ceil(-halfY/tick)*tick;y<=halfY;y+=tick) {
     add('line',{x1:left,y1:Y(y),x2:left+w,y2:Y(y),stroke:'var(--grid)','stroke-width':0.5,'pointer-events':'none'});
     add('text',{x:left-10,y:Y(y)+4,'text-anchor':'end'},String(y));
   }
   add('line',{x1:left,y1:cy,x2:left+w,y2:cy,stroke:'var(--ink)','stroke-width':1,'pointer-events':'none'});
   add('line',{x1:cx,y1:top,x2:cx,y2:top+h,stroke:'var(--ink)','stroke-width':1,'pointer-events':'none'});
   add('circle',{cx,cy,r:4,fill:'var(--bg)',stroke:'var(--ink)','pointer-events':'none'});
   add('text',{x:cx+8,y:cy-10},'(0, 0)');
   add('text',{x:cx,y:height-8,'text-anchor':'middle','data-axis':'x'},'FCS Δx (cm)');
   add('text',{transform:`translate(17 ${cy}) rotate(-90)`,'text-anchor':'middle','data-axis':'y'},'FCS Δy (cm)');
 }
 slider.addEventListener('input',draw);
 new ResizeObserver(draw).observe(get('plot'));draw();
})();
</script></body></html>'''
    output_path.write_text(template.replace("__ROBUSTNESS_DATA__", json.dumps(data).replace("<", "\\u003c")), encoding="utf-8")
    return str(output_path)


def plot_robustness_report(metadata_path, open_pdf=True, show_plot=True):
    """Report coarse samples only, including from older mixed-resolution datasets."""
    import csv
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.cm import ScalarMappable

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    with open(metadata["summary_path"], newline="", encoding="utf-8") as stream:
        saved_points = list(csv.DictReader(stream))
    points = [point for point in saved_points if point.get("phase", "coarse") == "coarse"]
    report_status = "coarse search complete" if any(p.get("phase") == "fine" for p in saved_points) else metadata["status"]
    report_path = Path(metadata["report_path"])
    if not report_path.stem.endswith("_coarse"):
        report_path = report_path.with_name(report_path.stem + "_coarse.pdf")
    complete = [point for point in points if point["status"] in {"pass", "fail"}]
    skipped = sum(point["status"] == "skipped" for point in points)
    incomplete = sum(point["status"] == "incomplete" for point in points)
    passed = sum(point["status"] == "pass" for point in complete)
    config, settings = metadata["sweep"], metadata["settings"]
    norm = Normalize(0, 1)
    cmap = ListedColormap(["#b9574f", "#53aa70"])
    fig = plt.figure(figsize=(11.7, 8.3))
    ax = fig.add_subplot(111, projection="3d")
    styles = (
        ("coarse", "pass", "o", "Coarse: 100% correct", 14),
        ("coarse", "fail", "x", "Coarse: below 100%", 24),
    )
    for phase, status, marker, label, size in styles:
        group = [point for point in complete if point["status"] == status and point.get("phase", "coarse") == phase]
        if group:
            ax.scatter([float(p["dx_cm"]) for p in group], [float(p["dy_cm"]) for p in group],
                       [float(p["da_deg"]) for p in group], c=[int(p["status"] == "pass") for p in group],
                       cmap=cmap, norm=norm, marker=marker, s=size, alpha=0.8, label=label)
    if complete:
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text2D(0.1, 0.9, "No completed grid points", transform=ax.transAxes)
    ax.set(xlabel="FCS delta x (cm)", ylabel="FCS delta y (cm)", zlabel="FCS delta a (deg)")
    fig.suptitle(f"Coarse robustness: {passed}/{len(complete)} tested points at 100% correct\n"
                 f"Up to {int(config['runs'])} random-start trials per point | {report_status}", y=0.98, fontsize=12)
    fig.text(0.07, 0.89, "\n".join(scan_report_condition_lines(metadata)), va="top", fontsize=8, linespacing=1.3)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.12,
                           ticks=[0, 1], label="Convergence verdict")
    colorbar.ax.set_yticklabels(["Below 100%", "100%"])
    fig.text(0.07, 0.04, "Coarse samples only; symmetric convergence is failure. No fine refinement.\n"
             "Only plotted cells are measured. Skipped and untested cells are not measured failures.", fontsize=9)
    fig.subplots_adjust(bottom=0.14, top=0.65)
    with PdfPages(report_path) as pdf:
        pdf.savefig(fig)
        # Show the three coordinate planes through zero without inventing values
        # between tested samples or filling unexplored space as a failure.
        slices, axes = plt.subplots(1, 3, figsize=(11.7, 4.8))
        for axis, fixed, horizontal, vertical, title in (
            (axes[0], "da_deg", "dx_cm", "dy_cm", "delta a = 0 deg"),
            (axes[1], "dy_cm", "dx_cm", "da_deg", "delta y = 0 cm"),
            (axes[2], "dx_cm", "dy_cm", "da_deg", "delta x = 0 cm"),
        ):
            group = [point for point in complete if float(point[fixed]) == 0.0]
            for phase, status, marker, label, size in styles:
                selected = [point for point in group if point["status"] == status and point.get("phase", "coarse") == phase]
                axis.scatter([float(p[horizontal]) for p in selected], [float(p[vertical]) for p in selected],
                             c=[int(p["status"] == "pass") for p in selected], marker=marker,
                             cmap=cmap, norm=norm, s=30)
            labels = {"dx_cm": "FCS delta x (cm)", "dy_cm": "FCS delta y (cm)", "da_deg": "FCS delta a (deg)"}
            axis.set(title=title, xlabel=labels[horizontal], ylabel=labels[vertical])
            axis.grid(alpha=0.2)
        slices.suptitle("Coarse central slices: circles = 100%; crosses = below 100%; blank = untested")
        slices.subplots_adjust(left=0.08, right=0.88, bottom=0.18, top=0.83, wspace=0.5)
        colorbar = slices.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes),
                                   ticks=[0, 1], label="Convergence verdict")
        colorbar.ax.set_yticklabels(["Below 100%", "100%"])
        pdf.savefig(slices)
        plt.close(slices)
        details = [
            "Robustness test settings", "", f"Run: {metadata['settings_id']}",
            f"Algorithm: {metadata['algorithm_name']} | Status: {report_status}",
            f"Complete points: {len(complete)} | 100%: {passed} | Below 100%: {len(complete) - passed}",
            f"Incomplete points: {incomplete} | Skipped cells: {skipped} (excluded from graph and pass/fail counts)",
            "100% means all sampled trials passed; it is not a guarantee for every possible start.",
            "Both starts: field plus 80 cm on each side. Later motion is unrestricted by this start domain.",
            "Each offset samples uniformly from its admissible actual-start region; that region varies by offset.",
            "Search: " + metadata["search"],
            "A passing cell requires all N trials. New searches stop failed cells at their first failed trial; no failure rate is estimated.",
            "No refinement; failures between sampled offsets can be missed. Historical fine samples are excluded.",
            "Coarse spacing: " + " / ".join(f"{config[key] * config.get('coarse_multiplier', 1):g}" for key in
                                            ("step_x_cm", "step_y_cm", "step_angle_deg")) + " (cm / cm / deg).",
            "Search ends at failures or configured range limits; limits are not measured failure boundaries.",
            "Convergence: " + json.dumps(metadata["convergence"]),
            "Sweep: " + json.dumps(config), "", "Simulation settings:",
        ]
        details.extend(f"{key}: {value}" for key, value in settings.items())
        lines = [line for detail in details for line in (textwrap.wrap(detail, 112) or [""])]
        for start in range(0, len(lines), 44):
            page = plt.figure(figsize=(11.7, 8.3))
            page.text(0.06, 0.95, "\n".join(lines[start:start + 44]), va="top", fontsize=9,
                      family="monospace", linespacing=1.35)
            pdf.savefig(page)
            plt.close(page)
    plt.close(fig)
    angle_view_path = export_robustness_angle_view(metadata_path)
    if show_plot:
        import webbrowser
        webbrowser.open(Path(angle_view_path).resolve().as_uri())
    elif open_pdf:
        os.startfile(str(report_path))
    return str(report_path)


# =========================
# CONVERGENCE TESTS
# Standalone block on top of the core experiment runner.
# Uses the core `run_experiment(...)` in background mode without UI updates.
# =========================
def robustness_sweep_defaults():
    return {
        "runs": CONVERGE_TEST_RUNS, "step_x_cm": 50.0, "step_y_cm": 50.0,
        "step_angle_deg": 15.0, "max_x_cm": 1000.0,
        "max_y_cm": 1000.0, "max_angle_deg": 180.0,
        "seed": 1,
    }


def robustness_axes(config):
    """Signed FCS offset lattice; +/-180 degrees are the same condition."""
    if int(config["runs"]) != config["runs"] or config["runs"] < 1:
        raise ValueError("Trials per point must be a positive integer")
    if int(config["seed"]) != config["seed"]:
        raise ValueError("Seed must be an integer")
    axes = []
    for step_key, max_key, limit in (
        ("step_x_cm", "max_x_cm", FIELD_LENGTH_CM + 160.0),
        ("step_y_cm", "max_y_cm", FIELD_WIDTH_CM + 160.0),
        ("step_angle_deg", "max_angle_deg", 180.0),
    ):
        step, maximum = config[step_key], config[max_key]
        if not math.isfinite(step) or step <= 0 or not math.isfinite(maximum) or not 0 <= maximum <= limit:
            raise ValueError(f"{step_key}: positive step; {max_key}: between 0 and {limit:g}")
        count = int(math.floor(maximum / step + 1e-10))
        axis = [round(i * step, 10) for i in range(-count, count + 1)]
        # Include the configured start-domain limits even when the coarse step
        # does not divide them exactly (for example, +/-1960 cm with 50 cm).
        if maximum > axis[-1]:
            axis = [-maximum] + axis + [maximum]
        if max_key == "max_angle_deg" and axis[-1] == 180.0:
            axis.pop()
        axes.append(axis)
    return axes


def robustness_spiral_indices(axes):
    """Visit every bounded XY grid cell once, in widening square spiral rings."""
    center_x, center_y = (axis.index(0.0) for axis in axes[:2])
    yield center_x, center_y
    rings = max(center_x, center_y, len(axes[0]) - center_x - 1, len(axes[1]) - center_y - 1)
    for radius in range(1, rings + 1):
        # Walk right edge upward, top edge leftward, left edge downward,
        # and bottom edge rightward. Corners belong to exactly one edge.
        edges = (
            ((radius, y) for y in range(-radius + 1, radius + 1)),
            ((x, radius) for x in range(radius - 1, -radius - 1, -1)),
            ((-radius, y) for y in range(radius - 1, -radius - 1, -1)),
            ((x, -radius) for x in range(-radius + 1, radius + 1)),
        )
        for edge in edges:
            for x, y in edge:
                index = (center_x + x, center_y + y)
                if 0 <= index[0] < len(axes[0]) and 0 <= index[1] < len(axes[1]):
                    yield index


def run_robustness_sweep(config, trial_runner, record_trial, record_point, progress_callback=None):
    """Spiral outward per heading, skipping cells behind failed/skipped inner cells.

    Map positions are sampled N times per offset, never gridded. Reuse random
    quantiles across offsets, mapped into each offset's admissible start region.
    trial_runner owns simulated execution; callbacks stream forensic records.
    """
    axes = robustness_axes(config)
    points = []
    angles_completed = 0
    stopped = False
    half_x, half_y = FIELD_LENGTH_CM / 2 + 80.0, FIELD_WIDTH_CM / 2 + 80.0

    def evaluate(offset):
        nonlocal stopped
        phase = "coarse"
        dx, dy, da = offset
        point_id = len(points) + 1
        point = {
            "point_id": point_id, "phase": phase, "dx_cm": dx, "dy_cm": dy, "da_deg": da,
            "completed": 0, "correct": 0, "symmetric": 0, "timeout": 0,
            "blocked_by_dx_cm": None, "blocked_by_dy_cm": None,
        }
        x_low, x_high = max(-half_x, -half_x - dx), min(half_x, half_x - dx)
        y_low, y_high = max(-half_y, -half_y - dy), min(half_y, half_y - dy)
        for trial in range(int(config["runs"])):
            # The same numbered trial has the same quantiles and sensor seed
            # at every offset. Different offsets have different admissible areas.
            rng = random.Random(int(config["seed"]) + trial)
            actual_start = {
                "x_cm": rng.uniform(x_low, x_high), "y_cm": rng.uniform(y_low, y_high),
                "theta_rad": rng.uniform(-math.pi, math.pi),
            }
            trial_seed = rng.getrandbits(64)

            def progress(step, maximum):
                if progress_callback is not None:
                    return progress_callback(point_id, (dx, dy, da), trial + 1,
                                             int(config["runs"]), step, maximum, phase)

            if progress(0, INFER_STEPS_MAX) is False:
                stopped = True
                break
            result = trial_runner(actual_start, (dx, dy, da), trial_seed, progress)
            record_trial(dict(result, point_id=point_id, phase=phase, trial_id=trial + 1,
                              seed=trial_seed, dx_cm=dx, dy_cm=dy, da_deg=da))
            if result["outcome"] == "cancelled":
                stopped = True
                break
            point["completed"] += 1
            point[result["outcome"]] += 1
            if result["outcome"] != "correct":
                break
        point["status"] = (
            "fail" if point["symmetric"] or point["timeout"] else
            "incomplete" if point["completed"] < config["runs"] else
            "pass" if point["correct"] == config["runs"] else "fail"
        )
        # Failed cells have a binary verdict, not an estimated success rate.
        point["convergence_pct"] = 100.0 if point["status"] == "pass" else None
        points.append(point)
        record_point(point)
        return point

    for heading in sorted(axes[2], key=lambda value: (abs(value), value)):
        center = tuple(axis.index(0.0) for axis in axes[:2])
        statuses = {}
        for index in robustness_spiral_indices(axes):
            offset = (axes[0][index[0]], axes[1][index[1]], heading)
            # This diagonal/axial neighbor is one step nearer zero on each
            # nonzero coordinate and always belongs to the previous ring.
            inner = tuple(i - ((i > zero) - (i < zero)) for i, zero in zip(index, center))
            if index != center and statuses[inner] != "pass":
                point = {
                    "point_id": len(points) + 1, "phase": "coarse", "dx_cm": offset[0],
                    "dy_cm": offset[1], "da_deg": heading, "completed": 0, "correct": 0,
                    "symmetric": 0, "timeout": 0, "convergence_pct": None, "status": "skipped",
                    "blocked_by_dx_cm": axes[0][inner[0]], "blocked_by_dy_cm": axes[1][inner[1]],
                }
                points.append(point)
                record_point(point)
                # Keep cancellation responsive even when a large region is
                # skipped without running any simulation steps.
                if progress_callback is not None and point["point_id"] % 100 == 0:
                    if progress_callback(point["point_id"], offset, 0, int(config["runs"]), 0,
                                         INFER_STEPS_MAX, "coarse") is False:
                        stopped = True
            else:
                point = evaluate(offset)
            statuses[index] = point["status"]
            if stopped:
                break
        if stopped:
            break
        angles_completed += 1

    return {"points": points, "stopped": stopped,
            "tested_points": sum(p["status"] in {"pass", "fail"} for p in points),
            "skipped_points": sum(p["status"] == "skipped" for p in points),
            "angles_completed": angles_completed, "angles_total": len(axes[2]),
            "grid_shape": [len(axis) for axis in axes]}


def random_pose_near_field(rng=None):
    rng = random if rng is None else rng
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    margin = 80
    return {
        "x": rng.uniform(-halfL - margin, halfL + margin),
        "y": rng.uniform(-halfW - margin, halfW + margin),
        "theta": rng.uniform(-math.pi, math.pi),
    }


def random_pose_inside_field(rng=None):
    rng = random if rng is None else rng
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    return {
        "x": rng.uniform(-halfL, halfL),
        "y": rng.uniform(-halfW, halfW),
        "theta": rng.uniform(-math.pi, math.pi),
    }


def random_pose_outside_corner(rng=None):
    rng = random if rng is None else rng
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    corner_x = halfL + rng.uniform(40.0, min(OUTSIDE_MARGIN_CM, 140.0))
    corner_y = halfW + rng.uniform(40.0, min(OUTSIDE_MARGIN_CM, 140.0))
    if rng.random() < 0.5:
        corner_x = -corner_x
    if rng.random() < 0.5:
        corner_y = -corner_y
    return {
        "x": corner_x,
        "y": corner_y,
        "theta": rng.uniform(-math.pi, math.pi),
    }


def random_nearby_belief_pose(actual_pose, rng=None, near_start_limits=None):
    """Sample default or per-run near offsets in FCS, retaining the field clamp."""
    rng = random if rng is None else rng
    limits = NEAR_START_LIMITS if near_start_limits is None else near_start_limits
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2

    dx = rng.uniform(-limits["x_cm"], limits["x_cm"])
    dy = rng.uniform(-limits["y_cm"], limits["y_cm"])
    x = min(halfL, max(-halfL, actual_pose["x"] + dx))
    y = min(halfW, max(-halfW, actual_pose["y"] + dy))
    dtheta = math.radians(rng.uniform(-limits["angle_deg"], limits["angle_deg"]))
    theta = wrap_rad(actual_pose["theta"] + dtheta)
    return {"x": x, "y": y, "theta": theta}


def is_pose_outside_field(pose):
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    return (
        pose["x_cm"] < -halfL or pose["x_cm"] > halfL or
        pose["y_cm"] < -halfW or pose["y_cm"] > halfW
    )


def make_converge_start_pair(start_mode, rng=None, near_start_limits=None):
    rng = random if rng is None else rng
    if start_mode == "nearby":
        actual_start = random_pose_inside_field(rng)
        belief_start = random_nearby_belief_pose(actual_start, rng=rng, near_start_limits=near_start_limits)
    elif start_mode == "outside_corner":
        actual_start = random_pose_outside_corner(rng)
        belief_start = random_pose_near_field(rng)
    else:
        actual_start = random_pose_near_field(rng)
        belief_start = random_pose_near_field(rng)
    return actual_start, belief_start


PROFILE_REGRESSION_START_POSES = [
    {
        "actual": {"x": -135.812355, "y": -347.835710, "theta": 1.440911784813},
        "belief": {"x": 496.567953, "y": -233.992380, "theta": -1.938229181303},
    },
    {
        "actual": {"x": 384.110408, "y": 408.886058, "theta": 2.366121544958},
        "belief": {"x": -719.104256, "y": 220.178917, "theta": 2.787711593577},
    },
    {
        "actual": {"x": 801.532237, "y": 503.794575, "theta": -3.141592653590},
        "belief": {"x": 150.329891, "y": 143.890399, "theta": 2.528742525561},
    },
    {
        "actual": {"x": -376.610866, "y": -382.187690, "theta": -0.380684346676},
        "belief": {"x": -162.582586, "y": -420.018037, "theta": 1.664462940932},
    },
    {
        "actual": {"x": -245.933746, "y": -125.556840, "theta": -3.007640155320},
        "belief": {"x": 665.033648, "y": -673.435203, "theta": -2.093122661586},
    },
    {
        "actual": {"x": -629.348241, "y": -61.266388, "theta": 2.351910637349},
        "belief": {"x": -178.467288, "y": 224.270071, "theta": -0.387238767547},
    },
    {
        "actual": {"x": -802.787507, "y": 606.992828, "theta": -2.687807048071},
        "belief": {"x": -43.184008, "y": 524.039961, "theta": 0.475278094656},
    },
    {
        "actual": {"x": 631.213895, "y": -323.646210, "theta": -2.412666358514},
        "belief": {"x": 443.166058, "y": -178.589838, "theta": -0.129384289984},
    },
    {
        "actual": {"x": 966.017433, "y": 585.005832, "theta": -3.141592653590},
        "belief": {"x": -542.929694, "y": -465.330646, "theta": -1.476152543834},
    },
    {
        "actual": {"x": -769.622420, "y": -333.731311, "theta": -1.291543646476},
        "belief": {"x": -393.720462, "y": 494.920507, "theta": -0.511553210148},
    },
    {
        "actual": {"x": 928.153914, "y": 99.073408, "theta": -2.623439725931},
        "belief": {"x": -333.165063, "y": 329.232051, "theta": -2.251058986267},
    },
    {
        "actual": {"x": -290.470886, "y": -544.240080, "theta": 3.062292540628},
        "belief": {"x": 568.971704, "y": -464.560987, "theta": -1.807565407242},
    },
    {
        "actual": {"x": -164.925610, "y": 216.882815, "theta": 0.546720336276},
        "belief": {"x": 7.194361, "y": 192.132270, "theta": 0.181670403508},
    },
    {
        "actual": {"x": -843.347543, "y": 45.065006, "theta": -2.589889360725},
        "belief": {"x": 399.352795, "y": 192.980473, "theta": 1.490337325966},
    },
    {
        "actual": {"x": -161.204921, "y": 193.087303, "theta": -1.364629409584},
        "belief": {"x": -486.976421, "y": 339.418297, "theta": 1.532553882192},
    },
    {
        "actual": {"x": -18.345339, "y": -269.087211, "theta": 0.320841552954},
        "belief": {"x": 735.215947, "y": -596.506119, "theta": 2.701259566390},
    },
    {
        "actual": {"x": 469.376136, "y": -516.676012, "theta": -2.513917888117},
        "belief": {"x": 50.952876, "y": -118.478500, "theta": -0.227363657817},
    },
    {
        "actual": {"x": 255.553613, "y": -11.038938, "theta": -0.816573943363},
        "belief": {"x": 546.897126, "y": 505.169856, "theta": 2.298417270457},
    },
    {
        "actual": {"x": -154.842992, "y": -178.750229, "theta": -1.163505474420},
        "belief": {"x": 481.277627, "y": 514.224418, "theta": -0.898359460889},
    },
    {
        "actual": {"x": -667.636751, "y": 109.844115, "theta": -0.694038265109},
        "belief": {"x": -841.687187, "y": 235.228532, "theta": 0.277616837446},
    },
]


def masterdata_setting_profile_by_name(profile_name, precision_distance_modifier_enabled=None):
    if precision_distance_modifier_enabled is None:
        precision_distance_modifier_enabled = PRECISION_DISTANCE_MODIFIER_ENABLED
    for profile in MASTERDATA_SETTING_PROFILES:
        if (
            profile["name"] == profile_name
            and profile["precision_distance_modifier_enabled"] == bool(precision_distance_modifier_enabled)
        ):
            return profile
    valid_names = ", ".join(MASTERDATA_FREQUENCY_NAMES)
    raise ValueError(
        f"Unknown masterdata profile combination '{profile_name}', "
        f"distance modifier={bool(precision_distance_modifier_enabled)}. Use one of: {valid_names}"
    )


def make_robots_from_start_pair(actual_start, belief_start):
    return {
        "actual": make_robot_gc_state(actual_start["x"], actual_start["y"], actual_start["theta"]),
        "belief": make_robot_gc_state(belief_start["x"], belief_start["y"], belief_start["theta"]),
    }


def belief_pose_at_time(history, profile_dt_s, time_s):
    if not history:
        raise ValueError("history must contain at least one record")
    step_idx = max(0, min(len(history) - 1, int(round(time_s / profile_dt_s)) - 1))
    return history[step_idx]["belief_pose"]


def pose_delta(reference_pose, candidate_pose):
    return {
        "dx_cm": candidate_pose["x_cm"] - reference_pose["x_cm"],
        "dy_cm": candidate_pose["y_cm"] - reference_pose["y_cm"],
        "dtheta_rad": wrap_rad(candidate_pose["theta_rad"] - reference_pose["theta_rad"]),
    }


def pose_delta_within_tolerance(delta, tol_x_cm, tol_y_cm, tol_angle_rad):
    return (
        abs(delta["dx_cm"]) <= tol_x_cm and
        abs(delta["dy_cm"]) <= tol_y_cm and
        abs(delta["dtheta_rad"]) <= tol_angle_rad
    )


# Optional comparison tooling:
# Keeps the old 200Hz-vs-selected-profile regression report available for manual use,
# but it is no longer exposed as a primary UI action.
def run_masterdata_profile_regression_test(
    reference_profile_name="200Hz",
    candidate_profile_name="40Hz",
    duration_s=2.0,
    start_cases=PROFILE_REGRESSION_START_POSES,
    algorithm_name=FUSED_MOTION_LANDMARK_ALGORITHM_NAME,
    sample_count=10,
    tol_x_cm=3.0,
    tol_y_cm=3.0,
    tol_angle_rad=math.radians(3.0),
    include_histories=False,
):
    global NOISE_ENABLED, ODOMETRY_BIAS_ENABLED, MOTION_SLIP_ENABLED
    global LANDMARK_DROPOUT_ENABLED, LANDMARK_OBSTRUCTION_ENABLED, VIEWING_ANGLE_ENABLED

    reference_profile = masterdata_setting_profile_by_name(reference_profile_name)
    candidate_profile = masterdata_setting_profile_by_name(candidate_profile_name)
    saved_state = {
        "dt": DT,
        "pi_cam": PI_CAM,
        "pi_odo": PI_ODO,
        "pi_imu": PI_IMU,
        "lr_xy": LR_XY,
        "lr_angle": LR_ANGLE,
        "motion_lr_vel_xy": MOTION_LR_VEL_XY,
        "motion_lr_vel_theta": MOTION_LR_VEL_THETA,
        "motion_lr_acc_xy": MOTION_LR_ACC_XY,
        "motion_lr_acc_theta": MOTION_LR_ACC_THETA,
        "precision_distance_modifier_enabled": PRECISION_DISTANCE_MODIFIER_ENABLED,
        "noise_enabled": NOISE_ENABLED,
        "odometry_bias_enabled": ODOMETRY_BIAS_ENABLED,
        "motion_slip_enabled": MOTION_SLIP_ENABLED,
        "landmark_dropout_enabled": LANDMARK_DROPOUT_ENABLED,
        "landmark_obstruction_enabled": LANDMARK_OBSTRUCTION_ENABLED,
        "viewing_angle_enabled": VIEWING_ANGLE_ENABLED,
    }

    def run_profile(profile, start_case):
        global NOISE_ENABLED, ODOMETRY_BIAS_ENABLED, MOTION_SLIP_ENABLED
        global LANDMARK_DROPOUT_ENABLED, LANDMARK_OBSTRUCTION_ENABLED, VIEWING_ANGLE_ENABLED

        apply_masterdata_setting_profile(profile)
        NOISE_ENABLED = False
        ODOMETRY_BIAS_ENABLED = False
        MOTION_SLIP_ENABLED = False
        LANDMARK_DROPOUT_ENABLED = False
        LANDMARK_OBSTRUCTION_ENABLED = False
        VIEWING_ANGLE_ENABLED = False
        reset_motion_slip_state()
        reset_landmark_obstruction_state()
        steps = max(1, int(round(duration_s / profile["dt"])))
        robots = make_robots_from_start_pair(start_case["actual"], start_case["belief"])
        return run_experiment(robots, algorithm_name, max_steps=steps, until_convergence=False)

    results = []
    try:
        for idx, start_case in enumerate(start_cases, start=1):
            reference_history = run_profile(reference_profile, start_case)
            candidate_history = run_profile(candidate_profile, start_case)
            endpoint_delta = pose_delta(reference_history[-1]["belief_pose"], candidate_history[-1]["belief_pose"])

            max_sample_delta = {"dx_cm": 0.0, "dy_cm": 0.0, "dtheta_rad": 0.0}
            for sample_idx in range(1, sample_count + 1):
                sample_time_s = duration_s * sample_idx / sample_count
                reference_pose = belief_pose_at_time(reference_history, reference_profile["dt"], sample_time_s)
                candidate_pose = belief_pose_at_time(candidate_history, candidate_profile["dt"], sample_time_s)
                sample_delta = pose_delta(reference_pose, candidate_pose)
                max_sample_delta["dx_cm"] = max(max_sample_delta["dx_cm"], abs(sample_delta["dx_cm"]))
                max_sample_delta["dy_cm"] = max(max_sample_delta["dy_cm"], abs(sample_delta["dy_cm"]))
                max_sample_delta["dtheta_rad"] = max(max_sample_delta["dtheta_rad"], abs(sample_delta["dtheta_rad"]))

            endpoint_pass = pose_delta_within_tolerance(endpoint_delta, tol_x_cm, tol_y_cm, tol_angle_rad)
            result = {
                "case": idx,
                "actual_start": start_case["actual"],
                "belief_start": start_case["belief"],
                "reference_steps": len(reference_history),
                "candidate_steps": len(candidate_history),
                "endpoint_delta": endpoint_delta,
                "max_sample_delta": max_sample_delta,
                "endpoint_pass": endpoint_pass,
            }
            if include_histories:
                result["reference_history"] = reference_history
                result["candidate_history"] = candidate_history
            results.append(result)
    finally:
        apply_masterdata_setting_profile(saved_state)
        NOISE_ENABLED = saved_state["noise_enabled"]
        ODOMETRY_BIAS_ENABLED = saved_state["odometry_bias_enabled"]
        MOTION_SLIP_ENABLED = saved_state["motion_slip_enabled"]
        LANDMARK_DROPOUT_ENABLED = saved_state["landmark_dropout_enabled"]
        LANDMARK_OBSTRUCTION_ENABLED = saved_state["landmark_obstruction_enabled"]
        VIEWING_ANGLE_ENABLED = saved_state["viewing_angle_enabled"]

    return {
        "reference_profile": reference_profile_name,
        "candidate_profile": candidate_profile_name,
        "reference_profile_settings": dict(reference_profile),
        "candidate_profile_settings": dict(candidate_profile),
        "duration_s": duration_s,
        "algorithm_name": algorithm_name,
        "precision_distance_modifier_enabled": PRECISION_DISTANCE_MODIFIER_ENABLED,
        "tolerance": {
            "x_cm": tol_x_cm,
            "y_cm": tol_y_cm,
            "theta_rad": tol_angle_rad,
        },
        "results": results,
        "endpoint_pass_count": sum(1 for result in results if result["endpoint_pass"]),
    }


def plot_masterdata_profile_regression_report(report):
    import datetime
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(
        reports_dir,
        f"profile_regression_{report['reference_profile']}_vs_{report['candidate_profile']}_{timestamp_file}.pdf",
    )

    halfW = FIELD_WIDTH_CM / 2
    halfL = FIELD_LENGTH_CM / 2
    cases_per_page = 2
    results = report["results"]
    total_pages = max(1, math.ceil(len(results) / cases_per_page))
    tol = report["tolerance"]
    reference_profile_settings = report.get("reference_profile_settings", {})
    candidate_profile_settings = report.get("candidate_profile_settings", {})

    def path_xy(history):
        return (
            [record["belief_pose"]["y_cm"] for record in history],
            [record["belief_pose"]["x_cm"] for record in history],
        )

    def draw_pose_heading(ax, pose, color, scale_cm=70.0):
        dx = scale_cm * math.sin(pose["theta_rad"])
        dy = scale_cm * math.cos(pose["theta_rad"])
        ax.arrow(
            pose["y_cm"],
            pose["x_cm"],
            dx,
            dy,
            head_width=28,
            head_length=28,
            length_includes_head=True,
            color=color,
            linewidth=1.4,
        )

    def start_pose_view(start_pose):
        return {
            "x_cm": start_pose["x"],
            "y_cm": start_pose["y"],
            "theta_rad": start_pose["theta"],
        }

    def draw_case(ax, result):
        ax.set_xlim(halfW + 80, -halfW - 80)
        ax.set_ylim(-halfL - 80, halfL + 80)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

        rect = Rectangle((halfW, -halfL), -FIELD_WIDTH_CM, FIELD_LENGTH_CM, fill=False, linewidth=1.4)
        ax.add_patch(rect)
        ax.plot([halfW, -halfW], [0, 0], linewidth=1.0, color="0.35")

        ref_x, ref_y = path_xy(result["reference_history"])
        cand_x, cand_y = path_xy(result["candidate_history"])
        ax.plot(ref_x, ref_y, color="red", linewidth=1.8, label=f"{report['reference_profile']} belief path")
        ax.plot(cand_x, cand_y, color="tab:blue", linewidth=1.8, linestyle="--", label=f"{report['candidate_profile']} belief path")

        actual_start = result["actual_start"]
        belief_start = result["belief_start"]
        reference_end = result["reference_history"][-1]["belief_pose"]
        candidate_end = result["candidate_history"][-1]["belief_pose"]
        ax.scatter([actual_start["y"]], [actual_start["x"]], marker="x", s=110, color="green", linewidths=2.2, label="actual")
        ax.scatter([belief_start["y"]], [belief_start["x"]], marker="o", s=95, facecolors="none", edgecolors="red", linewidths=2.0, label="belief start")
        ax.scatter([belief_start["y"]], [belief_start["x"]], marker="s", s=70, facecolors="none", edgecolors="tab:blue", linewidths=1.8, label="candidate start")
        ax.scatter([reference_end["y_cm"]], [reference_end["x_cm"]], marker="o", s=85, facecolors="none", edgecolors="red", linewidths=2.0, label=f"{report['reference_profile']} end")
        ax.scatter([candidate_end["y_cm"]], [candidate_end["x_cm"]], marker="s", s=75, facecolors="none", edgecolors="tab:blue", linewidths=2.0, label=f"{report['candidate_profile']} end")
        draw_pose_heading(ax, start_pose_view(actual_start), "green")
        draw_pose_heading(ax, start_pose_view(belief_start), "red")
        draw_pose_heading(ax, start_pose_view(belief_start), "tab:blue", scale_cm=55.0)
        draw_pose_heading(ax, reference_end, "red")
        draw_pose_heading(ax, candidate_end, "tab:blue", scale_cm=55.0)

        endpoint_delta = result["endpoint_delta"]
        max_sample_delta = result["max_sample_delta"]
        title = (
            f"Case {result['case']}  endpoint={'PASS' if result['endpoint_pass'] else 'FAIL'}"
        )
        detail = (
            f"end d=({endpoint_delta['dx_cm']:.1f}, {endpoint_delta['dy_cm']:.1f}) cm, "
            f"{math.degrees(endpoint_delta['dtheta_rad']):.2f} deg\n"
            f"visual path delta samples: max=({max_sample_delta['dx_cm']:.1f}, {max_sample_delta['dy_cm']:.1f}) cm, "
            f"{math.degrees(max_sample_delta['dtheta_rad']):.2f} deg"
        )
        ax.set_title(title + "\n" + detail, fontsize=10)

    with PdfPages(pdf_path) as pdf:
        for page_idx in range(total_pages):
            page_results = results[page_idx * cases_per_page:(page_idx + 1) * cases_per_page]
            fig = plt.figure(figsize=(14, 5.2 + 5.8 * len(page_results)))
            gs = GridSpec(len(page_results) + 1, 1, figure=fig, height_ratios=[1.3] + [5.2] * len(page_results))
            fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.04, hspace=0.35)

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.axis("off")
            header = (
                f"Masterdata Profile Regression   |   Time: {timestamp}   |   Page {page_idx + 1}/{total_pages}\n"
                f"{report['reference_profile']} vs {report['candidate_profile']}   |   "
                f"duration={report['duration_s']:.2f}s   algorithm={report['algorithm_name']}   "
                f"endpoint pass={report['endpoint_pass_count']}/{len(results)}   path comparison=visual only\n"
                f"{report['reference_profile']} {precision_report_row(reference_profile_settings)}\n"
                f"{report['reference_profile']} {learning_rate_report_row(reference_profile_settings)}\n"
                f"{report['candidate_profile']} {precision_report_row(candidate_profile_settings)}\n"
                f"{report['candidate_profile']} {learning_rate_report_row(candidate_profile_settings)}\n"
                f"Distance precision modifier: {'YES' if report.get('precision_distance_modifier_enabled', True) else 'NO'}\n"
                f"Tolerance: |dx|<={tol['x_cm']:.1f} cm   |dy|<={tol['y_cm']:.1f} cm   "
                f"|dtheta|<={math.degrees(tol['theta_rad']):.1f} deg"
            )
            ax0.text(0.01, 0.98, header, fontsize=11, family="monospace", va="top")

            for row_idx, result in enumerate(page_results, start=1):
                ax = fig.add_subplot(gs[row_idx, 0])
                draw_case(ax, result)
                if page_idx == 0 and row_idx == 1:
                    ax.legend(loc="upper right", fontsize=9)

            pdf.savefig(fig)
            plt.close(fig)

    os.startfile(pdf_path)
    return pdf_path


def pose_symmetrically_converged(actual_pose, belief_pose):
    symmetrical_pose = {
        "x_cm": -actual_pose["x_cm"],
        "y_cm": -actual_pose["y_cm"],
        "theta_rad": wrap_rad(actual_pose["theta_rad"] + math.pi),
    }
    return pose_converged_ros(
        symmetrical_pose, belief_pose, CONV_TOL_X_CM, CONV_TOL_Y_CM, CONV_TOL_ANGLE_RAD,
    )


def history_has_sequential_convergence(history, symmetrical=False):
    convergence_hits = 0
    for record in history:
        if symmetrical:
            matches = pose_symmetrically_converged(record["actual_pose"], record["belief_pose"])
        else:
            matches = pose_converged_ros(
                record["actual_pose"],
                record["belief_pose"],
                CONV_TOL_X_CM,
                CONV_TOL_Y_CM,
                CONV_TOL_ANGLE_RAD,
            )
        if matches:
            convergence_hits += 1
            if convergence_hits >= CONV_REQUIRED_SEQUENTIAL_HITS:
                return True
        else:
            convergence_hits = 0
    return False


def step_stats(step_counts):
    if not step_counts:
        return None, None

    mean_steps = sum(step_counts) / len(step_counts)
    variance = sum((steps - mean_steps) ** 2 for steps in step_counts) / len(step_counts)
    return mean_steps, math.sqrt(variance)


def percentile_step(step_counts, percentile):
    if not step_counts:
        return None

    sorted_steps = sorted(step_counts)
    rank = math.ceil((percentile / 100.0) * len(sorted_steps))
    index = min(len(sorted_steps) - 1, max(0, rank - 1))
    return sorted_steps[index]


def run_converge_test(
    robots,
    algorithm_name,
    runs=CONVERGE_TEST_RUNS,
    progress_callback=None,
    start_mode="random",
    start_cases=None,
    trial_specs=None,
    collect_failures=True,
    stop_check=None,
    near_start_limits=None,
):
    converged = 0
    non_converged_symmetrical = 0
    non_converged = 0
    failures = []
    last_history = []
    trial_steps = []
    converged_steps = []
    near_start_limits = dict(NEAR_START_LIMITS if near_start_limits is None else near_start_limits)
    stopped = False

    if start_cases is not None:
        trial_specs = [
            {"actual_start": case["actual"], "belief_start": case["belief"]}
            for case in start_cases
        ]
        start_mode = "regression_set"

    if trial_specs is None:
        trial_specs = [None] * runs
    else:
        runs = len(trial_specs)

    for i in range(runs):
        if stop_check is not None and stop_check():
            stopped = True
            break

        if trial_specs[i] is None:
            actual_start, belief_start = make_converge_start_pair(start_mode, near_start_limits=near_start_limits)
        else:
            actual_start = dict(trial_specs[i]["actual_start"])
            belief_start = dict(trial_specs[i]["belief_start"])
        scalar_2_gc(
            robots["actual"],
            x_cm=actual_start["x"],
            y_cm=actual_start["y"],
            theta_rad=actual_start["theta"],
        )
        scalar_2_gc(
            robots["belief"],
            x_cm=belief_start["x"],
            y_cm=belief_start["y"],
            theta_rad=belief_start["theta"],
        )
        zero_robot_generalized_motion(robots["actual"])
        zero_robot_generalized_motion(robots["belief"])

        start_actual = robot_pose_view(robots["actual"])
        start_belief = robot_pose_view(robots["belief"])

        history = run_experiment(
            robots,
            algorithm_name,
            max_steps=INFER_STEPS_MAX,
            until_convergence=True,
            stop_on_symmetrical_convergence=True,
            step_callback=(lambda: False if stop_check is not None and stop_check() else None) if stop_check is not None else None,
            callback_interval=25,
        )
        if history:
            # Per-step snapshots share this settings dictionary.
            history[0]["settings"]["near_start_limits"] = dict(near_start_limits)
        last_history = history
        steps_taken = len(history)
        trial_steps.append(steps_taken)

        if stop_check is not None and stop_check():
            stopped = True
            break

        if history and history_has_sequential_convergence(history):
            converged += 1
            converged_steps.append(steps_taken)
        else:
            symmetrical = history_has_sequential_convergence(history, symmetrical=True)
            if symmetrical:
                non_converged_symmetrical += 1
            else:
                non_converged += 1
            if collect_failures:
                actual_end = robot_pose_view(robots["actual"])
                belief_end = robot_pose_view(robots["belief"])
                visible_landmark_ids = sorted(obs["obs_id"] for obs in simulation_sense_landmarks(robots["actual"], active_simulation_landmarks()))
                failures.append({
                    "symmetrical": symmetrical,
                    "actual_start": start_actual,
                    "belief_start": start_belief,
                    "actual_end": actual_end,
                    "belief_end": belief_end,
                    "actual_start_outside": is_pose_outside_field(start_actual),
                    "belief_start_outside": is_pose_outside_field(start_belief),
                    "visible_landmark_count": len(visible_landmark_ids),
                    "visible_landmark_ids": visible_landmark_ids,
                })

        if progress_callback is not None:
            keep_running = progress_callback(i + 1, runs, converged, non_converged + non_converged_symmetrical)
            if keep_running is False:
                stopped = True
                break

    mean_steps, sd_steps = step_stats(converged_steps)
    max_steps = max(converged_steps) if converged_steps else None
    p95_steps = percentile_step(converged_steps, 95)
    p99_steps = percentile_step(converged_steps, 99)

    return {
        "runs": runs,
        "converged": converged,
        "non_converged_symmetrical": non_converged_symmetrical,
        "non_converged": non_converged,
        "trial_steps": trial_steps,
        "converged_steps": converged_steps,
        "mean_steps_to_converge": mean_steps,
        "sd_steps_to_converge": sd_steps,
        "max_steps_to_converge": max_steps,
        "p95_steps_to_converge": p95_steps,
        "p99_steps_to_converge": p99_steps,
        "failures": failures,
        "last_history": last_history,
        "start_mode": start_mode,
        "near_start_limits": near_start_limits,
        "stopped": stopped,
    }


def plot_converge_test_report(report, algorithm_name="Unknown"):
    if report is None:
        return

    import datetime
    import os
    from matplotlib.gridspec import GridSpec
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    converged = report["converged"]
    non_converged_symmetrical = report.get("non_converged_symmetrical", 0)
    non_converged = report["non_converged"]
    mean_steps = report.get("mean_steps_to_converge")
    sd_steps = report.get("sd_steps_to_converge")
    max_steps = report.get("max_steps_to_converge")
    p95_steps = report.get("p95_steps_to_converge")
    p99_steps = report.get("p99_steps_to_converge")
    failures = [fail for fail in report["failures"] if not fail.get("symmetrical", False)]
    counts_text = (
        f"Converged: {converged}    Non-converged-symmetrical: {non_converged_symmetrical}    "
        f"Non-converged: {non_converged}"
    )
    convergence_condition_text = convergence_condition_report_row(
        CONV_TOL_X_CM, CONV_TOL_Y_CM, CONV_TOL_ANGLE_RAD, CONV_REQUIRED_SEQUENTIAL_HITS)
    last_history = report.get("last_history", [])
    if last_history:
        settings = last_history[0].get("settings", experiment_settings(algorithm_name))
    else:
        settings = experiment_settings(algorithm_name)
    near_condition = near_start_condition_text(report.get("near_start_limits", settings.get("near_start_limits", NEAR_START_LIMITS)))

    failures_per_page = 3
    total_failures = len(failures)
    total_pages = max(1, math.ceil(total_failures / failures_per_page))
    timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    start_mode = report.get("start_mode", "random")
    if start_mode in {"dynamic_demo_random", "dynamic_demo_nearby"}:
        mode_label = "Dynamic Demo Converge Test"
        demo_path_name = report.get("demo_path_name", "Unknown")
        if start_mode == "dynamic_demo_nearby":
            start_condition_text = f"Start condition: moving actual robot on demo path '{demo_path_name}'; {near_condition}"
        else:
            start_condition_text = f"Start condition: moving actual robot on demo path '{demo_path_name}'; actual and belief independently random near field"
        start_condition_text += (
            f"\nBall position (FCS): x={report['look_target_x_cm']:.1f} cm   "
            f"y={report['look_target_y_cm']:.1f} cm"
        )
    elif start_mode == "nearby":
        mode_label = "Landmark Converge Nearby"
        start_condition_text = f"Start condition: landmark-only static actual pose; {near_condition}"
    elif start_mode == "outside_corner":
        mode_label = "Landmark Converge Outside Corner"
        start_condition_text = "Start condition: landmark-only static actual outside corner; belief random near field"
    elif start_mode == "regression_set":
        mode_label = "Landmark Converge Regression Set"
        start_condition_text = "Start condition: fixed regression actual/belief start pairs; every actual start sees at least 2 landmarks at default FoV"
    else:
        mode_label = "Landmark Converge Test"
        start_condition_text = "Start condition: landmark-only static actual pose; actual and belief independently random near field"
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, f"{start_mode}_converge_test_report_{timestamp_file}.pdf")

    if mean_steps is None or sd_steps is None or max_steps is None or p95_steps is None or p99_steps is None:
        step_stats_text = "Steps to converge (converged cases only): n/a"
    else:
        step_stats_text = f"Steps to converge (converged cases only): mean={mean_steps:.1f}   sd={sd_steps:.1f}   p95={p95_steps}   p99={p99_steps}   max={max_steps}   n={converged}"

    halfW = FIELD_WIDTH_CM / 2
    halfL = FIELD_LENGTH_CM / 2
    condition_text = "\n".join(convergence_report_condition_lines(settings))

    def draw_field_snapshot(ax, actual_pose, belief_pose, title, history=None):
        ax.set_title(title, fontsize=7)
        ax.set_xlim(halfW + 80, -halfW - 80)
        ax.set_ylim(-halfL - 80, halfL + 80)
        ax.set_aspect("equal", adjustable="box")

        rect = Rectangle((halfW, -halfL), -FIELD_WIDTH_CM, FIELD_LENGTH_CM, fill=False, linewidth=0.9)
        ax.add_patch(rect)
        ax.plot([halfW, -halfW], [0, 0], linewidth=0.7)

        if history:
            actual_y = [record["actual_pose"]["y_cm"] for record in history]
            actual_x = [record["actual_pose"]["x_cm"] for record in history]
            belief_y = [record["belief_pose"]["y_cm"] for record in history]
            belief_x = [record["belief_pose"]["x_cm"] for record in history]
            ax.plot(actual_y, actual_x, color="green", linewidth=0.6, alpha=0.75)
            ax.plot(belief_y, belief_x, color="red", linewidth=0.6, alpha=0.75)

        ax.scatter([actual_pose["y_cm"]], [actual_pose["x_cm"]], marker="o", s=32, facecolors="none", edgecolors="green", linewidths=1.0)
        ax.scatter([belief_pose["y_cm"]], [belief_pose["x_cm"]], marker="o", s=32, facecolors="none", edgecolors="red", linewidths=1.0)

        for pose, color in [(actual_pose, "green"), (belief_pose, "red")]:
            dx = 80 * math.sin(pose["theta_rad"])
            dy = 80 * math.cos(pose["theta_rad"])
            ax.arrow(
                pose["y_cm"],
                pose["x_cm"],
                dx,
                dy,
                head_width=36,
                head_length=36,
                length_includes_head=True,
                color=color,
                linewidth=0.8,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    report_rows = failures_per_page
    overview_header_height = 4.0
    if start_mode in {"dynamic_demo_random", "dynamic_demo_nearby"}:
        overview_header_height += 0.3

    with PdfPages(pdf_path) as pdf:
        for page_idx in range(total_pages):
            page_failures = failures[page_idx * failures_per_page:(page_idx + 1) * failures_per_page]
            n_rows = report_rows
            is_first_page = page_idx == 0
            header_height = overview_header_height if is_first_page else 0.35

            first_case_row = 2 if is_first_page else 1
            fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
            gs = GridSpec(
                n_rows + first_case_row,
                3,
                figure=fig,
                height_ratios=[header_height] + ([0.7] if is_first_page else []) + [5.4] * n_rows,
                width_ratios=[2.5, 2.5, 3.2],
            )
            fig.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.04, hspace=0.32, wspace=0.18)

            ax0 = fig.add_subplot(gs[0, :])
            ax0.axis("off")
            if is_first_page:
                header = (
                    f"{mode_label}   |   Time: {timestamp}   |   Algorithm: {algorithm_name}   |   Page {page_idx + 1}/{total_pages}\n"
                    f"{counts_text}\n"
                    f"{convergence_condition_text}\n"
                    f"{step_stats_text}\n"
                    f"{start_condition_text}\n"
                    f"{condition_text}"
                )
            else:
                header = f"{mode_label}   |   Time: {timestamp}"
            header = "\n".join(
                textwrap.fill(
                    line,
                    width=120,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                for line in header.splitlines()
            )
            ax0.text(0.01, 0.98, header, fontsize=6.5 if is_first_page else 7, family="monospace", va="top")

            if is_first_page:
                ax_section = fig.add_subplot(gs[1, :])
                ax_section.axis("off")
                ax_section.text(0.01, 0.95, "Failure plots", fontsize=10, fontweight="bold", va="top")
                ax_section.text(0.01, 0.40, "Actual = green   |   Belief = red", fontsize=7, va="top")

            if not page_failures:
                ax_no = fig.add_subplot(gs[first_case_row, :])
                ax_no.axis("off")
                ax_no.text(0.01, 0.98, "No non-converged cases to display.", fontsize=7, family="monospace", va="top")
            else:
                for row_idx, fail in enumerate(page_failures, start=1):
                    fail_number = page_idx * failures_per_page + row_idx
                    a0 = fail["actual_start"]
                    b0 = fail["belief_start"]
                    a1 = fail["actual_end"]
                    b1 = fail["belief_end"]
                    case_row = row_idx + first_case_row - 1
                    ax_start = fig.add_subplot(gs[case_row, 0])
                    draw_field_snapshot(ax_start, a0, b0, f"Fail {fail_number} START")

                    ax_end = fig.add_subplot(gs[case_row, 1])
                    path_history = fail.get("path_history")
                    end_title = f"Fail {fail_number} END"
                    if path_history:
                        end_title += f" (last {len(path_history)} steps)"
                    draw_field_snapshot(ax_end, a1, b1, end_title, history=path_history)

                    ax_txt = fig.add_subplot(gs[case_row, 2])
                    ax_txt.axis("off")
                    txt = (
                        f"Fail {fail_number}\n"
                        f"Non-converged\n"
                        f"{format_ros_pose_line('Start Actual', a0, decimals=0, angle_decimals=3)}"
                        f"   outside={fail['actual_start_outside']}\n"
                        f"{format_ros_pose_line('Start Belief', b0, decimals=0, angle_decimals=3)}"
                        f"   outside={fail['belief_start_outside']}\n"
                        f"Visible landmarks at fail end: {fail['visible_landmark_count']}\n"
                        f"{format_ros_pose_line('End   Actual', a1, decimals=0, angle_decimals=3)}\n"
                        f"{format_ros_pose_line('End   Belief', b1, decimals=0, angle_decimals=3)}"
                    )
                    txt = "\n".join(
                        textwrap.fill(line, width=50, break_long_words=False, break_on_hyphens=False)
                        for line in txt.splitlines()
                    )
                    ax_txt.text(0.01, 0.98, txt, fontsize=6, family="monospace", va="top")

            pdf.savefig(fig)
            plt.close(fig)

    os.startfile(pdf_path)
    return pdf_path


# =========================
# ONE-STEP ANALYSIS
# =========================
def wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def ros_pose_to_legacy_pose(x_cm, y_cm, theta_rad):
    return (
        -y_cm,
        x_cm,
        wrap_deg(math.degrees(theta_rad) + 90.0),
    )


def pose_converged_ros(actual_pose, belief_pose, tol_x, tol_y, tol_angle_rad):
    dx = abs(belief_pose["x_cm"] - actual_pose["x_cm"])
    dy = abs(belief_pose["y_cm"] - actual_pose["y_cm"])
    da = abs(wrap_rad(belief_pose["theta_rad"] - actual_pose["theta_rad"]))
    return dx <= tol_x and dy <= tol_y and da <= tol_angle_rad


def format_ros_pose_line(name, pose_ros, decimals=1, angle_decimals=3):
    return (
        f"{name}: x={pose_ros['x_cm']:.{decimals}f}  "
        f"y={pose_ros['y_cm']:.{decimals}f}  "
        f"theta={pose_ros['theta_rad']:.{angle_decimals}f} rad"
    )


def format_display_value(value, decimals=4):
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 10 ** (-decimals):
        return f"{value:.3e}"
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def rcs_to_fcs(robot_fcs_gc, obs):
    # Input:
    #   robot state in generalized coordinates
    #   obs in robot coordinates (RCS): x_rcs_cm, y_rcs_cm
    # Output:
    #   projected observation position in FCS as (x_cm, y_cm)
    # Function:
    #   RCS -> FCS translation
    #   inverse of the fcs_to_rcs, mainly used for visualization/debugging
    rx, ry, theta = gc_2_scalar(robot_fcs_gc)
    c = math.cos(theta)
    s = math.sin(theta)

    dx = c * obs["x_rcs_cm"] - s * obs["y_rcs_cm"]
    dy = s * obs["x_rcs_cm"] + c * obs["y_rcs_cm"]
    return rx + dx, ry + dy


def landmark_analysis_data(actual_fcs_gc, belief_fcs_gc, diagnostics=None, ground_truth=None):
    if diagnostics is None:
        observations, ground_truth = simulation_sense_landmarks(actual_fcs_gc, active_simulation_landmarks(), return_ground_truth=True)
        diagnostics = simulation_step_from_observations(
            belief_fcs_gc, observations, None, LANDMARK_ONLY_ALGORITHM_NAME, DT)["diagnostics"]
    y_obs_map, yhat_map = diagnostics["y_obs"], diagnostics["y_hat"]
    prediction_gc = diagnostics["landmark_prediction_gc"]
    observed_landmark_ids = list(y_obs_map)
    used_candidate_ids = set()
    details = []
    for association in diagnostics["landmark_terms"]:
        if not association["used_for_update"]:
            continue
        obs_id, lid = association["obs_id"], association["candidate_id"]
        used_candidate_ids.add(lid)
        actual_landmark_id = ground_truth[obs_id]
        association_mismatch = (
            association["association_name"] == DATA_ASSOCIATION_GATED_NEAREST_NAME
            and actual_landmark_id is not None
            and actual_landmark_id != lid
        )
        ex = association["eps"][0]
        ey = association["eps"][1]
        ea = association["eps"][2] / ANGLE_WEIGHT if ANGLE_WEIGHT != 0.0 else 0.0
        obs_fcs = rcs_to_fcs(prediction_gc, y_obs_map[obs_id])
        pred_fcs = rcs_to_fcs(prediction_gc, yhat_map[lid])
        contrib = association["jt_p_eps"]
        contrib_mag = math.sqrt(contrib[0] * contrib[0] + contrib[1] * contrib[1] + contrib[2] * contrib[2])
        error_mag = math.sqrt(ex * ex + ey * ey + ea * ea)
        details.append({
            "id": lid,
            "obs_id": obs_id,
            "name": y_obs_map[obs_id]["name"],
            "landmark_id": actual_landmark_id,
            "actual_landmark_id": actual_landmark_id,
            "associated_landmark_id": lid,
            "association_mismatch": association_mismatch,
            "obs_fcs": obs_fcs,
            "pred_fcs": pred_fcs,
            "ex": ex,
            "ey": ey,
            "ea": ea,
            "precision": association["precision"],
            "range_cm": math.hypot(yhat_map[lid]["x_rcs_cm"], yhat_map[lid]["y_rcs_cm"]),
            "error_mag": error_mag,
            "association_error_cm": association["position_error_cm"],
            "association_name": association["association_name"],
            "contrib": contrib,
            "contrib_mag": contrib_mag,
            "visible": True,
        })

    details.sort(key=lambda d: d["contrib_mag"], reverse=True)
    total_contrib = [0.0, 0.0, 0.0]
    for d in details:
        total_contrib[0] += d["contrib"][0]
        total_contrib[1] += d["contrib"][1]
        total_contrib[2] += d["contrib"][2]
    return {
        "y_obs": y_obs_map,
        "y_hat": yhat_map,
        "details": details,
        "total_contrib": total_contrib,
        "observed_landmark_ids": observed_landmark_ids,
        "hidden_landmark_ids": [lid for lid in active_simulation_landmarks() if lid not in used_candidate_ids],
        "association_mismatch_count": sum(1 for d in details if d["association_mismatch"]),
    }


def motion_analysis_details(belief_fcs_gc, y_motion_obs, motion_terms):
    return [
        {
            "id": "x",
            "obs_pos": None,
            "pred_pos": belief_fcs_gc["x"][0],
            "pos_error": 0.0,
            "obs_vel": y_motion_obs["x_vel_cm_s"],
            "pred_vel": belief_fcs_gc["x"][1],
            "vel_error": motion_terms["x"]["vel_error"],
            "obs_acc": y_motion_obs["x_acc_cm_s2"],
            "pred_acc": belief_fcs_gc["x"][2],
            "acc_error": motion_terms["x"]["acc_error"],
            "jerk_error": motion_terms["x"]["jerk_error"],
        },
        {
            "id": "y",
            "obs_pos": None,
            "pred_pos": belief_fcs_gc["y"][0],
            "pos_error": 0.0,
            "obs_vel": y_motion_obs["y_vel_cm_s"],
            "pred_vel": belief_fcs_gc["y"][1],
            "vel_error": motion_terms["y"]["vel_error"],
            "obs_acc": y_motion_obs["y_acc_cm_s2"],
            "pred_acc": belief_fcs_gc["y"][2],
            "acc_error": motion_terms["y"]["acc_error"],
            "jerk_error": motion_terms["y"]["jerk_error"],
        },
        {
            "id": "theta",
            "obs_pos": None,
            "pred_pos": wrap_rad(belief_fcs_gc["theta"][0]),
            "pos_error": 0.0,
            "obs_vel": y_motion_obs["theta_vel_rad_s"],
            "pred_vel": belief_fcs_gc["theta"][1],
            "vel_error": motion_terms["theta"]["vel_error"],
            "obs_acc": y_motion_obs["theta_acc_rad_s2"],
            "pred_acc": belief_fcs_gc["theta"][2],
            "acc_error": motion_terms["theta"]["acc_error"],
            "jerk_error": motion_terms["theta"]["jerk_error"],
        },
    ]


def one_step_analysis(robots, algorithm_name=ACTIVE_ALGORITHM_NAME):
    actual_pose = robot_pose_view(robots["actual"])
    belief_pose = robot_pose_view(robots["belief"])
    y_motion_obs = simulation_sense_motion(robots["actual"])

    if algorithm_uses_landmarks(algorithm_name):
        observations, ground_truth = simulation_sense_landmarks(robots["actual"], active_simulation_landmarks(), return_ground_truth=True)
    else:
        observations, ground_truth = [], {}
    diagnostics = simulation_step_from_observations(
        robots["belief"], observations, y_motion_obs, algorithm_name, DT)["diagnostics"]

    if algorithm_name == MOTION_ONLY_ALGORITHM_NAME:
        details = motion_analysis_details(robots["belief"], y_motion_obs, diagnostics["motion_terms"])
        return {
            "mode": "motion",
            "actual_pose": actual_pose,
            "belief_pose": belief_pose,
            "details": details,
        }

    landmark_data = landmark_analysis_data(robots["actual"], robots["belief"], diagnostics, ground_truth)

    if algorithm_name == FUSED_MOTION_LANDMARK_ALGORITHM_NAME:
        motion_details = motion_analysis_details(robots["belief"], y_motion_obs, diagnostics["motion_terms"])
        return {
            "mode": "fused",
            "actual_pose": actual_pose,
            "belief_pose": belief_pose,
            "y_obs": landmark_data["y_obs"],
            "y_hat": landmark_data["y_hat"],
            "details": landmark_data["details"],
            "total_contrib": landmark_data["total_contrib"],
            "observed_landmark_ids": landmark_data["observed_landmark_ids"],
            "hidden_landmark_ids": landmark_data["hidden_landmark_ids"],
            "association_mismatch_count": landmark_data["association_mismatch_count"],
            "motion_details": motion_details,
        }

    return {
        "mode": "landmark",
        "actual_pose": actual_pose,
        "belief_pose": belief_pose,
        "y_obs": landmark_data["y_obs"],
        "y_hat": landmark_data["y_hat"],
        "details": landmark_data["details"],
        "total_contrib": landmark_data["total_contrib"],
        "observed_landmark_ids": landmark_data["observed_landmark_ids"],
        "hidden_landmark_ids": landmark_data["hidden_landmark_ids"],
        "association_mismatch_count": landmark_data["association_mismatch_count"],
    }


# =========================
# UI
# =========================
# Self-contained interactive layer.
# Keep coordinate/screen helpers, rendering helpers, demo playback helpers, and the
# pygame entrypoint here so the UI layer can later be moved without pulling core logic.

# --- World and demo helpers ---
def clamp_world(x_cm, y_cm):
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    min_x = -halfL - OUTSIDE_MARGIN_CM - GOAL_DEPTH_CM
    max_x = halfL + OUTSIDE_MARGIN_CM + GOAL_DEPTH_CM
    min_y = -halfW - OUTSIDE_MARGIN_CM
    max_y = halfW + OUTSIDE_MARGIN_CM
    x_cm = max(min_x, min(max_x, x_cm))
    y_cm = max(min_y, min(max_y, y_cm))
    if ROUND_CM:
        x_cm = round(x_cm)
        y_cm = round(y_cm)
    return float(x_cm), float(y_cm)


def clamp_world_exact(x_cm, y_cm):
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    min_x = -halfL - OUTSIDE_MARGIN_CM - GOAL_DEPTH_CM
    max_x = halfL + OUTSIDE_MARGIN_CM + GOAL_DEPTH_CM
    min_y = -halfW - OUTSIDE_MARGIN_CM
    max_y = halfW + OUTSIDE_MARGIN_CM
    x_cm = max(min_x, min(max_x, x_cm))
    y_cm = max(min_y, min(max_y, y_cm))
    return float(x_cm), float(y_cm)


def smoothstep01(value):
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def clamp_value(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def set_robot_kinematics(
    robot_fcs_gc,
    vx_cm_s=0.0,
    vy_cm_s=0.0,
    omega_rad_s=0.0,
    ax_cm_s2=0.0,
    ay_cm_s2=0.0,
    alpha_rad_s2=0.0,
):
    robot_fcs_gc["x"][1] = float(vx_cm_s)
    robot_fcs_gc["y"][1] = float(vy_cm_s)
    robot_fcs_gc["theta"][1] = float(omega_rad_s)
    if len(robot_fcs_gc["x"]) > 2:
        robot_fcs_gc["x"][2] = float(ax_cm_s2)
        robot_fcs_gc["y"][2] = float(ay_cm_s2)
        robot_fcs_gc["theta"][2] = float(alpha_rad_s2)
    return robot_fcs_gc


def zero_robot_generalized_motion(robot_fcs_gc):
    for axis in ("x", "y", "theta"):
        for idx in range(1, len(robot_fcs_gc[axis])):
            robot_fcs_gc[axis][idx] = 0.0
    return robot_fcs_gc


def sample_smooth_polyline(points, segment_durations_s, time_s):
    duration_s = sum(segment_durations_s)
    if duration_s <= 0.0:
        x_cm, y_cm = points[0]
        return {"x_cm": x_cm, "y_cm": y_cm, "duration_s": 0.0}

    time_in_cycle_s = time_s % duration_s
    elapsed_s = 0.0
    for idx, segment_duration_s in enumerate(segment_durations_s):
        if time_in_cycle_s <= elapsed_s + segment_duration_s or idx == len(segment_durations_s) - 1:
            t = (time_in_cycle_s - elapsed_s) / segment_duration_s if segment_duration_s > 0.0 else 0.0
            t = smoothstep01(t)
            x0_cm, y0_cm = points[idx]
            x1_cm, y1_cm = points[idx + 1]
            x_cm = x0_cm + (x1_cm - x0_cm) * t
            y_cm = y0_cm + (y1_cm - y0_cm) * t
            return {"x_cm": x_cm, "y_cm": y_cm, "duration_s": duration_s}
        elapsed_s += segment_duration_s

    x_cm, y_cm = points[-1]
    return {"x_cm": x_cm, "y_cm": y_cm, "duration_s": duration_s}


def rotate_xy(x_cm, y_cm, angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return c * x_cm - s * y_cm, s * x_cm + c * y_cm


def demo_phase_warp(unit_phase, amplitude=0.0, harmonic=1.0):
    if harmonic <= 0.0:
        return unit_phase
    return unit_phase + amplitude * math.sin(2.0 * math.pi * harmonic * unit_phase) / (2.0 * math.pi * harmonic)


def world_bounds_cm():
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    return {
        "min_x": -halfL - OUTSIDE_MARGIN_CM - GOAL_DEPTH_CM,
        "max_x": halfL + OUTSIDE_MARGIN_CM + GOAL_DEPTH_CM,
        "min_y": -halfW - OUTSIDE_MARGIN_CM,
        "max_y": halfW + OUTSIDE_MARGIN_CM,
    }


def demo_circle_local_pose(time_s):
    duration_s = 10.0
    radius_cm = 240.0
    unit_phase = (time_s % duration_s) / duration_s
    phase = 2.0 * math.pi * demo_phase_warp(unit_phase, amplitude=0.28, harmonic=1.0)
    x_cm = radius_cm * (math.cos(phase) - 1.0)
    y_cm = radius_cm * math.sin(phase)
    return {"x_cm": x_cm, "y_cm": y_cm, "duration_s": duration_s}


def demo_stand_still_local_pose(time_s):
    return {"x_cm": 0.0, "y_cm": 0.0, "duration_s": 1.0}


def demo_figure_eight_local_pose(time_s):
    duration_s = 12.0
    amp_x_cm = 320.0
    amp_y_cm = 210.0
    unit_phase = (time_s % duration_s) / duration_s
    phase = 2.0 * math.pi * demo_phase_warp(unit_phase, amplitude=0.20, harmonic=2.0)
    x_cm = amp_x_cm * math.sin(phase)
    y_cm = 0.5 * amp_y_cm * math.sin(2.0 * phase)
    return {"x_cm": x_cm, "y_cm": y_cm, "duration_s": duration_s}


def demo_rectangle_local_pose(time_s):
    return sample_smooth_polyline(
        points=[
            (0.0, 0.0),
            (520.0, 0.0),
            (520.0, 300.0),
            (0.0, 300.0),
            (0.0, 0.0),
        ],
        segment_durations_s=[3.0, 2.0, 4.4, 2.6],
        time_s=time_s,
    )


def demo_accel_slalom_local_pose(time_s):
    # A stop-go slalom with clearly visible acceleration and deceleration on each segment.
    return sample_smooth_polyline(
        points=[
            (0.0, 0.0),
            (210.0, 0.0),
            (420.0, 140.0),
            (680.0, -170.0),
            (940.0, 160.0),
            (1180.0, 0.0),
            (980.0, -120.0),
            (720.0, 120.0),
            (420.0, -90.0),
            (0.0, 0.0),
        ],
        segment_durations_s=[1.6, 1.2, 0.9, 1.8, 1.0, 1.5, 1.1, 1.4, 2.0],
        time_s=time_s,
    )


def choose_random_demo_target(current_x_cm, current_y_cm, min_distance_cm=260.0, edge_padding_cm=70.0, max_tries=80):
    world = world_bounds_cm()
    min_x_cm = world["min_x"] + edge_padding_cm
    max_x_cm = world["max_x"] - edge_padding_cm
    min_y_cm = world["min_y"] + edge_padding_cm
    max_y_cm = world["max_y"] - edge_padding_cm

    best_target = (clamp_value(current_x_cm, min_x_cm, max_x_cm), clamp_value(current_y_cm, min_y_cm, max_y_cm))
    best_distance_cm = -1.0
    for _ in range(max_tries):
        target_x_cm = random.uniform(min_x_cm, max_x_cm)
        target_y_cm = random.uniform(min_y_cm, max_y_cm)
        distance_cm = math.hypot(target_x_cm - current_x_cm, target_y_cm - current_y_cm)
        if distance_cm >= min_distance_cm:
            return target_x_cm, target_y_cm
        if distance_cm > best_distance_cm:
            best_distance_cm = distance_cm
            best_target = (target_x_cm, target_y_cm)
    return best_target


def make_random_target_demo_state(start_x_cm, start_y_cm):
    target_x_cm, target_y_cm = choose_random_demo_target(start_x_cm, start_y_cm)
    return {
        "x_cm": float(start_x_cm),
        "y_cm": float(start_y_cm),
        "target_x_cm": target_x_cm,
        "target_y_cm": target_y_cm,
        "speed_cm_s": 0.0,
    }


DEMO_TURN_STEPS = 100
DEMO_TURN_TOTAL_RAD = math.radians(260.0)


DEMO_PATHS = [
    {
        "name": "Stand still",
        "mode": "stand_still",
        "sampler": demo_stand_still_local_pose,
        "turn_speed_rad_s": 2.3,
        "turn_accel_rad_s2": 6.5,
    },
    {
        "name": "Circle",
        "sampler": demo_circle_local_pose,
        "turn_speed_rad_s": 2.3,
        "turn_accel_rad_s2": 6.5,
    },
    {
        "name": "Figure 8",
        "sampler": demo_figure_eight_local_pose,
        "turn_speed_rad_s": 2.9,
        "turn_accel_rad_s2": 8.5,
    },
    {
        "name": "Rectangle",
        "sampler": demo_rectangle_local_pose,
        "turn_speed_rad_s": 2.0,
        "turn_accel_rad_s2": 7.0,
    },
    {
        "name": "Accel slalom",
        "sampler": demo_accel_slalom_local_pose,
        "turn_speed_rad_s": 3.0,
        "turn_accel_rad_s2": 9.0,
    },
    {
        "name": "turn",
        "mode": "turn",
    },
    {
        "name": "Random targets",
        "mode": "random_targets",
        "turn_speed_rad_s": 5.0,
        "turn_accel_rad_s2": 18.0,
        "max_speed_cm_s": 520.0,
        "accel_cm_s2": 900.0,
        "brake_cm_s2": 980.0,
        "target_tolerance_cm": 18.0,
        "stop_speed_cm_s": 30.0,
    },
]


def sample_demo_local_pose(demo_spec, time_s):
    return demo_spec["sampler"](time_s)


def transformed_demo_point(local_pose, anchor_x_cm, anchor_y_cm, scale, rotation_rad):
    x_local_cm, y_local_cm = rotate_xy(scale * local_pose["x_cm"], scale * local_pose["y_cm"], rotation_rad)
    return anchor_x_cm + x_local_cm, anchor_y_cm + y_local_cm


def heading_to_target(x_cm, y_cm, target_x_cm=0.0, target_y_cm=0.0, fallback_theta_rad=0.0):
    dx_cm = target_x_cm - x_cm
    dy_cm = target_y_cm - y_cm
    if abs(dx_cm) < 1e-9 and abs(dy_cm) < 1e-9:
        return wrap_rad(fallback_theta_rad)
    return wrap_rad(math.atan2(dy_cm, dx_cm))


def step_heading_dynamics(current_theta_rad, current_omega_rad_s, target_theta_rad, dt_s, max_speed_rad_s, max_accel_rad_s2):
    if dt_s <= 0.0:
        return wrap_rad(current_theta_rad), float(current_omega_rad_s), 0.0

    angle_error_rad = wrap_rad(target_theta_rad - current_theta_rad)
    if abs(angle_error_rad) < 1e-9 and abs(current_omega_rad_s) < 1e-9:
        return wrap_rad(target_theta_rad), 0.0, 0.0

    desired_omega_mag = min(max_speed_rad_s, math.sqrt(max(0.0, 2.0 * max_accel_rad_s2 * abs(angle_error_rad))))
    desired_omega_rad_s = math.copysign(desired_omega_mag, angle_error_rad) if abs(angle_error_rad) > 1e-12 else 0.0

    # POC simulator heading control:
    # Keep the simple bang-bang turn controller, but ramp the acceleration clamp
    # down around near-zero look-at-ball error. This avoids a hard jump from the
    # small near-target clamp to the full turn acceleration, which shows up as
    # narrow theta acceleration spikes in reports.
    near_error_ratio = min(1.0, abs(angle_error_rad) / DEMO_TURN_NEAR_ZERO_ERROR_RAD)
    effective_max_accel_rad_s2 = (
        DEMO_TURN_NEAR_ZERO_ACCEL_RAD_S2 +
        near_error_ratio * (max_accel_rad_s2 - DEMO_TURN_NEAR_ZERO_ACCEL_RAD_S2)
    )

    omega_delta_rad_s = clamp_value(
        desired_omega_rad_s - current_omega_rad_s,
        -effective_max_accel_rad_s2 * dt_s,
        effective_max_accel_rad_s2 * dt_s,
    )
    next_omega_rad_s = clamp_value(
        current_omega_rad_s + omega_delta_rad_s,
        -max_speed_rad_s,
        max_speed_rad_s,
    )
    alpha_rad_s2 = (next_omega_rad_s - current_omega_rad_s) / dt_s
    next_theta_rad = wrap_rad(current_theta_rad + next_omega_rad_s * dt_s)

    return next_theta_rad, next_omega_rad_s, alpha_rad_s2


def estimate_demo_local_heading(demo_spec, time_s=0.0, dt_s=1e-3):
    pose_now = sample_demo_local_pose(demo_spec, time_s)
    pose_next = sample_demo_local_pose(demo_spec, time_s + dt_s)
    dx_cm = pose_next["x_cm"] - pose_now["x_cm"]
    dy_cm = pose_next["y_cm"] - pose_now["y_cm"]
    if abs(dx_cm) < 1e-9 and abs(dy_cm) < 1e-9:
        return 0.0
    return math.atan2(dy_cm, dx_cm)


def sample_demo_pose(
    demo_spec,
    time_s,
    anchor_x_cm=0.0,
    anchor_y_cm=0.0,
    scale=1.0,
    rotation_rad=0.0,
    look_target_x_cm=0.0,
    look_target_y_cm=0.0,
    fallback_theta_rad=0.0,
):
    local_pose = sample_demo_local_pose(demo_spec, time_s)
    x_cm, y_cm = transformed_demo_point(local_pose, anchor_x_cm, anchor_y_cm, scale, rotation_rad)
    theta_rad = heading_to_target(
        x_cm,
        y_cm,
        target_x_cm=look_target_x_cm,
        target_y_cm=look_target_y_cm,
        fallback_theta_rad=fallback_theta_rad,
    )

    return {
        "x_cm": x_cm,
        "y_cm": y_cm,
        "theta_rad": wrap_rad(theta_rad),
        "duration_s": local_pose["duration_s"],
    }


def estimate_demo_local_bounds(demo_spec, rotation_rad, samples=240):
    base_pose = sample_demo_local_pose(demo_spec, 0.0)
    duration_s = base_pose["duration_s"]
    min_x_cm = float("inf")
    max_x_cm = float("-inf")
    min_y_cm = float("inf")
    max_y_cm = float("-inf")
    for idx in range(samples + 1):
        time_s = duration_s * idx / samples if samples > 0 else 0.0
        local_pose = sample_demo_local_pose(demo_spec, time_s)
        x_cm, y_cm = rotate_xy(local_pose["x_cm"], local_pose["y_cm"], rotation_rad)
        min_x_cm = min(min_x_cm, x_cm)
        max_x_cm = max(max_x_cm, x_cm)
        min_y_cm = min(min_y_cm, y_cm)
        max_y_cm = max(max_y_cm, y_cm)
    return {
        "min_x_cm": min_x_cm,
        "max_x_cm": max_x_cm,
        "min_y_cm": min_y_cm,
        "max_y_cm": max_y_cm,
    }


def compute_demo_fit_scale(bounds, anchor_x_cm, anchor_y_cm, max_scale=1.0):
    world = world_bounds_cm()
    scale_limits = [max_scale]
    if bounds["min_x_cm"] < 0.0:
        scale_limits.append((anchor_x_cm - world["min_x"]) / (-bounds["min_x_cm"]))
    if bounds["max_x_cm"] > 0.0:
        scale_limits.append((world["max_x"] - anchor_x_cm) / bounds["max_x_cm"])
    if bounds["min_y_cm"] < 0.0:
        scale_limits.append((anchor_y_cm - world["min_y"]) / (-bounds["min_y_cm"]))
    if bounds["max_y_cm"] > 0.0:
        scale_limits.append((world["max_y"] - anchor_y_cm) / bounds["max_y_cm"])
    return max(0.0, min(scale_limits))


def choose_demo_transform(demo_spec, anchor_x_cm, anchor_y_cm, desired_heading_rad):
    initial_local_heading_rad = estimate_demo_local_heading(demo_spec)
    rotation_rad = wrap_rad(desired_heading_rad - initial_local_heading_rad)
    bounds = estimate_demo_local_bounds(demo_spec, rotation_rad)
    scale = compute_demo_fit_scale(bounds, anchor_x_cm, anchor_y_cm)
    return rotation_rad, scale


def reset_demo_playback_state(robot_fcs_gc, demo_state):
    demo_spec = demo_state["paths"][demo_state["selected_idx"]]
    pose = robot_pose_view(robot_fcs_gc)
    anchor_x_cm = pose["x_cm"]
    anchor_y_cm = pose["y_cm"]
    start_theta_rad = pose["theta_rad"]
    look_target_x_cm = float(demo_state.get("look_target_x_cm", 0.0))
    look_target_y_cm = float(demo_state.get("look_target_y_cm", 0.0))
    if demo_spec.get("mode") == "random_targets":
        demo_state["pending_reset"] = False
        demo_state["time_s"] = 0.0
        demo_state["sim_accumulator_s"] = 0.0
        demo_state["anchor_x_cm"] = anchor_x_cm
        demo_state["anchor_y_cm"] = anchor_y_cm
        demo_state["rotation_rad"] = 0.0
        demo_state["scale"] = 1.0
        demo_state["trace"] = [(anchor_x_cm, anchor_y_cm)]
        demo_state["path_state"] = make_random_target_demo_state(anchor_x_cm, anchor_y_cm)
        scalar_2_gc(
            robot_fcs_gc,
            x_cm=anchor_x_cm,
            y_cm=anchor_y_cm,
            theta_rad=start_theta_rad,
        )
        zero_robot_generalized_motion(robot_fcs_gc)
        return

    if demo_spec.get("mode") == "turn":
        demo_state["pending_reset"] = False
        demo_state["time_s"] = 0.0
        demo_state["sim_accumulator_s"] = 0.0
        demo_state["anchor_x_cm"] = anchor_x_cm
        demo_state["anchor_y_cm"] = anchor_y_cm
        demo_state["rotation_rad"] = 0.0
        demo_state["scale"] = 1.0
        demo_state["trace"] = [(anchor_x_cm, anchor_y_cm)]
        demo_state["turn_start_theta_rad"] = start_theta_rad
        scalar_2_gc(
            robot_fcs_gc,
            x_cm=anchor_x_cm,
            y_cm=anchor_y_cm,
            theta_rad=start_theta_rad,
        )
        zero_robot_generalized_motion(robot_fcs_gc)
        return

    rotation_rad, scale = choose_demo_transform(demo_spec, anchor_x_cm, anchor_y_cm, start_theta_rad)
    start_pose = sample_demo_pose(
        demo_spec,
        0.0,
        anchor_x_cm=anchor_x_cm,
        anchor_y_cm=anchor_y_cm,
        scale=scale,
        rotation_rad=rotation_rad,
        look_target_x_cm=look_target_x_cm,
        look_target_y_cm=look_target_y_cm,
        fallback_theta_rad=start_theta_rad,
    )
    demo_state["pending_reset"] = False
    demo_state["time_s"] = 0.0
    demo_state["sim_accumulator_s"] = 0.0
    demo_state["anchor_x_cm"] = anchor_x_cm
    demo_state["anchor_y_cm"] = anchor_y_cm
    demo_state["rotation_rad"] = rotation_rad
    demo_state["scale"] = scale
    demo_state["trace"] = [(start_pose["x_cm"], start_pose["y_cm"])]
    scalar_2_gc(
        robot_fcs_gc,
        x_cm=start_pose["x_cm"],
        y_cm=start_pose["y_cm"],
        theta_rad=start_theta_rad,
    )
    zero_robot_generalized_motion(robot_fcs_gc)


def append_demo_trace_point(trace, x_cm, y_cm):
    point = (float(x_cm), float(y_cm))
    if not trace or math.hypot(trace[-1][0] - point[0], trace[-1][1] - point[1]) >= 2.0:
        trace.append(point)
        if len(trace) > DEMO_TRACE_MAX_POINTS:
            del trace[: len(trace) - DEMO_TRACE_MAX_POINTS]


def step_demo_playback(robot_fcs_gc, demo_state, dt_s):
    if dt_s <= 0.0:
        return

    demo_spec = demo_state["paths"][demo_state["selected_idx"]]
    max_turn_speed_rad_s = float(demo_spec.get("turn_speed_rad_s", DEMO_TURN_MAX_SPEED_RAD_S))
    max_turn_accel_rad_s2 = float(demo_spec.get("turn_accel_rad_s2", DEMO_TURN_MAX_ACCEL_RAD_S2))
    look_target_x_cm = float(demo_state.get("look_target_x_cm", 0.0))
    look_target_y_cm = float(demo_state.get("look_target_y_cm", 0.0))
    pose = robot_pose_view(robot_fcs_gc)
    if demo_spec.get("mode") == "turn":
        turn_duration_s = DEMO_TURN_STEPS * dt_s
        prev_phase = demo_state["time_s"] / turn_duration_s
        next_time_s = demo_state["time_s"] + dt_s
        next_phase = next_time_s / turn_duration_s
        start_theta_rad = float(demo_state.get("turn_start_theta_rad", pose["theta_rad"]))
        prev_theta_rad = wrap_rad(start_theta_rad + DEMO_TURN_TOTAL_RAD * prev_phase)
        next_theta_rad = wrap_rad(start_theta_rad + DEMO_TURN_TOTAL_RAD * next_phase)
        omega_rad_s = wrap_rad(next_theta_rad - prev_theta_rad) / dt_s
        alpha_rad_s2 = (omega_rad_s - robot_fcs_gc["theta"][1]) / dt_s
        anchor_x_cm = float(demo_state.get("anchor_x_cm", pose["x_cm"]))
        anchor_y_cm = float(demo_state.get("anchor_y_cm", pose["y_cm"]))

        scalar_2_gc(
            robot_fcs_gc,
            x_cm=anchor_x_cm,
            y_cm=anchor_y_cm,
            theta_rad=next_theta_rad,
        )
        set_robot_kinematics(
            robot_fcs_gc,
            vx_cm_s=0.0,
            vy_cm_s=0.0,
            omega_rad_s=omega_rad_s,
            ax_cm_s2=0.0,
            ay_cm_s2=0.0,
            alpha_rad_s2=alpha_rad_s2,
        )
        append_demo_trace_point(demo_state["trace"], anchor_x_cm, anchor_y_cm)
        demo_state["time_s"] = next_time_s
        return

    if demo_spec.get("mode") == "random_targets":
        path_state = demo_state.setdefault("path_state", make_random_target_demo_state(pose["x_cm"], pose["y_cm"]))
        target_x_cm = float(path_state["target_x_cm"])
        target_y_cm = float(path_state["target_y_cm"])
        current_x_cm = float(path_state.get("x_cm", pose["x_cm"]))
        current_y_cm = float(path_state.get("y_cm", pose["y_cm"]))
        dx_cm = target_x_cm - current_x_cm
        dy_cm = target_y_cm - current_y_cm
        distance_cm = math.hypot(dx_cm, dy_cm)

        speed_cm_s = float(path_state.get("speed_cm_s", math.hypot(robot_fcs_gc["x"][1], robot_fcs_gc["y"][1])))
        max_speed_cm_s = float(demo_spec.get("max_speed_cm_s", 500.0))
        accel_cm_s2 = float(demo_spec.get("accel_cm_s2", 500.0))
        brake_cm_s2 = float(demo_spec.get("brake_cm_s2", 900.0))
        target_tolerance_cm = float(demo_spec.get("target_tolerance_cm", 20.0))
        stop_speed_cm_s = float(demo_spec.get("stop_speed_cm_s", 25.0))

        if distance_cm <= target_tolerance_cm and speed_cm_s <= stop_speed_cm_s:
            # Switch targets from the current position without snapping to the old target.
            target_x_cm, target_y_cm = choose_random_demo_target(current_x_cm, current_y_cm)
            path_state["target_x_cm"] = target_x_cm
            path_state["target_y_cm"] = target_y_cm
            dx_cm = target_x_cm - current_x_cm
            dy_cm = target_y_cm - current_y_cm
            distance_cm = math.hypot(dx_cm, dy_cm)

        if distance_cm > 1e-9:
            dir_x = dx_cm / distance_cm
            dir_y = dy_cm / distance_cm
        else:
            dir_x = 0.0
            dir_y = 0.0

        stopping_distance_cm = (speed_cm_s * speed_cm_s) / (2.0 * brake_cm_s2) if brake_cm_s2 > 1e-9 else 0.0
        if distance_cm <= stopping_distance_cm + target_tolerance_cm:
            next_speed_cm_s = max(0.0, speed_cm_s - brake_cm_s2 * dt_s)
        else:
            next_speed_cm_s = min(max_speed_cm_s, speed_cm_s + accel_cm_s2 * dt_s)

        travel_cm = min(distance_cm, next_speed_cm_s * dt_s)
        next_x_cm = current_x_cm + dir_x * travel_cm
        next_y_cm = current_y_cm + dir_y * travel_cm
        next_x_cm, next_y_cm = clamp_world_exact(next_x_cm, next_y_cm)
        actual_dx_cm = next_x_cm - current_x_cm
        actual_dy_cm = next_y_cm - current_y_cm
        vx_cm_s = actual_dx_cm / dt_s
        vy_cm_s = actual_dy_cm / dt_s
        ax_cm_s2 = (vx_cm_s - robot_fcs_gc["x"][1]) / dt_s
        ay_cm_s2 = (vy_cm_s - robot_fcs_gc["y"][1]) / dt_s
        path_state["x_cm"] = next_x_cm
        path_state["y_cm"] = next_y_cm
        path_state["speed_cm_s"] = math.hypot(vx_cm_s, vy_cm_s)

        target_theta_rad = heading_to_target(
            next_x_cm,
            next_y_cm,
            target_x_cm=look_target_x_cm,
            target_y_cm=look_target_y_cm,
            fallback_theta_rad=pose["theta_rad"],
        )
        next_theta_rad, omega_rad_s, alpha_rad_s2 = step_heading_dynamics(
            pose["theta_rad"],
            robot_fcs_gc["theta"][1],
            target_theta_rad,
            dt_s,
            max_turn_speed_rad_s,
            max_turn_accel_rad_s2,
        )

        scalar_2_gc(
            robot_fcs_gc,
            x_cm=next_x_cm,
            y_cm=next_y_cm,
            theta_rad=next_theta_rad,
        )
        set_robot_kinematics(
            robot_fcs_gc,
            vx_cm_s=vx_cm_s,
            vy_cm_s=vy_cm_s,
            omega_rad_s=omega_rad_s,
            ax_cm_s2=ax_cm_s2,
            ay_cm_s2=ay_cm_s2,
            alpha_rad_s2=alpha_rad_s2,
        )
        append_demo_trace_point(demo_state["trace"], next_x_cm, next_y_cm)
        demo_state["time_s"] += dt_s
        return

    prev_pose = sample_demo_pose(
        demo_spec,
        demo_state["time_s"],
        anchor_x_cm=demo_state.get("anchor_x_cm", pose["x_cm"]),
        anchor_y_cm=demo_state.get("anchor_y_cm", pose["y_cm"]),
        scale=demo_state.get("scale", 1.0),
        rotation_rad=demo_state.get("rotation_rad", 0.0),
        look_target_x_cm=look_target_x_cm,
        look_target_y_cm=look_target_y_cm,
        fallback_theta_rad=pose["theta_rad"],
    )
    next_time_s = demo_state["time_s"] + dt_s
    next_pose = sample_demo_pose(
        demo_spec,
        next_time_s,
        anchor_x_cm=demo_state.get("anchor_x_cm", pose["x_cm"]),
        anchor_y_cm=demo_state.get("anchor_y_cm", pose["y_cm"]),
        scale=demo_state.get("scale", 1.0),
        rotation_rad=demo_state.get("rotation_rad", 0.0),
        look_target_x_cm=look_target_x_cm,
        look_target_y_cm=look_target_y_cm,
        fallback_theta_rad=prev_pose["theta_rad"],
    )

    prev_vx_cm_s = robot_fcs_gc["x"][1]
    prev_vy_cm_s = robot_fcs_gc["y"][1]
    prev_omega_rad_s = robot_fcs_gc["theta"][1]
    vx_cm_s = (next_pose["x_cm"] - prev_pose["x_cm"]) / dt_s
    vy_cm_s = (next_pose["y_cm"] - prev_pose["y_cm"]) / dt_s
    ax_cm_s2 = (vx_cm_s - prev_vx_cm_s) / dt_s
    ay_cm_s2 = (vy_cm_s - prev_vy_cm_s) / dt_s
    next_x_cm, next_y_cm = clamp_world(next_pose["x_cm"], next_pose["y_cm"])
    next_theta_rad, omega_rad_s, alpha_rad_s2 = step_heading_dynamics(
        pose["theta_rad"],
        prev_omega_rad_s,
        next_pose["theta_rad"],
        dt_s,
        max_turn_speed_rad_s,
        max_turn_accel_rad_s2,
    )

    scalar_2_gc(
        robot_fcs_gc,
        x_cm=next_x_cm,
        y_cm=next_y_cm,
        theta_rad=next_theta_rad,
    )
    set_robot_kinematics(
        robot_fcs_gc,
        vx_cm_s=vx_cm_s,
        vy_cm_s=vy_cm_s,
        omega_rad_s=omega_rad_s,
        ax_cm_s2=ax_cm_s2,
        ay_cm_s2=ay_cm_s2,
        alpha_rad_s2=alpha_rad_s2,
    )
    append_demo_trace_point(demo_state["trace"], next_x_cm, next_y_cm)
    demo_state["time_s"] = next_time_s


def capture_paper_lm_setup(robots, ball, demo_state, algorithm_name, profile_name):
    """Save the UI selection and fixed starting poses for the later P-LM batch."""
    from copy import deepcopy

    settings = experiment_settings(algorithm_name)
    settings["profile_name"] = profile_name
    settings["demo_path_name"] = demo_state["paths"][demo_state["selected_idx"]]["name"]
    # Save positions/headings exactly; every trial starts with zero motion.
    saved_robots = {
        name: make_robot_gc_state(*gc_2_scalar(robots[name]))
        for name in ("actual", "belief")
    }
    saved_ball = dict(ball)
    saved_demo = {
        "paths": deepcopy(demo_state["paths"]),
        "selected_idx": demo_state["selected_idx"],
        "playing": False,
        "pending_reset": True,
        "look_target_x_cm": saved_ball["x_cm"],
        "look_target_y_cm": saved_ball["y_cm"],
    }
    # Give random-target playback its own repeatable sequence, independent of
    # sensor noise and of how many landmarks a trial observes.
    sensor_random_state = random.getstate()
    try:
        reset_demo_playback_state(deepcopy(saved_robots["actual"]), saved_demo)
        path_random_state = random.getstate()
    finally:
        random.setstate(sensor_random_state)
    return {
        "runs": CONVERGE_TEST_RUNS,
        "algorithm_name": algorithm_name,
        "settings": settings,
        "robots": saved_robots,
        "ball": saved_ball,
        "demo_state": saved_demo,
        "path_random_state": path_random_state,
    }


def step_paper_lm_demo(experiment):
    """Replay the saved path without consuming the sensor random sequence."""
    sensor_random_state = random.getstate()
    try:
        random.setstate(experiment["path_random_state"])
        step_demo_playback(
            experiment["robots"]["actual"],
            experiment["demo_state"],
            experiment["settings"]["dt"],
        )
        experiment["path_random_state"] = random.getstate()
    finally:
        random.setstate(sensor_random_state)


def export_demo_tracking_dataset(
    robots,
    algorithm_name,
    demo_paths,
    selected_demo_idx,
    max_steps=None,
    duration_s=None,
    profile_name=None,
    look_target_x_cm=0.0,
    look_target_y_cm=0.0,
    progress_callback=None,
    filename_prefix="paper_visibility_stress_dataset",
):
    import csv
    import datetime
    import json
    import os

    if max_steps is None:
        if duration_s is None:
            duration_s = PAPER_VISIBILITY_STRESS_DURATION_S
        max_steps = step_count_for_duration(duration_s)
    else:
        max_steps = max(1, int(max_steps))

    timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(reports_dir, f"{filename_prefix}_{timestamp_file}.csv")
    json_path = os.path.join(reports_dir, f"{filename_prefix}_{timestamp_file}.json")

    demo_path_name = demo_paths[selected_demo_idx]["name"]
    actual_start = robot_pose_view(robots["actual"])
    belief_start = robot_pose_view(robots["belief"])
    settings = experiment_settings(algorithm_name)
    if profile_name is not None:
        settings["profile_name"] = profile_name
    settings["demo_path_name"] = demo_path_name
    settings["export_steps"] = max_steps
    settings["export_duration_s"] = max_steps * DT

    local_demo_state = {
        "paths": demo_paths,
        "selected_idx": selected_demo_idx,
        "playing": False,
        "pending_reset": True,
        "time_s": 0.0,
        "sim_accumulator_s": 0.0,
        "anchor_x_cm": actual_start["x_cm"],
        "anchor_y_cm": actual_start["y_cm"],
        "rotation_rad": 0.0,
        "scale": 1.0,
        "trace": [(actual_start["x_cm"], actual_start["y_cm"])],
        "look_target_x_cm": look_target_x_cm,
        "look_target_y_cm": look_target_y_cm,
    }
    reset_landmark_obstruction_state()
    reset_motion_slip_state()
    reset_demo_playback_state(robots["actual"], local_demo_state)

    fieldnames = [
        "step",
        "time_s",
        "actual_x_cm",
        "actual_y_cm",
        "actual_theta_rad",
        "belief_x_cm",
        "belief_y_cm",
        "belief_theta_rad",
        "position_error_cm",
        "heading_error_deg",
        "visible_landmark_count",
        "visible_landmark_ids",
        "obstruction_active_count",
        "precision_distance_modifier_enabled",
        "free_energy",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for k in range(1, max_steps + 1):
            step_demo_playback(robots["actual"], local_demo_state, DT)
            y_obs_map, yhat_map, y_motion_obs, free_energy = simulation_inference_step(robots, algorithm_name)
            record = snapshot_record(k, robots, y_obs_map, yhat_map, y_motion_obs, free_energy, settings)
            writer.writerow(compact_tracking_export_row(record, k * DT))

            if progress_callback is not None and (k == 1 or k % 5000 == 0 or k == max_steps):
                progress_callback(k, max_steps, csv_path)

    metadata = {
        "csv_path": csv_path,
        "steps": max_steps,
        "duration_s": max_steps * DT,
        "algorithm_name": algorithm_name,
        "demo_path_name": demo_path_name,
        "actual_start": actual_start,
        "belief_start": belief_start,
        "actual_end": robot_pose_view(robots["actual"]),
        "belief_end": robot_pose_view(robots["belief"]),
        "settings": settings,
    }
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)

    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "steps": max_steps,
        "duration_s": max_steps * DT,
    }


def demo_path_index_by_name(path_name, demo_paths=DEMO_PATHS):
    for idx, demo_path in enumerate(demo_paths):
        if demo_path["name"].lower() == path_name.lower():
            return idx
    valid_names = ", ".join(path["name"] for path in demo_paths)
    raise ValueError(f"Unknown demo path '{path_name}'. Use one of: {valid_names}")


def run_headless_paper_dataset_export(
    steps=None,
    duration_s=None,
    profile_name="40Hz",
    demo_path_name="Circle",
    algorithm_name=FUSED_MOTION_LANDMARK_ALGORITHM_NAME,
    apply_paper_stress=False,
    start_mode="fixed_offset",
    filename_prefix="paper_visibility_stress_dataset_FOR_PAPER",
    association_name=None,
    precision_distance_modifier_enabled=True,
):
    global PRECISION_DISTANCE_MODIFIER_ENABLED
    PRECISION_DISTANCE_MODIFIER_ENABLED = bool(precision_distance_modifier_enabled)
    apply_masterdata_setting_profile(
        masterdata_setting_profile_by_name(profile_name, PRECISION_DISTANCE_MODIFIER_ENABLED)
    )
    if association_name is None:
        association_name = DATA_ASSOCIATION_UNIQUE_ID_NAME if apply_paper_stress else active_association_name()
    if apply_paper_stress:
        apply_visibility_stress_paper_conditions(association_name)
    else:
        select_association_profile(association_name)
    selected_demo_idx = demo_path_index_by_name(demo_path_name)
    if start_mode in {"nearby", "random"}:
        actual_start, belief_start = make_converge_start_pair(start_mode)
    elif start_mode == "converged":
        actual_start = random_pose_inside_field()
        belief_start = dict(actual_start)
    elif start_mode == "fixed_offset":
        actual_start = {"x": 0.0, "y": 0.0, "theta": 0.0}
        belief_start = {"x": 0.0, "y": -200.0, "theta": -math.pi / 2.0}
    else:
        raise ValueError("paper data start mode must be one of: fixed_offset, nearby, random, converged")
    robots = {
        "actual": make_robot_gc_state(actual_start["x"], actual_start["y"], wrap_rad(actual_start["theta"])),
        "belief": make_robot_gc_state(belief_start["x"], belief_start["y"], wrap_rad(belief_start["theta"])),
    }

    def progress_update(current, total, csv_path):
        if current % 5000 == 0 or current == 1 or current == total:
            pct = 100.0 * current / total if total > 0 else 100.0
            print(f"paper data: {current}/{total} ({pct:.1f}%) -> {csv_path}")

    return export_demo_tracking_dataset(
        robots,
        algorithm_name,
        DEMO_PATHS,
        selected_demo_idx,
        max_steps=steps,
        duration_s=duration_s,
        profile_name=profile_name,
        look_target_x_cm=300.0,
        look_target_y_cm=0.0,
        progress_callback=progress_update,
        filename_prefix=filename_prefix,
    )


def run_dynamic_converge_test(
    robots,
    algorithm_name,
    demo_paths,
    selected_demo_idx,
    runs=CONVERGE_TEST_RUNS,
    progress_callback=None,
    start_mode="random",
    start_cases=None,
    look_target_x_cm=0.0,
    look_target_y_cm=0.0,
    profile_name=None,
    stop_check=None,
    near_start_limits=None,
):
    converged = 0
    non_converged_symmetrical = 0
    non_converged = 0
    failures = []
    last_history = []
    trial_steps = []
    converged_steps = []
    near_start_limits = dict(NEAR_START_LIMITS if near_start_limits is None else near_start_limits)
    stopped = False
    demo_path_name = demo_paths[selected_demo_idx]["name"]

    if start_cases is not None:
        trial_specs = [
            {"actual_start": case["actual"], "belief_start": case["belief"]}
            for case in start_cases
        ]
        runs = len(trial_specs)
    else:
        trial_specs = [None] * runs

    for i in range(runs):
        if stop_check is not None and stop_check():
            stopped = True
            break

        if trial_specs[i] is None:
            actual_start, belief_start = make_converge_start_pair(start_mode, near_start_limits=near_start_limits)
        else:
            actual_start = dict(trial_specs[i]["actual_start"])
            belief_start = dict(trial_specs[i]["belief_start"])

        scalar_2_gc(robots["actual"], x_cm=actual_start["x"], y_cm=actual_start["y"], theta_rad=actual_start["theta"])
        scalar_2_gc(robots["belief"], x_cm=belief_start["x"], y_cm=belief_start["y"], theta_rad=belief_start["theta"])
        zero_robot_generalized_motion(robots["actual"])
        zero_robot_generalized_motion(robots["belief"])

        local_demo_state = {
            "paths": demo_paths,
            "selected_idx": selected_demo_idx,
            "playing": False,
            "pending_reset": True,
            "time_s": 0.0,
            "sim_accumulator_s": 0.0,
            "anchor_x_cm": actual_start["x"],
            "anchor_y_cm": actual_start["y"],
            "rotation_rad": 0.0,
            "scale": 1.0,
            "trace": [(actual_start["x"], actual_start["y"])],
            "look_target_x_cm": look_target_x_cm,
            "look_target_y_cm": look_target_y_cm,
        }
        reset_demo_playback_state(robots["actual"], local_demo_state)

        start_actual = robot_pose_view(robots["actual"])
        start_belief = robot_pose_view(robots["belief"])
        settings = experiment_settings(algorithm_name)
        settings["near_start_limits"] = dict(near_start_limits)
        if profile_name is not None:
            settings["profile_name"] = profile_name
        settings["demo_path_name"] = demo_path_name

        history = []
        convergence_hits = 0
        symmetrical_hits = 0
        for k in range(1, INFER_STEPS_MAX + 1):
            if stop_check is not None and stop_check():
                stopped = True
                break

            step_demo_playback(robots["actual"], local_demo_state, DT)
            y_obs_map, yhat_map, y_motion_obs, free_energy = simulation_inference_step(robots, algorithm_name)
            history.append(snapshot_record(k, robots, y_obs_map, yhat_map, y_motion_obs, free_energy, settings))

            if pose_converged_ros(
                robot_pose_view(robots["actual"]),
                robot_pose_view(robots["belief"]),
                CONV_TOL_X_CM,
                CONV_TOL_Y_CM,
                CONV_TOL_ANGLE_RAD,
            ):
                convergence_hits += 1
                if convergence_hits >= CONV_REQUIRED_SEQUENTIAL_HITS:
                    break
            else:
                convergence_hits = 0

            if pose_symmetrically_converged(history[-1]["actual_pose"], history[-1]["belief_pose"]):
                symmetrical_hits += 1
                if symmetrical_hits >= CONV_REQUIRED_SEQUENTIAL_HITS:
                    break
            else:
                symmetrical_hits = 0

        last_history = history
        steps_taken = len(history)
        trial_steps.append(steps_taken)

        if stopped:
            break

        if history and history_has_sequential_convergence(history):
            converged += 1
            converged_steps.append(steps_taken)
        else:
            symmetrical = history_has_sequential_convergence(history, symmetrical=True)
            if symmetrical:
                non_converged_symmetrical += 1
            else:
                non_converged += 1
            actual_end = robot_pose_view(robots["actual"])
            belief_end = robot_pose_view(robots["belief"])
            visible_landmark_ids = sorted(obs["obs_id"] for obs in simulation_sense_landmarks(robots["actual"], active_simulation_landmarks()))
            failures.append({
                "symmetrical": symmetrical,
                "actual_start": start_actual,
                "belief_start": start_belief,
                "actual_end": actual_end,
                "belief_end": belief_end,
                "actual_start_outside": is_pose_outside_field(start_actual),
                "belief_start_outside": is_pose_outside_field(start_belief),
                "visible_landmark_count": len(visible_landmark_ids),
                "visible_landmark_ids": visible_landmark_ids,
                "path_history": history[-300:],
            })

        if progress_callback is not None:
            keep_running = progress_callback(i + 1, runs, converged, non_converged + non_converged_symmetrical)
            if keep_running is False:
                stopped = True
                break

    mean_steps, sd_steps = step_stats(converged_steps)
    max_steps = max(converged_steps) if converged_steps else None
    p95_steps = percentile_step(converged_steps, 95)
    p99_steps = percentile_step(converged_steps, 99)
    return {
        "runs": runs,
        "converged": converged,
        "non_converged_symmetrical": non_converged_symmetrical,
        "non_converged": non_converged,
        "trial_steps": trial_steps,
        "converged_steps": converged_steps,
        "mean_steps_to_converge": mean_steps,
        "sd_steps_to_converge": sd_steps,
        "max_steps_to_converge": max_steps,
        "p95_steps_to_converge": p95_steps,
        "p99_steps_to_converge": p99_steps,
        "failures": failures,
        "last_history": last_history,
        "start_mode": f"dynamic_demo_{start_mode}",
        "near_start_limits": near_start_limits,
        "demo_path_name": demo_path_name,
        "look_target_x_cm": look_target_x_cm,
        "look_target_y_cm": look_target_y_cm,
        "stopped": stopped,
    }

# --- Coordinate and rendering helpers ---
def compute_scale(world_w_px, world_h_px):
    world_width_cm = FIELD_WIDTH_CM + 2 * OUTSIDE_MARGIN_CM
    world_length_cm = FIELD_LENGTH_CM + 2 * (OUTSIDE_MARGIN_CM + GOAL_DEPTH_CM)
    usable_w = world_w_px - 2 * MARGIN_PX
    usable_h = world_h_px - 2 * MARGIN_PX
    sx = usable_w / world_width_cm
    sy = usable_h / world_length_cm
    return min(sx, sy)


def fcs_to_screen(x_cm, y_cm, world_w_px, world_h_px, scale):
    cx = world_w_px / 2
    cy = world_h_px / 2
    x_px = cx - y_cm * scale
    y_px = cy - x_cm * scale
    return int(x_px), int(y_px)


def screen_to_fcs(x_px, y_px, world_w_px, world_h_px, scale):
    cx = world_w_px / 2
    cy = world_h_px / 2
    x_cm = (cy - y_px) / scale
    y_cm = (cx - x_px) / scale
    if ROUND_CM:
        x_cm = round(x_cm)
        y_cm = round(y_cm)
    return float(x_cm), float(y_cm)


# --- Rendering helpers ---
def rect_from_corners_fcs(x1, y1, x2, y2, world_w_px, world_h_px, scale):
    p1 = fcs_to_screen(x1, y1, world_w_px, world_h_px, scale)
    p2 = fcs_to_screen(x2, y2, world_w_px, world_h_px, scale)
    left = min(p1[0], p2[0])
    top = min(p1[1], p2[1])
    w = abs(p1[0] - p2[0])
    h = abs(p1[1] - p2[1])
    return pygame.Rect(left, top, w, h)


def draw_button(screen, rect, text, font, bg, fg):
    pygame.draw.rect(screen, bg, rect, border_radius=8)
    pygame.draw.rect(screen, (255, 255, 255), rect, width=2, border_radius=8)
    surf = font.render(text, True, fg)
    tx = rect.x + (rect.w - surf.get_width()) // 2
    ty = rect.y + (rect.h - surf.get_height()) // 2
    screen.blit(surf, (tx, ty))


def toggle_color(enabled):
    return (90, 255, 120) if enabled else (255, 95, 95)


def draw_toggle_mark(screen, x, center_y, enabled):
    color = toggle_color(enabled)
    if enabled:
        pygame.draw.line(screen, color, (x, center_y + 1), (x + 5, center_y + 6), 3)
        pygame.draw.line(screen, color, (x + 5, center_y + 6), (x + 14, center_y - 6), 3)
    else:
        pygame.draw.line(screen, color, (x, center_y - 6), (x + 12, center_y + 6), 3)
        pygame.draw.line(screen, color, (x + 12, center_y - 6), (x, center_y + 6), 3)


def draw_toggle_button(screen, rect, label, enabled, font):
    bg = (45, 45, 45)
    pygame.draw.rect(screen, bg, rect, border_radius=8)
    pygame.draw.rect(screen, (255, 255, 255), rect, width=2, border_radius=8)

    label_surf = font.render(f"{label} ", True, (255, 255, 255))
    mark_w = 16
    total_w = label_surf.get_width() + mark_w
    x = rect.x + (rect.w - total_w) // 2
    y = rect.y + (rect.h - label_surf.get_height()) // 2
    screen.blit(label_surf, (x, y))
    draw_toggle_mark(screen, x + label_surf.get_width() + 1, rect.centery, enabled)


def fit_button_label(text, font, max_width_px):
    text = str(text)
    if font.size(text)[0] <= max_width_px:
        return text
    ellipsis = "..."
    if font.size(ellipsis)[0] > max_width_px:
        return ""
    compact_text = text
    while compact_text and font.size(compact_text + ellipsis)[0] > max_width_px:
        compact_text = compact_text[:-1]
    return compact_text + ellipsis


def draw_input_box(screen, rect, text, font, active, suffix="cm"):
    bg = (40, 70, 120) if active else (45, 45, 45)
    border = (120, 180, 255) if active else (200, 200, 200)
    pygame.draw.rect(screen, bg, rect, border_radius=6)
    pygame.draw.rect(screen, border, rect, width=2, border_radius=6)

    display_text = text if text else ""
    suffix_surf = font.render(suffix, True, (210, 210, 210))
    max_text_w = max(24, rect.w - suffix_surf.get_width() - 24)
    if font.size(display_text)[0] > max_text_w:
        compact_text = display_text
        while compact_text and font.size(compact_text)[0] > max_text_w:
            compact_text = compact_text[:-1]
        display_text = compact_text if compact_text else ""

    surf = font.render(display_text, True, (255, 255, 255))
    screen.blit(surf, (rect.x + 8, rect.y + (rect.h - surf.get_height()) // 2))

    suffix_x = rect.right - suffix_surf.get_width() - 8
    suffix_y = rect.y + (rect.h - suffix_surf.get_height()) // 2
    screen.blit(suffix_surf, (suffix_x, suffix_y))

    if active:
        caret_x = min(rect.right - suffix_surf.get_width() - 14, rect.x + 10 + surf.get_width())
        caret_top = rect.y + 8
        caret_bottom = rect.bottom - 8
        pygame.draw.line(screen, (255, 255, 255), (caret_x, caret_top), (caret_x, caret_bottom), 2)


def draw_tooltip(screen, text, mouse_pos, font):
    padding_x = 8
    padding_y = 6
    offset_x = 14
    offset_y = 18
    surf = font.render(text, True, (245, 245, 245))
    rect = pygame.Rect(
        mouse_pos[0] + offset_x,
        mouse_pos[1] + offset_y,
        surf.get_width() + 2 * padding_x,
        surf.get_height() + 2 * padding_y,
    )
    rect.right = min(rect.right, screen.get_width() - 8)
    rect.bottom = min(rect.bottom, screen.get_height() - 8)
    pygame.draw.rect(screen, (35, 35, 35), rect, border_radius=6)
    pygame.draw.rect(screen, (220, 220, 220), rect, width=1, border_radius=6)
    screen.blit(surf, (rect.x + padding_x, rect.y + padding_y))


def convergence_setup_dialog(screen, config, settings, near_test=None):
    """Edit scan or near-run parameters; return None to cancel, False to quit."""
    fields = [
        ("x_cm", "Maximum |delta x| (cm)"), ("y_cm", "Maximum |delta y| (cm)"),
        ("angle_deg", "Maximum |delta a| (deg)"),
    ] if near_test else [
        ("runs", "Max trials per offset (N)"), ("step_x_cm", "Delta x step (cm)"),
        ("step_y_cm", "Delta y step (cm)"), ("step_angle_deg", "Delta a step (deg)"),
        ("max_x_cm", "Maximum |delta x| (cm)"), ("max_y_cm", "Maximum |delta y| (cm)"),
        ("max_angle_deg", "Maximum |delta a| (deg)"),
    ]
    values = [format_display_value(config[key]) for key, _ in fields]
    active, replace, error = 0, True, ""
    font = pygame.font.SysFont("consolas", 15)
    bold = pygame.font.SysFont("consolas", 17, bold=True)
    on_off = lambda enabled: "ON" if enabled else "OFF"
    inherited_lines = [
        f"Algorithm: {settings['algorithm_name']} | {settings['profile_name']} | "
        f"Path: {'Static' if near_test == 'S-NEAR' else settings['demo_path']['name']}",
        f"Landmarks: {settings['landmark_scenario_profile']}",
        f"FoV: {on_off(settings['viewing_angle_enabled'])} ({settings['viewing_angle_deg']:g} deg) | "
        f"Noise: {on_off(settings['noise_enabled'])} ({settings['noise_x_cm']:g} cm; motion {settings['motion_noise_ratio']:.0%})",
        f"Dropout: {on_off(settings['landmark_dropout_enabled'])} ({settings['landmark_dropout_probability']:.0%}) | "
        f"Obstruction: {on_off(settings['landmark_obstruction_enabled'])} | Slip: {on_off(settings['motion_slip_enabled'])}",
        f"Odom bias: {on_off(settings['odometry_bias_enabled'])}",
        f"Distance precision modifier: {on_off(settings['precision_distance_modifier_enabled'])}",
    ]
    if near_test != "S-NEAR":
        inherited_lines.append(f"Ball: x={settings['ball_fcs']['x_cm']:g}, y={settings['ball_fcs']['y_cm']:g} cm")
    panel = pygame.Rect(0, 0, min(650, screen.get_width() - 20), min(450 if near_test else 625, screen.get_height() - 20))
    panel.center = screen.get_rect().center
    rects = [pygame.Rect(panel.right - 175, panel.top + (60 if near_test else 84) + i * 35, 155, 28) for i in range(len(fields))]
    start_btn = pygame.Rect(panel.left + 20, panel.bottom - 48, 180, 30)
    cancel_btn = pygame.Rect(panel.right - 150, panel.bottom - 48, 130, 30)
    background = screen.copy()
    clock = pygame.time.Clock()
    while True:
        screen.blit(background, (0, 0))
        pygame.draw.rect(screen, (30, 35, 42), panel, border_radius=8)
        title = f"{near_test} CONVERGENCE" if near_test else "SCAN CONVERGENCE"
        screen.blit(bold.render(title, True, (245, 245, 245)), (panel.left + 18, panel.top + 16))
        if not near_test:
            screen.blit(font.render("Warning: scan may take 12 hours or longer.", True, (255, 195, 100)),
                        (panel.left + 18, panel.top + 42))
        for i, ((key, label), rect) in enumerate(zip(fields, rects)):
            screen.blit(font.render(label, True, (235, 235, 235)), (panel.left + 20, rect.y + 5))
            draw_input_box(screen, rect, values[i], font, active == i, "")
        inherited_top = rects[-1].bottom + 14
        screen.blit(bold.render("Test conditions", True, (235, 235, 235)), (panel.left + 20, inherited_top))
        for i, line in enumerate(inherited_lines):
            screen.blit(font.render(line, True, (200, 210, 220)), (panel.left + 20, inherited_top + 24 + i * 20))
        if error:
            for i, line in enumerate(textwrap.wrap(error, max(25, (panel.width - 36) // 9))):
                screen.blit(font.render(line, True, (255, 150, 130)), (panel.left + 18, panel.bottom - 95 + i * 17))
        draw_button(screen, start_btn, "START TEST" if near_test else "START SCAN", bold, (70, 100, 65), (255, 255, 255))
        draw_button(screen, cancel_btn, "CANCEL", bold, (95, 65, 65), (255, 255, 255))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            start = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if cancel_btn.collidepoint(event.pos):
                    return None
                start = start_btn.collidepoint(event.pos)
                for i, rect in enumerate(rects):
                    if rect.collidepoint(event.pos):
                        active, replace = i, True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_TAB:
                    active = (active + (-1 if event.mod & pygame.KMOD_SHIFT else 1)) % len(fields)
                    replace = True
                elif event.key == pygame.K_RETURN:
                    start = True
                elif event.key == pygame.K_BACKSPACE:
                    values[active] = "" if replace else values[active][:-1]
                    replace = False
            if event.type == pygame.TEXTINPUT:
                if all(char in "0123456789.-+eE" for char in event.text):
                    values[active] = event.text if replace else values[active] + event.text
                    replace = False
            if start:
                try:
                    result = dict(config)
                    result.update({key: float(value) for (key, _), value in zip(fields, values)})
                    if near_test:
                        if any(not math.isfinite(value) or value < 0 for value in result.values()):
                            raise ValueError("Near limits must be finite and non-negative")
                    else:
                        robustness_axes(result)
                        result["runs"], result["seed"] = int(result["runs"]), int(result["seed"])
                    return result
                except (ValueError, OverflowError) as exc:
                    error = str(exc)
        clock.tick(30)


def robustness_demo_adapter(demo_paths, selected_demo_idx, ball):
    """UI-owned playback adapter; path randomness is independent of sensor draws."""
    def initialize(actual):
        state = {"paths": deepcopy(demo_paths), "selected_idx": selected_demo_idx,
                 "look_target_x_cm": ball["x_cm"], "look_target_y_cm": ball["y_cm"]}
        saved_random_state = random.getstate()
        try:
            reset_demo_playback_state(actual, state)
            return {"robots": {"actual": actual}, "demo_state": state,
                    "path_random_state": random.getstate(), "settings": {"dt": DT}}
        finally:
            random.setstate(saved_random_state)

    def step(actual, path):
        step_paper_lm_demo(path)

    return initialize, step


# --- Main UI entrypoint ---
def main():
    global NOISE_ENABLED, ODOMETRY_BIAS_ENABLED, VIEWING_ANGLE_ENABLED, MOTION_SLIP_ENABLED, LANDMARK_DROPOUT_ENABLED, LANDMARK_OBSTRUCTION_ENABLED, PRECISION_DISTANCE_MODIFIER_ENABLED, ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME
    if getattr(pygame, "_is_stub", False) or getattr(plt, "_is_stub", False):
        raise ModuleNotFoundError("Interactive mode requires both pygame and matplotlib to be installed")
    ensure_simulation_core_configuration()
    pygame.init()
    pygame.key.start_text_input()
    info = pygame.display.Info()
    screen_w = info.current_w
    screen_h = info.current_h

    win_w = min(1380, int(screen_w * 0.95))
    win_h = min(880, int(screen_h * 0.90))

    panel_w = min(PANEL_W_PX, max(390, win_w // 4))
    world_w = win_w - panel_w
    world_h = win_h

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Active Inference Localization - Pygame")

    font = pygame.font.SysFont("consolas", 15)
    font_b = pygame.font.SysFont("consolas", 16, bold=True)
    font_ui = pygame.font.SysFont("consolas", 14, bold=True)
    font_info = pygame.font.SysFont("consolas", 13)
    clock = pygame.time.Clock()
    scale = compute_scale(world_w, world_h)

    robots = {
        "actual": make_robot_gc_state(0.0, 0.0, wrap_rad(0.0)),
        "belief": make_robot_gc_state(0.0, -200.0, wrap_rad(-math.pi / 2.0)),
    }
    ball = {
        "x_cm": 300.0,
        "y_cm": 0.0,
    }
    robot_colors = {
        "actual": (0, 255, 0),
        "belief": (255, 0, 0),
    }
    actual_pose = robot_pose_view(robots["actual"])

    selected = "actual"
    ROBOT_PICK_RADIUS_PX = 20
    BALL_PICK_RADIUS_PX = 18
    ANIMATION_INTERVAL = 20
    dragging = False
    drag_offset_cm = (0.0, 0.0)
    analysis_on = False
    algorithm_names = [
        LANDMARK_ONLY_ALGORITHM_NAME,
        MOTION_ONLY_ALGORITHM_NAME,
        FUSED_MOTION_LANDMARK_ALGORITHM_NAME,
    ]
    selected_algorithm_idx = algorithm_names.index(ACTIVE_ALGORITHM_NAME)
    selected_settings_profile_idx = 0
    apply_masterdata_setting_profile(
        masterdata_setting_profile_by_name(
            MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx],
            PRECISION_DISTANCE_MODIFIER_ENABLED,
        )
    )
    landmark_scenario_profile_names = [profile["name"] for profile in LANDMARK_SCENARIO_PROFILES]
    selected_landmark_scenario_profile_idx = landmark_scenario_profile_names.index(ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME)
    last_history = []
    converge_test_status = None
    paper_lm_setup = None
    paper_lm_result = None
    robustness_config = robustness_sweep_defaults()
    default_demo_path_idx = next(
        (idx for idx, path in enumerate(DEMO_PATHS) if path["name"] == "Circle"),
        0,
    )
    demo_state = {
        "paths": DEMO_PATHS,
        "selected_idx": default_demo_path_idx,
        "playing": False,
        "pending_reset": True,
        "time_s": 0.0,
        "sim_accumulator_s": 0.0,
        "anchor_x_cm": actual_pose["x_cm"],
        "anchor_y_cm": actual_pose["y_cm"],
        "rotation_rad": 0.0,
        "scale": 1.0,
        "trace": [(actual_pose["x_cm"], actual_pose["y_cm"])],
        "look_target_x_cm": ball["x_cm"],
        "look_target_y_cm": ball["y_cm"],
    }
    input_fields = {
        "landmark_noise": {
            "text": f"{NOISE_STD_X_CM:.2f}".rstrip("0").rstrip("."),
            "active": False,
            "replace_on_type": False,
            "suffix": "cm/deg",
            "label": None,
            "getter": lambda: NOISE_STD_X_CM,
            "setter": set_noise_cm,
        },
        "motion_noise": {
            "text": f"{MOTION_NOISE_RATIO * 100.0:.1f}".rstrip("0").rstrip("."),
            "active": False,
            "replace_on_type": False,
            "suffix": "%",
            "label": None,
            "getter": lambda: MOTION_NOISE_RATIO * 100.0,
            "setter": set_motion_noise_percent,
        },
        "viewing_angle": {
            "text": f"{VIEWING_ANGLE_DEG:.1f}".rstrip("0").rstrip("."),
            "active": False,
            "replace_on_type": False,
            "suffix": "deg",
            "label": "FOV",
            "getter": lambda: VIEWING_ANGLE_DEG,
            "setter": set_viewing_angle_deg,
        },
        "converge_runs": {
            "text": "reg" if CONVERGE_TEST_USE_REGRESSION_SET else str(CONVERGE_TEST_RUNS),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "CONV CASES",
            "getter": lambda: "reg" if CONVERGE_TEST_USE_REGRESSION_SET else CONVERGE_TEST_RUNS,
            "setter": set_converge_test_runs,
        },
        "dt": {
            "text": format_display_value(DT),
            "active": False,
            "replace_on_type": False,
            "suffix": "s",
            "label": "DT",
            "getter": lambda: DT,
            "setter": set_dt,
        },
        "precision": {
            "text": f"{PI_CAM:.4f}".rstrip("0").rstrip("."),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "\u03c0 CAM",
            "getter": lambda: PI_CAM,
            "setter": set_precision,
        },
        "precision_odo": {
            "text": f"{PI_ODO:.4f}".rstrip("0").rstrip("."),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "\u03c0 ODO",
            "getter": lambda: PI_ODO,
            "setter": set_precision_odo,
        },
        "precision_imu": {
            "text": f"{PI_IMU:.4f}".rstrip("0").rstrip("."),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "\u03c0 IMU",
            "getter": lambda: PI_IMU,
            "setter": set_precision_imu,
        },
        "lr_xy": {
            "text": f"{LR_XY:.4f}".rstrip("0").rstrip("."),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "LR X,Y",
            "getter": lambda: LR_XY,
            "setter": set_learning_rate_xy,
        },
        "lr_angle": {
            "text": format_display_value(LR_ANGLE),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "LR ANGLE",
            "getter": lambda: LR_ANGLE,
            "setter": set_learning_rate_angle,
        },
        "motion_lr_vel_xy": {
            "text": format_display_value(MOTION_LR_VEL_XY),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "LR VEL XY",
            "getter": lambda: MOTION_LR_VEL_XY,
            "setter": set_motion_lr_vel_xy,
        },
        "motion_lr_vel_theta": {
            "text": format_display_value(MOTION_LR_VEL_THETA),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "LR VEL A",
            "getter": lambda: MOTION_LR_VEL_THETA,
            "setter": set_motion_lr_vel_theta,
        },
        "motion_lr_acc_xy": {
            "text": format_display_value(MOTION_LR_ACC_XY),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "LR ACC XY",
            "getter": lambda: MOTION_LR_ACC_XY,
            "setter": set_motion_lr_acc_xy,
        },
        "motion_lr_acc_theta": {
            "text": format_display_value(MOTION_LR_ACC_THETA),
            "active": False,
            "replace_on_type": False,
            "suffix": "",
            "label": "LR ACC A",
            "getter": lambda: MOTION_LR_ACC_THETA,
            "setter": set_motion_lr_acc_theta,
        },
    }

    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    goal_half_w = GOAL_WIDTH_CM / 2
    legend_padding = 8
    legend_line_h = 18
    legend_w = 150
    legend_h = legend_padding * 2 + legend_line_h * 3

    panel_left = world_w + 16
    panel_inner_w = panel_w - 32
    btn_h = 26
    row_gap = 3
    section_gap = 8
    small_btn_gap = 6
    small_btn_w = (panel_inner_w - 2 * small_btn_gap) // 3
    row_y = 16

    quit_btn = pygame.Rect(panel_left, row_y, panel_inner_w, btn_h)
    row_y += btn_h + row_gap
    analysis_btn = pygame.Rect(panel_left, row_y, panel_inner_w, btn_h)
    row_y += btn_h + row_gap
    algorithm_profile_gap = 8
    algorithm_profile_w = (panel_inner_w - algorithm_profile_gap) // 2
    algo_btn = pygame.Rect(panel_left, row_y, algorithm_profile_w, btn_h)
    settings_profile_btn = pygame.Rect(algo_btn.right + algorithm_profile_gap, row_y,
                                       panel_inner_w - algorithm_profile_w - algorithm_profile_gap, btn_h)
    row_y += btn_h + row_gap
    landmark_scenario_profile_btn = pygame.Rect(panel_left, row_y, panel_inner_w, btn_h)
    row_y += btn_h + row_gap
    step_btn_gap = 6
    step_btn_w = (panel_inner_w - 2 * step_btn_gap) // 3
    exp1_btn = pygame.Rect(panel_left, row_y, step_btn_w, btn_h)
    exp100_btn = pygame.Rect(panel_left + step_btn_w + step_btn_gap, row_y, step_btn_w, btn_h)
    exp2000_btn = pygame.Rect(panel_left + 2 * (step_btn_w + step_btn_gap), row_y, step_btn_w, btn_h)
    row_y += btn_h + row_gap
    plot_btn = pygame.Rect(panel_left, row_y, panel_inner_w, btn_h)
    row_y += btn_h + section_gap
    paper_reset_gap = 6
    paper_reset_btn_w = (panel_inner_w - paper_reset_gap) // 2
    precision_distance_toggle_btn = pygame.Rect(panel_left, row_y, paper_reset_btn_w, btn_h)
    reset_belief_btn = pygame.Rect(precision_distance_toggle_btn.right + paper_reset_gap, row_y, panel_inner_w - paper_reset_btn_w - paper_reset_gap, btn_h)
    row_y += btn_h + section_gap
    demo_btn_gap = 8
    demo_btn_w = (panel_inner_w - demo_btn_gap) // 2
    demo_play_btn = pygame.Rect(panel_left, row_y, demo_btn_w, btn_h)
    demo_btn = pygame.Rect(panel_left + demo_btn_w + demo_btn_gap, row_y, panel_inner_w - demo_btn_w - demo_btn_gap, btn_h)
    row_y += btn_h + section_gap

    noise_toggle_w = 114
    noise_gap = 8
    noise_toggle_btn = pygame.Rect(panel_left, row_y, noise_toggle_w, btn_h)
    noise_field_gap = 6
    noise_fields_x = panel_left + noise_toggle_w + noise_gap
    noise_fields_w = panel_inner_w - noise_toggle_w - noise_gap
    landmark_noise_input_rect = pygame.Rect(noise_fields_x, row_y, (noise_fields_w - noise_field_gap) // 2, btn_h)
    motion_noise_input_rect = pygame.Rect(landmark_noise_input_rect.right + noise_field_gap, row_y, noise_fields_w - landmark_noise_input_rect.w - noise_field_gap, btn_h)
    row_y += btn_h + row_gap
    viewing_toggle_w = 114
    viewing_gap = 8
    viewing_toggle_btn = pygame.Rect(panel_left, row_y, viewing_toggle_w, btn_h)
    compact_btn_gap = 6
    compact_btn_w = (panel_inner_w - 3 * compact_btn_gap) // 4
    viewing_angle_input_rect = pygame.Rect(panel_left + viewing_toggle_w + viewing_gap, row_y, viewing_toggle_w, btn_h)
    robustness_btn = pygame.Rect(viewing_angle_input_rect.right + viewing_gap, row_y, viewing_toggle_w, btn_h)
    row_y += btn_h + row_gap
    slip_btn = pygame.Rect(panel_left, row_y, compact_btn_w, btn_h)
    dropout_btn = pygame.Rect(slip_btn.right + compact_btn_gap, row_y, compact_btn_w, btn_h)
    obstruction_btn = pygame.Rect(dropout_btn.right + compact_btn_gap, row_y, compact_btn_w, btn_h)
    odometry_bias_btn = pygame.Rect(obstruction_btn.right + compact_btn_gap, row_y, panel_left + panel_inner_w - obstruction_btn.right - compact_btn_gap, btn_h)
    row_y += btn_h + row_gap
    conv_btn_gap = compact_btn_gap
    conv_btn_w = compact_btn_w
    convtest_btn = pygame.Rect(panel_left, row_y, conv_btn_w, btn_h)
    convnear_btn = pygame.Rect(convtest_btn.right + conv_btn_gap, row_y, conv_btn_w, btn_h)
    convcorner_btn = pygame.Rect(convnear_btn.right + conv_btn_gap, row_y, conv_btn_w, btn_h)
    converge_runs_input_rect = pygame.Rect(convcorner_btn.right + conv_btn_gap, row_y, panel_left + panel_inner_w - convcorner_btn.right - conv_btn_gap, btn_h)
    row_y += btn_h + row_gap
    convdynamic_btn = pygame.Rect(panel_left, row_y, conv_btn_w, btn_h)
    convdynamic_near_btn = pygame.Rect(convdynamic_btn.right + conv_btn_gap, row_y, conv_btn_w, btn_h)
    paper_data_btn = pygame.Rect(convdynamic_near_btn.right + conv_btn_gap, row_y, conv_btn_w, btn_h)
    paper_lm_btn = pygame.Rect(paper_data_btn.right + conv_btn_gap, row_y, conv_btn_w, btn_h)
    row_y += btn_h + row_gap
    info_top = row_y + 8

    settings_w = 180
    settings_x = world_w - settings_w - 14
    settings_y = 14 + legend_h + 10
    settings_fields_y = settings_y
    dt_input_rect = pygame.Rect(settings_x, settings_fields_y, settings_w, btn_h)
    precision_input_rect = pygame.Rect(settings_x, settings_fields_y + 1 * (btn_h + row_gap), settings_w, btn_h)
    precision_odo_input_rect = pygame.Rect(settings_x, settings_fields_y + 2 * (btn_h + row_gap), settings_w, btn_h)
    precision_imu_input_rect = pygame.Rect(settings_x, settings_fields_y + 3 * (btn_h + row_gap), settings_w, btn_h)
    lr_xy_input_rect = pygame.Rect(settings_x, settings_fields_y + 4 * (btn_h + row_gap), settings_w, btn_h)
    lr_angle_input_rect = pygame.Rect(settings_x, settings_fields_y + 5 * (btn_h + row_gap), settings_w, btn_h)
    motion_lr_vel_xy_input_rect = pygame.Rect(settings_x, settings_fields_y + 6 * (btn_h + row_gap), settings_w, btn_h)
    motion_lr_vel_theta_input_rect = pygame.Rect(settings_x, settings_fields_y + 7 * (btn_h + row_gap), settings_w, btn_h)
    motion_lr_acc_xy_input_rect = pygame.Rect(settings_x, settings_fields_y + 8 * (btn_h + row_gap), settings_w, btn_h)
    motion_lr_acc_theta_input_rect = pygame.Rect(settings_x, settings_fields_y + 9 * (btn_h + row_gap), settings_w, btn_h)

    field_rects = {
        "landmark_noise": landmark_noise_input_rect,
        "motion_noise": motion_noise_input_rect,
        "viewing_angle": viewing_angle_input_rect,
        "converge_runs": converge_runs_input_rect,
        "dt": dt_input_rect,
        "precision": precision_input_rect,
        "precision_odo": precision_odo_input_rect,
        "precision_imu": precision_imu_input_rect,
        "lr_xy": lr_xy_input_rect,
        "lr_angle": lr_angle_input_rect,
        "motion_lr_vel_xy": motion_lr_vel_xy_input_rect,
        "motion_lr_vel_theta": motion_lr_vel_theta_input_rect,
        "motion_lr_acc_xy": motion_lr_acc_xy_input_rect,
        "motion_lr_acc_theta": motion_lr_acc_theta_input_rect,
    }
    tooltip_delay_ms = 600
    tooltip_state = {"key": None, "since_ms": 0}
    tooltip_items = [
        (
            "quit",
            quit_btn,
            "Close the interactive simulator window",
        ),
        (
            "analysis",
            analysis_btn,
            "Show one-step prediction errors and update terms in the side panel",
        ),
        (
            "algorithm",
            algo_btn,
            "Cycle which observation sources the inference step uses: landmarks, motion, or both",
        ),
        (
            "step_1",
            exp1_btn,
            "Run 1 inference step with the selected algorithm",
        ),
        (
            "step_100",
            exp100_btn,
            "Run 100 inference steps with the selected algorithm",
        ),
        (
            "step_2000",
            exp2000_btn,
            "Run 2000 inference steps with the selected algorithm",
        ),
        (
            "paper_visibility_dataset",
            paper_data_btn,
            "Run 2 hours with a near-start belief and export visibility-stress CSV/JSON",
        ),
        (
            "paper_lm_setup",
            paper_lm_btn,
            "Run N trials of 30 simulated seconds per LM profile (1, 2, 3, 5, 9) with identical starts, ball and path. Save all data to CSV/JSON; report and paper graph show only LM 3, 5, 9",
        ),
        (
            "precision_distance_modifier",
            precision_distance_toggle_btn,
            lambda: f"Distance-based landmark precision modifier is {'ON' if PRECISION_DISTANCE_MODIFIER_ENABLED else 'OFF'}",
        ),
        (
            "reset_belief",
            reset_belief_btn,
            "Copy the green actual robot state into the red belief robot state",
        ),
        (
            "demo_play",
            demo_play_btn,
            "Start or stop playback of the selected actual-robot demo path",
        ),
        (
            "demo_path",
            demo_btn,
            lambda: f"Cycle demo path; current={demo_state['paths'][demo_state['selected_idx']]['name']}",
        ),
        (
            "settings_profile",
            settings_profile_btn,
            lambda: f"Masterdata profile: {MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx]}. Click to apply next profile",
        ),
        (
            "landmark_scenario_profile",
            landmark_scenario_profile_btn,
            lambda: f"Landmark scenario profile: {ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME} ({len(active_simulation_landmarks())} active simulated landmarks)",
        ),
        (
            "noise_toggle",
            noise_toggle_btn,
            lambda: f"Noise {'ON' if NOISE_ENABLED else 'OFF'} for landmark and motion observations",
        ),
        (
            "landmark_noise",
            landmark_noise_input_rect,
            lambda: (
                f"Landmark noise sigma: x={NOISE_STD_X_CM:.1f} cm, "
                f"y={NOISE_STD_Y_CM:.1f} cm, angle={NOISE_STD_ANGLE_DEG:.1f} deg"
            ),
        ),
        (
            "motion_noise",
            motion_noise_input_rect,
            lambda: f"Motion noise sigma: {MOTION_NOISE_RATIO * 100.0:.1f}% of each velocity/acceleration reading",
        ),
        (
            "fov",
            viewing_toggle_btn,
            "FoV limits simulated landmark observations to landmarks inside the actual robot camera cone",
        ),
        (
            "odometry_bias",
            odometry_bias_btn,
            "Bias adds fixed odometry drift: +18 cm/s forward, +8 cm/s left, +1.5 deg/s rotation",
        ),
        (
            "slip",
            slip_btn,
            "Slip simulates short odometry slip events by scaling motion velocity readings",
        ),
        (
            "dropout",
            dropout_btn,
            "Dropout simulates missed landmark detections by randomly removing 10% of observations",
        ),
        (
            "obstruction",
            obstruction_btn,
            "Obstruction simulates blocked landmarks for about 2.0 +/- 0.5 seconds",
        ),
        (
            "plot_last_experiment",
            plot_btn,
            "Create a PDF report from the most recent experiment history",
        ),
        (
            "converge_runs",
            converge_runs_input_rect,
            "Enter number of random convergence cases, or 'reg' to use the fixed regression start set",
        ),
        (
            "converge_random",
            convtest_btn,
            "Landmark-only static convergence: actual and belief independently random near field",
        ),
        (
            "converge_near",
            convnear_btn,
            lambda: f"Landmark-only static convergence: actual inside field; {near_start_condition_text()}",
        ),
        (
            "converge_corner",
            convcorner_btn,
            "Landmark-only static convergence: actual outside corner; belief random near field",
        ),
        (
            "converge_dynamic",
            convdynamic_btn,
            "Run convergence cases while the actual robot follows the demo path from random starts",
        ),
        (
            "converge_dynamic_near",
            convdynamic_near_btn,
            lambda: f"Run convergence cases while the actual robot follows the demo path; {near_start_condition_text()}",
        ),
        (
            "robustness",
            robustness_btn,
            "Sweep exact FCS dx/dy/angle offsets; N random-start trials per point; only 100% correct passes. ESC stops and saves partial results",
        ),
    ]

    def format_input_value(value, decimals=4):
        if isinstance(value, str):
            return value
        return format_display_value(value, decimals)

    def reset_input_field(name):
        field = input_fields[name]
        if name == "converge_runs":
            decimals = 0
        elif name in ("landmark_noise", "motion_noise"):
            decimals = 2
        elif name == "viewing_angle":
            decimals = 1
        else:
            decimals = 4
        field["text"] = format_input_value(field["getter"](), decimals)

    def deactivate_all_inputs(reset=False):
        for name, field in input_fields.items():
            field["active"] = False
            field["replace_on_type"] = False
            if reset:
                reset_input_field(name)

    def active_input_name():
        for name, field in input_fields.items():
            if field["active"]:
                return name
        return None

    def apply_input_field(name):
        field = input_fields[name]
        text = field["text"].strip()
        if not text:
            reset_input_field(name)
            return
        if name == "converge_runs" and text.lower() in ("reg", "regression"):
            set_converge_test_regression_set()
            reset_input_field(name)
            return
        try:
            value = max(0.0, float(text))
        except ValueError:
            reset_input_field(name)
            return
        field["setter"](value)
        reset_input_field(name)

    def run_demo_step_experiment(algorithm_name, max_steps, callback_interval=20):
        history = []
        settings = experiment_settings(algorithm_name)
        settings["profile_name"] = MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx]
        settings["demo_path_name"] = demo_state["paths"][demo_state["selected_idx"]]["name"]
        advance_demo = True

        if advance_demo and demo_state["pending_reset"]:
            reset_demo_playback_state(robots["actual"], demo_state)
        demo_state["playing"] = True
        try:
            for k in range(1, max_steps + 1):
                if advance_demo and DT > 0.0:
                    step_demo_playback(robots["actual"], demo_state, DT)
                y_obs_map, yhat_map, y_motion_obs, free_energy = simulation_inference_step(robots, algorithm_name)
                history.append(snapshot_record(k, robots, y_obs_map, yhat_map, y_motion_obs, free_energy, settings))

                if k % callback_interval == 0 or k == 1 or k == max_steps:
                    safe_draw()
                    plt.pause(0.001)
        finally:
            demo_state["playing"] = False

        return history

    def draw():
        analysis_data = None
        if analysis_on:
            analysis_data = one_step_analysis(robots, algorithm_names[selected_algorithm_idx])

        screen.fill((25, 120, 25))
        pygame.draw.rect(screen, (10, 80, 10), pygame.Rect(0, 0, world_w, world_h), width=2)

        opp_goal = rect_from_corners_fcs(halfL, -goal_half_w, halfL + GOAL_DEPTH_CM, goal_half_w, world_w, world_h, scale)
        own_goal = rect_from_corners_fcs(-halfL - GOAL_DEPTH_CM, -goal_half_w, -halfL, goal_half_w, world_w, world_h, scale)
        pygame.draw.rect(screen, (240, 240, 240), opp_goal, width=FIELD_MARKING_WIDTH_PX)
        pygame.draw.rect(screen, (240, 240, 240), own_goal, width=FIELD_MARKING_WIDTH_PX)

        field_rect = rect_from_corners_fcs(-halfL, -halfW, halfL, halfW, world_w, world_h, scale)
        pygame.draw.rect(screen, (255, 255, 255), field_rect, width=FIELD_MARKING_WIDTH_PX)

        goal_area_half_w = GOAL_AREA_WIDTH_CM / 2
        penalty_area_half_w = PENALTY_AREA_WIDTH_CM / 2

        opp_goal_area = rect_from_corners_fcs(
            halfL - GOAL_AREA_DEPTH_CM,
            -goal_area_half_w,
            halfL,
            goal_area_half_w,
            world_w,
            world_h,
            scale,
        )
        own_goal_area = rect_from_corners_fcs(
            -halfL,
            -goal_area_half_w,
            -halfL + GOAL_AREA_DEPTH_CM,
            goal_area_half_w,
            world_w,
            world_h,
            scale,
        )
        opp_penalty_area = rect_from_corners_fcs(
            halfL - PENALTY_AREA_DEPTH_CM,
            -penalty_area_half_w,
            halfL,
            penalty_area_half_w,
            world_w,
            world_h,
            scale,
        )
        own_penalty_area = rect_from_corners_fcs(
            -halfL,
            -penalty_area_half_w,
            -halfL + PENALTY_AREA_DEPTH_CM,
            penalty_area_half_w,
            world_w,
            world_h,
            scale,
        )
        pygame.draw.rect(screen, (255, 255, 255), opp_goal_area, width=FIELD_MARKING_WIDTH_PX)
        pygame.draw.rect(screen, (255, 255, 255), own_goal_area, width=FIELD_MARKING_WIDTH_PX)
        pygame.draw.rect(screen, (255, 255, 255), opp_penalty_area, width=FIELD_MARKING_WIDTH_PX)
        pygame.draw.rect(screen, (255, 255, 255), own_penalty_area, width=FIELD_MARKING_WIDTH_PX)

        p1 = fcs_to_screen(0, -halfW, world_w, world_h, scale)
        p2 = fcs_to_screen(0, halfW, world_w, world_h, scale)
        pygame.draw.line(screen, (255, 255, 255), p1, p2, FIELD_MARKING_WIDTH_PX)

        center_px = fcs_to_screen(0, 0, world_w, world_h, scale)
        center_circle_radius_px = max(1, int(CENTER_CIRCLE_RADIUS_CM * scale))
        pygame.draw.circle(screen, (255, 255, 255), center_px, center_circle_radius_px, width=FIELD_MARKING_WIDTH_PX)

        legend_rect = pygame.Rect(world_w - legend_w - 14, 14, legend_w, legend_h)
        legend_surface = pygame.Surface((legend_rect.w, legend_rect.h), pygame.SRCALPHA)
        legend_surface.fill((10, 10, 10, 135))
        screen.blit(legend_surface, legend_rect.topleft)
        pygame.draw.rect(screen, (235, 235, 235), legend_rect, width=1, border_radius=8)
        screen.blit(font_b.render("Green = actual", True, (0, 255, 0)), (legend_rect.x + legend_padding, legend_rect.y + legend_padding))
        screen.blit(font_b.render("Red = belief", True, (255, 80, 80)), (legend_rect.x + legend_padding, legend_rect.y + legend_padding + legend_line_h))
        screen.blit(font_b.render("Q/E to rotate", True, (235, 235, 235)), (legend_rect.x + legend_padding, legend_rect.y + legend_padding + legend_line_h * 2))

        for landmark_id, (lx, ly) in active_simulation_landmarks().items():
            px, py = fcs_to_screen(lx, ly, world_w, world_h, scale)
            landmark_visible = True
            if VIEWING_ANGLE_ENABLED:
                landmark_visible = landmark_is_within_viewing_angle(
                    robots["actual"],
                    lx,
                    ly,
                    VIEWING_HALF_ANGLE_RAD,
                )
            landmark_color = (210, 185, 245) if landmark_visible else (155, 135, 190)
            landmark_radius = 5 if landmark_visible else 3
            pygame.draw.circle(screen, landmark_color, (px, py), landmark_radius)
            if LANDMARK_OBSTRUCTION_ENABLED and LANDMARK_OBSTRUCTION_REMAINING_S.get(landmark_id, 0.0) > 0.0:
                marker_radius = landmark_radius + 8
                pygame.draw.circle(screen, (0, 0, 0), (px, py), marker_radius, width=4)
                pygame.draw.line(screen, (0, 0, 0), (px - marker_radius, py - marker_radius), (px + marker_radius, py + marker_radius), 3)
                pygame.draw.line(screen, (0, 0, 0), (px - marker_radius, py + marker_radius), (px + marker_radius, py - marker_radius), 3)

        ball_px, ball_py = fcs_to_screen(ball["x_cm"], ball["y_cm"], world_w, world_h, scale)
        pygame.draw.circle(screen, (255, 220, 40), (ball_px, ball_py), BALL_RADIUS_PX)
        pygame.draw.circle(screen, (180, 140, 0), (ball_px, ball_py), BALL_RADIUS_PX, width=1)
        if selected == "ball":
            pygame.draw.circle(screen, (255, 255, 255), (ball_px, ball_py), BALL_RADIUS_PX + 5, width=2)

        if VIEWING_ANGLE_ENABLED:
            actual_pose = robot_pose_view(robots["actual"])
            robot_px, robot_py = fcs_to_screen(actual_pose["x_cm"], actual_pose["y_cm"], world_w, world_h, scale)
            cone_len_cm = max(FIELD_LENGTH_CM, FIELD_WIDTH_CM) * 0.7
            left_heading = actual_pose["theta_rad"] + VIEWING_HALF_ANGLE_RAD
            right_heading = actual_pose["theta_rad"] - VIEWING_HALF_ANGLE_RAD
            left_x_cm = actual_pose["x_cm"] + cone_len_cm * math.cos(left_heading)
            left_y_cm = actual_pose["y_cm"] + cone_len_cm * math.sin(left_heading)
            right_x_cm = actual_pose["x_cm"] + cone_len_cm * math.cos(right_heading)
            right_y_cm = actual_pose["y_cm"] + cone_len_cm * math.sin(right_heading)
            left_px = fcs_to_screen(left_x_cm, left_y_cm, world_w, world_h, scale)
            right_px = fcs_to_screen(right_x_cm, right_y_cm, world_w, world_h, scale)
            pygame.draw.line(screen, (120, 220, 255), (robot_px, robot_py), left_px, 2)
            pygame.draw.line(screen, (120, 220, 255), (robot_px, robot_py), right_px, 2)

        if demo_state["playing"] and len(demo_state["trace"]) >= 2:
            trace_points = [fcs_to_screen(x_cm, y_cm, world_w, world_h, scale) for x_cm, y_cm in demo_state["trace"]]
            pygame.draw.lines(screen, (80, 255, 160), False, trace_points, 2)

        for name, r in robots.items():
            pose = robot_pose_view(r)
            px, py = fcs_to_screen(pose["x_cm"], pose["y_cm"], world_w, world_h, scale)
            color = robot_colors[name]
            pygame.draw.circle(screen, color, (px, py), ROBOT_RADIUS_PX)
            hx = px - int(HEADING_LEN_PX * math.sin(pose["theta_rad"]))
            hy = py - int(HEADING_LEN_PX * math.cos(pose["theta_rad"]))
            pygame.draw.line(screen, color, (px, py), (hx, hy), 3)
            if name == "actual" and MOTION_SLIP_ENABLED and motion_slip_is_active():
                marker_color = (255, 185, 40)
                marker_radius = ROBOT_RADIUS_PX + 11
                pygame.draw.circle(screen, marker_color, (px, py), marker_radius, width=4)
                badge_rect = pygame.Rect(px + marker_radius - 6, py - marker_radius - 6, 18, 18)
                pygame.draw.rect(screen, marker_color, badge_rect, border_radius=5)
                badge_surf = font_b.render("S", True, (20, 20, 20))
                screen.blit(
                    badge_surf,
                    (
                        badge_rect.x + (badge_rect.w - badge_surf.get_width()) // 2,
                        badge_rect.y + (badge_rect.h - badge_surf.get_height()) // 2,
                    ),
                )
            if name == selected:
                pygame.draw.circle(screen, (255, 255, 255), (px, py), ROBOT_RADIUS_PX + 5, width=2)

        if analysis_data is not None and analysis_data.get("mode") in {"landmark", "fused"}:
            belief_pose = analysis_data["belief_pose"]
            belief_origin = fcs_to_screen(belief_pose["x_cm"], belief_pose["y_cm"], world_w, world_h, scale)
            max_contrib = max((d["contrib_mag"] for d in analysis_data["details"]), default=1.0)
            used_landmark_ids = {
                d["associated_landmark_id"]
                for d in analysis_data["details"]
                if d.get("associated_landmark_id") is not None
            }

            for d in analysis_data["details"]:
                obs_px = fcs_to_screen(d["obs_fcs"][0], d["obs_fcs"][1], world_w, world_h, scale)
                pred_px = fcs_to_screen(d["pred_fcs"][0], d["pred_fcs"][1], world_w, world_h, scale)
                strength = d["contrib_mag"] / max_contrib if max_contrib > 0 else 0.0
                contrib_width = 1 + int(4 * strength)
                error_color = (255, 70, 70)
                if d.get("association_mismatch", False):
                    error_color = (255, 60, 220)

                pygame.draw.line(screen, (0, 220, 220), belief_origin, obs_px, 1)
                pygame.draw.line(screen, (255, 170, 0), belief_origin, pred_px, 1)
                pygame.draw.line(screen, error_color, pred_px, obs_px, contrib_width)
                pygame.draw.circle(screen, (0, 220, 220), obs_px, 3)
                pred_radius = 5 if d.get("association_mismatch", False) else 3
                pygame.draw.circle(screen, (255, 170, 0), pred_px, pred_radius, width=1)
                if d.get("association_mismatch", False):
                    association_label = str(d.get("actual_landmark_id"))
                    association_surf = font_info.render(association_label, True, error_color)
                    screen.blit(association_surf, (pred_px[0] + 7, pred_px[1] - 8))
                dx_px = pred_px[0] - belief_origin[0]
                dy_px = pred_px[1] - belief_origin[1]
                base_angle = math.atan2(dy_px, dx_px)
                ea_abs = abs(d["ea"])
                if ea_abs > 0.01:
                    direction = 1.0 if d["ea"] >= 0.0 else -1.0
                    # Keep the radius fixed so the visible arc length directly reflects the angle error.
                    arc_span = max(0.08, min(2.8, ea_abs))
                    arc_radius = 18
                    arc_rect = pygame.Rect(
                        pred_px[0] - arc_radius,
                        pred_px[1] - arc_radius,
                        2 * arc_radius,
                        2 * arc_radius,
                    )
                    start_angle = base_angle - direction * 0.5 * arc_span
                    end_angle = base_angle + direction * 0.5 * arc_span
                    draw_start = min(start_angle, end_angle)
                    draw_end = max(start_angle, end_angle)
                    pygame.draw.arc(screen, error_color, arc_rect, draw_start, draw_end, 3)

            for landmark_id, yhat in analysis_data["y_hat"].items():
                if landmark_id in used_landmark_ids:
                    continue
                label_px = fcs_to_screen(*rcs_to_fcs(robots["belief"], yhat), world_w, world_h, scale)
                unused_surf = font_info.render("not used", True, (255, 60, 220))
                screen.blit(unused_surf, (label_px[0] + 7, label_px[1] - 8))

        draw_input_box(screen, dt_input_rect, f"DT {input_fields['dt']['text']}", font_ui, input_fields["dt"]["active"], input_fields["dt"]["suffix"])
        draw_input_box(screen, precision_input_rect, f"\u03c0 cam {input_fields['precision']['text']}", font_ui, input_fields["precision"]["active"], input_fields["precision"]["suffix"])
        draw_input_box(screen, precision_odo_input_rect, f"\u03c0 odo {input_fields['precision_odo']['text']}", font_ui, input_fields["precision_odo"]["active"], input_fields["precision_odo"]["suffix"])
        draw_input_box(screen, precision_imu_input_rect, f"\u03c0 imu {input_fields['precision_imu']['text']}", font_ui, input_fields["precision_imu"]["active"], input_fields["precision_imu"]["suffix"])
        draw_input_box(screen, lr_xy_input_rect, f"LR cam xy {input_fields['lr_xy']['text']}", font_ui, input_fields["lr_xy"]["active"], input_fields["lr_xy"]["suffix"])
        draw_input_box(screen, lr_angle_input_rect, f"LR cam a {input_fields['lr_angle']['text']}", font_ui, input_fields["lr_angle"]["active"], input_fields["lr_angle"]["suffix"])
        draw_input_box(screen, motion_lr_vel_xy_input_rect, f"LR vel xy {input_fields['motion_lr_vel_xy']['text']}", font_ui, input_fields["motion_lr_vel_xy"]["active"], input_fields["motion_lr_vel_xy"]["suffix"])
        draw_input_box(screen, motion_lr_vel_theta_input_rect, f"LR vel a {input_fields['motion_lr_vel_theta']['text']}", font_ui, input_fields["motion_lr_vel_theta"]["active"], input_fields["motion_lr_vel_theta"]["suffix"])
        draw_input_box(screen, motion_lr_acc_xy_input_rect, f"LR acc xy {input_fields['motion_lr_acc_xy']['text']}", font_ui, input_fields["motion_lr_acc_xy"]["active"], input_fields["motion_lr_acc_xy"]["suffix"])
        draw_input_box(screen, motion_lr_acc_theta_input_rect, f"LR acc a {input_fields['motion_lr_acc_theta']['text']}", font_ui, input_fields["motion_lr_acc_theta"]["active"], input_fields["motion_lr_acc_theta"]["suffix"])

        pygame.draw.rect(screen, (15, 15, 15), pygame.Rect(world_w, 0, panel_w, win_h))
        draw_button(screen, quit_btn, "QUIT", font_b, (150, 40, 40), (255, 255, 255))
        draw_button(screen, analysis_btn, "1 STEP ANALYSIS", font_b, (40, 140, 40) if analysis_on else (70, 70, 70), (255, 255, 255))
        draw_button(screen, algo_btn, algorithm_names[selected_algorithm_idx], font_b, (60, 90, 140), (255, 255, 255))
        draw_button(
            screen,
            settings_profile_btn,
            fit_button_label(MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx], font_b, settings_profile_btn.w - 16),
            font_b,
            (85, 75, 135),
            (255, 255, 255),
        )
        draw_button(
            screen,
            landmark_scenario_profile_btn,
            fit_button_label(ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME, font_b, landmark_scenario_profile_btn.w - 16),
            font_b,
            (60, 90, 140),
            (255, 255, 255),
        )
        draw_button(screen, exp1_btn, "STEP 1", font_b, (70, 70, 120), (255, 255, 255))
        draw_button(screen, exp100_btn, "STEP 100", font_b, (70, 70, 120), (255, 255, 255))
        draw_button(screen, exp2000_btn, "STEP 2000", font_b, (70, 70, 120), (255, 255, 255))
        draw_button(
            screen,
            demo_btn,
            fit_button_label(
                demo_state["paths"][demo_state["selected_idx"]]["name"],
                font_b,
                demo_btn.w - 16,
            ),
            font_b,
            (60, 120, 95),
            (255, 255, 255),
        )
        draw_button(
            screen,
            demo_play_btn,
            "END DEMO" if demo_state["playing"] else "PLAY DEMO",
            font_b,
            (40, 140, 40) if demo_state["playing"] else (70, 70, 70),
            (255, 255, 255),
        )
        draw_toggle_button(screen, noise_toggle_btn, "NOISE", NOISE_ENABLED, font_b)
        draw_input_box(screen, landmark_noise_input_rect, input_fields["landmark_noise"]["text"], font_ui, input_fields["landmark_noise"]["active"], input_fields["landmark_noise"]["suffix"])
        draw_input_box(screen, motion_noise_input_rect, input_fields["motion_noise"]["text"], font_ui, input_fields["motion_noise"]["active"], input_fields["motion_noise"]["suffix"])
        draw_toggle_button(screen, viewing_toggle_btn, "FOV", VIEWING_ANGLE_ENABLED, font_b)
        draw_input_box(screen, viewing_angle_input_rect, input_fields["viewing_angle"]["text"], font_ui, input_fields["viewing_angle"]["active"], input_fields["viewing_angle"]["suffix"])
        draw_toggle_button(screen, slip_btn, "SLIP", MOTION_SLIP_ENABLED, font_b)
        draw_toggle_button(screen, dropout_btn, "DROP", LANDMARK_DROPOUT_ENABLED, font_b)
        draw_toggle_button(screen, obstruction_btn, "OBSTR", LANDMARK_OBSTRUCTION_ENABLED, font_b)
        draw_toggle_button(screen, odometry_bias_btn, "BIAS", ODOMETRY_BIAS_ENABLED, font_b)
        draw_button(screen, plot_btn, "PLOT LAST EXPERIMENT", font_b, (110, 80, 30), (255, 255, 255))
        draw_toggle_button(screen, precision_distance_toggle_btn, "DIST PREC", PRECISION_DISTANCE_MODIFIER_ENABLED, font_b)
        draw_button(screen, reset_belief_btn, "RESET", font_b, (120, 65, 45), (255, 255, 255))
        draw_input_box(screen, converge_runs_input_rect, input_fields["converge_runs"]["text"], font_ui, input_fields["converge_runs"]["active"], input_fields["converge_runs"]["suffix"])
        draw_button(screen, convtest_btn, "S-CONV", font_b, (90, 60, 120), (255, 255, 255))
        draw_button(screen, convnear_btn, "S-NEAR", font_b, (90, 60, 120), (255, 255, 255))
        draw_button(screen, convcorner_btn, "S-CORN", font_b, (90, 60, 120), (255, 255, 255))
        draw_button(screen, convdynamic_btn, "D-CONV", font_b, (90, 60, 120), (255, 255, 255))
        draw_button(screen, convdynamic_near_btn, "D-NEAR", font_b, (90, 60, 120), (255, 255, 255))
        draw_button(screen, paper_data_btn, "P-2hrs", font_b, (105, 45, 150), (255, 255, 255))
        draw_button(screen, paper_lm_btn, "P-LM", font_b, (105, 45, 150), (255, 255, 255))
        draw_button(screen, robustness_btn, "C-scan", font_b, (90, 60, 120), (255, 255, 255))

        lines = []
        if converge_test_status is not None:
            lines.extend(converge_test_status.splitlines())
        actual_pose = robot_pose_view(robots["actual"])
        belief_pose = robot_pose_view(robots["belief"])
        actual_fcs_gc = robots["actual"]
        belief_fcs_gc = robots["belief"]
        lines.extend([
            "ACTUAL",
            " axis      pos       vel       acc",
            f" x     {actual_pose['x_cm']:>7.0f}  {actual_fcs_gc['x'][1]:>8.1f}  {actual_fcs_gc['x'][2]:>8.1f}",
            f" y     {actual_pose['y_cm']:>7.0f}  {actual_fcs_gc['y'][1]:>8.1f}  {actual_fcs_gc['y'][2]:>8.1f}",
            f" a     {math.degrees(actual_pose['theta_rad']):>6.0f}d {actual_fcs_gc['theta'][1]:>8.2f}  {actual_fcs_gc['theta'][2]:>8.2f}",
            "BELIEF",
            " axis      pos       vel       acc",
            f" x     {belief_pose['x_cm']:>7.0f}  {belief_fcs_gc['x'][1]:>8.1f}  {belief_fcs_gc['x'][2]:>8.1f}",
            f" y     {belief_pose['y_cm']:>7.0f}  {belief_fcs_gc['y'][1]:>8.1f}  {belief_fcs_gc['y'][2]:>8.1f}",
            f" a     {math.degrees(belief_pose['theta_rad']):>6.0f}d {belief_fcs_gc['theta'][1]:>8.2f}  {belief_fcs_gc['theta'][2]:>8.2f}",
        ])
        if analysis_data is not None:
            lines.append("1 STEP ANALYSIS")
            if analysis_data.get("mode") == "motion":
                lines.append(" motion prediction errors")
                lines.append(" axis   v_obs  v_hat   e_v   a_obs  a_hat   e_a")
                for d in analysis_data["details"]:
                    lines.append(
                        f" {d['id'][:5]:<5} "
                        f"{d['obs_vel']:>6.1f} {d['pred_vel']:>6.1f} {d['vel_error']:>6.1f} "
                        f"{d['obs_acc']:>6.1f} {d['pred_acc']:>6.1f} {d['acc_error']:>6.1f}"
                    )
            else:
                lines.append(" cyan=obs orange=pred red=error")
                lines.append(
                    f" visible={len(analysis_data.get('observed_landmark_ids', []))}  "
                    f"hidden={len(analysis_data.get('hidden_landmark_ids', []))}  "
                    f"assoc_wrong={analysis_data.get('association_mismatch_count', 0)}"
                )
                lines.append(" id          e(x,y,a)    jtE(x,y,a)   assoc")
                for d in analysis_data["details"][:MAX_SENSE_LINES]:
                    assoc_text = ""
                    if d.get("association_mismatch", False):
                        assoc_text = f" ! {d.get('actual_landmark_id')}"
                    lines.append(
                        f" {str(d['id'])[:10]:<10} "
                        f"{int(round(d['ex'])):>3},{int(round(d['ey'])):>3},{math.degrees(d['ea']):>4.0f}   "
                        f"{d['contrib'][0]:>3.0f},{d['contrib'][1]:>3.0f},{d['contrib'][2]:>3.0f}"
                        f"{assoc_text}"
                    )
                lines.append(
                    f" sum = {analysis_data['total_contrib'][0]:.0f},"
                    f"{analysis_data['total_contrib'][1]:.0f},"
                    f"{analysis_data['total_contrib'][2]:.0f}"
                )
                if analysis_data.get("mode") == "fused":
                    lines.append(" motion prediction errors")
                    lines.append(" axis   v_obs  v_hat   e_v   a_obs  a_hat   e_a")
                    for d in analysis_data["motion_details"]:
                        lines.append(
                            f" {d['id'][:5]:<5} "
                            f"{d['obs_vel']:>6.1f} {d['pred_vel']:>6.1f} {d['vel_error']:>6.1f} "
                            f"{d['obs_acc']:>6.1f} {d['pred_acc']:>6.1f} {d['acc_error']:>6.1f}"
                        )
        y = info_top
        line_h = 16
        max_lines_fit = (win_h - y - 10) // line_h
        for s in lines[:max_lines_fit]:
            surf = font_info.render(s, True, (230, 230, 230))
            screen.blit(surf, (panel_left, y))
            y += line_h

        mouse_pos = pygame.mouse.get_pos()
        hovered_tooltip = None
        for key, rect, tooltip in tooltip_items:
            if rect.collidepoint(mouse_pos):
                hovered_tooltip = (key, tooltip)
                break
        now_ms = pygame.time.get_ticks()
        if hovered_tooltip is None:
            tooltip_state["key"] = None
            tooltip_state["since_ms"] = 0
        else:
            key, tooltip = hovered_tooltip
            if tooltip_state["key"] != key:
                tooltip_state["key"] = key
                tooltip_state["since_ms"] = now_ms
            elif now_ms - tooltip_state["since_ms"] >= tooltip_delay_ms:
                text = tooltip() if callable(tooltip) else tooltip
                draw_tooltip(screen, text, mouse_pos, font_info)

        pygame.display.flip()

    def safe_draw():
        try:
            draw()
        except Exception as exc:
            screen.fill((20, 20, 20))
            y = 24
            for s in ["UI draw error", exc.__class__.__name__, str(exc)]:
                surf = font_b.render(s, True, (255, 180, 180))
                screen.blit(surf, (24, y))
                y += 26
            pygame.display.flip()

    running = True
    while running:
        frame_dt_s = clock.tick(REALTIME_UI_FPS) / 1000.0
        frame_dt_s = min(frame_dt_s, 0.25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                current_input = active_input_name()
                if event.key == pygame.K_ESCAPE:
                    if current_input is not None:
                        deactivate_all_inputs(reset=True)
                    else:
                        running = False
                elif current_input is not None:
                    field = input_fields[current_input]
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        apply_input_field(current_input)
                        deactivate_all_inputs()
                    elif event.key == pygame.K_BACKSPACE:
                        if field["replace_on_type"]:
                            field["text"] = ""
                            field["replace_on_type"] = False
                        else:
                            field["text"] = field["text"][:-1]
                    elif event.key == pygame.K_TAB:
                        input_order = list(field_rects.keys())
                        current_idx = input_order.index(current_input)
                        apply_input_field(current_input)
                        deactivate_all_inputs()
                        next_name = input_order[(current_idx + 1) % len(input_order)]
                        input_fields[next_name]["active"] = True
                        input_fields[next_name]["replace_on_type"] = True
                    else:
                        allowed_chars = "0123456789.-"
                        if current_input == "converge_runs":
                            allowed_chars += "regREG"
                        if event.unicode and event.unicode in allowed_chars:
                            if field["replace_on_type"]:
                                field["text"] = ""
                                field["replace_on_type"] = False
                            if current_input == "converge_runs" and event.unicode.isalpha():
                                field["text"] += event.unicode.lower()
                            elif event.unicode != "-" or not field["text"]:
                                if event.unicode != "." or "." not in field["text"]:
                                    field["text"] += event.unicode
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if quit_btn.collidepoint(mx, my):
                        running = False
                        continue
                    clicked_input = None
                    for name, rect in field_rects.items():
                        if rect.collidepoint(mx, my):
                            clicked_input = name
                            break
                    if clicked_input is not None:
                        deactivate_all_inputs()
                        input_fields[clicked_input]["active"] = True
                        input_fields[clicked_input]["replace_on_type"] = True
                        continue
                    current_input = active_input_name()
                    if current_input is not None:
                        apply_input_field(current_input)
                        deactivate_all_inputs()
                    if settings_profile_btn.collidepoint(mx, my):
                        selected_settings_profile_idx = (
                            selected_settings_profile_idx + 1
                        ) % len(MASTERDATA_FREQUENCY_NAMES)
                        apply_masterdata_setting_profile(
                            masterdata_setting_profile_by_name(
                                MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx],
                                PRECISION_DISTANCE_MODIFIER_ENABLED,
                            )
                        )
                        deactivate_all_inputs(reset=True)
                        continue
                    if landmark_scenario_profile_btn.collidepoint(mx, my):
                        selected_landmark_scenario_profile_idx = (
                            selected_landmark_scenario_profile_idx + 1
                        ) % len(LANDMARK_SCENARIO_PROFILES)
                        ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME = LANDMARK_SCENARIO_PROFILES[selected_landmark_scenario_profile_idx]["name"]
                        ensure_simulation_core_configuration()
                        reset_landmark_obstruction_state()
                        continue
                    if analysis_btn.collidepoint(mx, my):
                        analysis_on = not analysis_on
                        continue
                    if algo_btn.collidepoint(mx, my):
                        selected_algorithm_idx = (selected_algorithm_idx + 1) % len(algorithm_names)
                        continue
                    if demo_btn.collidepoint(mx, my):
                        demo_state["selected_idx"] = (demo_state["selected_idx"] + 1) % len(demo_state["paths"])
                        demo_state["playing"] = False
                        demo_state["pending_reset"] = True
                        demo_state["time_s"] = 0.0
                        demo_state["sim_accumulator_s"] = 0.0
                        actual_pose = robot_pose_view(robots["actual"])
                        demo_state["trace"] = [(actual_pose["x_cm"], actual_pose["y_cm"])]
                        zero_robot_generalized_motion(robots["actual"])
                        continue
                    if demo_play_btn.collidepoint(mx, my):
                        if demo_state["pending_reset"]:
                            reset_demo_playback_state(robots["actual"], demo_state)
                        was_playing = demo_state["playing"]
                        demo_state["playing"] = not demo_state["playing"]
                        if was_playing and not demo_state["playing"]:
                            actual_pose = robot_pose_view(robots["actual"])
                            demo_state["pending_reset"] = True
                            demo_state["time_s"] = 0.0
                            demo_state["sim_accumulator_s"] = 0.0
                            demo_state["trace"] = [(actual_pose["x_cm"], actual_pose["y_cm"])]
                            zero_robot_generalized_motion(robots["actual"])
                        continue
                    if exp1_btn.collidepoint(mx, my):
                        if algorithm_uses_motion(algorithm_names[selected_algorithm_idx]):
                            last_history = run_demo_step_experiment(
                                algorithm_names[selected_algorithm_idx],
                                max_steps=1,
                                callback_interval=ANIMATION_INTERVAL,
                            )
                        else:
                            last_history = run_experiment(
                                robots,
                                algorithm_names[selected_algorithm_idx],
                                max_steps=1,
                                until_convergence=False,
                                step_callback=draw,
                                callback_interval=ANIMATION_INTERVAL,
                            )
                        continue
                    if exp100_btn.collidepoint(mx, my):
                        if algorithm_uses_motion(algorithm_names[selected_algorithm_idx]):
                            last_history = run_demo_step_experiment(
                                algorithm_names[selected_algorithm_idx],
                                max_steps=INFER_STEPS_SHORT,
                                callback_interval=ANIMATION_INTERVAL,
                            )
                        else:
                            last_history = run_experiment(
                                robots,
                                algorithm_names[selected_algorithm_idx],
                                max_steps=INFER_STEPS_SHORT,
                                until_convergence=False,
                                step_callback=draw,
                                callback_interval=ANIMATION_INTERVAL,
                            )
                        continue
                    if exp2000_btn.collidepoint(mx, my):
                        if algorithm_uses_motion(algorithm_names[selected_algorithm_idx]):
                            last_history = run_demo_step_experiment(
                                algorithm_names[selected_algorithm_idx],
                                max_steps=INFER_STEPS_LONG,
                                callback_interval=ANIMATION_INTERVAL,
                            )
                        else:
                            last_history = run_experiment(
                                robots,
                                algorithm_names[selected_algorithm_idx],
                                max_steps=INFER_STEPS_LONG,
                                until_convergence=False,
                                step_callback=draw,
                                callback_interval=ANIMATION_INTERVAL,
                            )
                        continue
                    if noise_toggle_btn.collidepoint(mx, my):
                        NOISE_ENABLED = not NOISE_ENABLED
                        continue
                    if viewing_toggle_btn.collidepoint(mx, my):
                        VIEWING_ANGLE_ENABLED = not VIEWING_ANGLE_ENABLED
                        continue
                    if odometry_bias_btn.collidepoint(mx, my):
                        ODOMETRY_BIAS_ENABLED = not ODOMETRY_BIAS_ENABLED
                        continue
                    if slip_btn.collidepoint(mx, my):
                        MOTION_SLIP_ENABLED = not MOTION_SLIP_ENABLED
                        reset_motion_slip_state()
                        continue
                    if dropout_btn.collidepoint(mx, my):
                        LANDMARK_DROPOUT_ENABLED = not LANDMARK_DROPOUT_ENABLED
                        continue
                    if obstruction_btn.collidepoint(mx, my):
                        LANDMARK_OBSTRUCTION_ENABLED = not LANDMARK_OBSTRUCTION_ENABLED
                        reset_landmark_obstruction_state()
                        continue
                    if precision_distance_toggle_btn.collidepoint(mx, my):
                        PRECISION_DISTANCE_MODIFIER_ENABLED = not PRECISION_DISTANCE_MODIFIER_ENABLED
                        apply_masterdata_setting_profile(
                            masterdata_setting_profile_by_name(
                                MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx],
                                PRECISION_DISTANCE_MODIFIER_ENABLED,
                            )
                        )
                        deactivate_all_inputs(reset=True)
                        continue
                    if plot_btn.collidepoint(mx, my):
                        plot_experiment(
                            last_history,
                            algorithm_names[selected_algorithm_idx],
                        )
                        plot_experiment(
                            last_history,
                            algorithm_names[selected_algorithm_idx],
                            paper_ready=True,
                            open_pdf=False,
                        )
                        continue
                    if robustness_btn.collidepoint(mx, my):
                        setup = dict(robustness_config, runs=CONVERGE_TEST_RUNS)
                        algorithm = algorithm_names[selected_algorithm_idx]
                        settings = experiment_settings(algorithm)
                        settings.update(algorithm_name=algorithm,
                                        profile_name=MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx],
                                        demo_path=deepcopy(demo_state["paths"][demo_state["selected_idx"]]),
                                        ball_fcs=dict(ball))
                        setup = convergence_setup_dialog(screen, setup, settings)
                        if setup is False:
                            running = False
                            continue
                        if setup is None:
                            continue
                        robustness_config = setup
                        initialize_path, step_path = robustness_demo_adapter(
                            demo_state["paths"], demo_state["selected_idx"], dict(ball),
                        )

                        def robustness_progress(point_id, offset, trial, runs, step, maximum, phase):
                            nonlocal converge_test_status, running
                            converge_test_status = f"ROB-{phase[0].upper()} {point_id}: {trial}/{runs} | {step}/{maximum} | ESC"
                            converge_test_status += f"\ndx {offset[0]:g} dy {offset[1]:g} cm | da {offset[2]:g} deg"
                            safe_draw()
                            for pending_event in pygame.event.get():
                                if pending_event.type == pygame.QUIT:
                                    running = False
                                    return False
                                if pending_event.type == pygame.KEYDOWN and pending_event.key == pygame.K_ESCAPE:
                                    return False
                                if pending_event.type == pygame.MOUSEBUTTONDOWN and quit_btn.collidepoint(pending_event.pos):
                                    running = False
                                    return False
                            return True

                        demo_state["playing"] = False
                        saved_analysis_on = analysis_on
                        analysis_on = False
                        try:
                            result = export_robustness_dataset(setup, algorithm, initialize_path, step_path,
                                                               settings, robustness_progress)
                        finally:
                            analysis_on = saved_analysis_on
                        outcome = "stopped" if result["stopped"] else "finished"
                        converge_test_status = f"ROB {outcome}: {result['tested_points']} tested\n{result['skipped_points']} skipped; reports/"
                        print(f"Robustness metadata: {result['metadata_path']}")
                        plot_robustness_report(result["metadata_path"], open_pdf=running, show_plot=running)
                        continue
                    if paper_lm_btn.collidepoint(mx, my):
                        paper_lm_setup = capture_paper_lm_setup(
                            robots,
                            ball,
                            demo_state,
                            algorithm_names[selected_algorithm_idx],
                            MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx],
                        )
                        def lm_progress(profile, trial, total, step, max_steps):
                            nonlocal converge_test_status, running
                            converge_test_status = f"P-LM {profile}: {trial}/{total}, step {step}/{max_steps}"
                            safe_draw()
                            for pending_event in pygame.event.get():
                                if pending_event.type == pygame.QUIT:
                                    running = False
                                    return False
                                if pending_event.type == pygame.KEYDOWN and pending_event.key == pygame.K_ESCAPE:
                                    return False
                                if pending_event.type == pygame.MOUSEBUTTONDOWN and quit_btn.collidepoint(pending_event.pos):
                                    running = False
                                    return False
                            return True

                        demo_state["playing"] = False
                        saved_analysis_on = analysis_on
                        analysis_on = False
                        try:
                            paper_lm_result = export_paper_lm_dataset(
                                paper_lm_setup, step_paper_lm_demo, progress_callback=lm_progress,
                            )
                        finally:
                            analysis_on = saved_analysis_on
                        last_history = paper_lm_result["last_history"]
                        completed = sum(trial["completed"] for trial in paper_lm_result["trials"])
                        outcome = "stopped" if paper_lm_result["stopped"] else "finished"
                        converge_test_status = f"P-LM {outcome}: {completed}/{len(paper_landmark_profiles()) * paper_lm_setup['runs']} trials saved in reports"
                        print(f"P-LM CSV: {paper_lm_result['csv_path']}")
                        print(f"P-LM metadata: {paper_lm_result['json_path']}")
                        paper_lm_result.update(plot_paper_lm_report(paper_lm_result["csv_path"], paper_lm_result["json_path"]))
                        continue
                    if paper_data_btn.collidepoint(mx, my):
                        actual_start = robot_pose_view(robots["actual"])
                        belief_start = random_nearby_belief_pose(
                            {"x": actual_start["x_cm"], "y": actual_start["y_cm"], "theta": actual_start["theta_rad"]},
                        )
                        robots["belief"] = make_robot_gc_state(belief_start["x"], belief_start["y"], belief_start["theta"])
                        paper_algorithm_name = algorithm_names[selected_algorithm_idx]
                        paper_profile_name = MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx]
                        paper_path_name = demo_state["paths"][demo_state["selected_idx"]]["name"]
                        converge_test_status = f"Paper data {paper_path_name}: starting"
                        safe_draw()
                        pygame.event.pump()
                        plt.pause(0.001)

                        def paper_progress(current, total, csv_path):
                            nonlocal converge_test_status
                            pct = 100.0 * current / total if total > 0 else 100.0
                            converge_test_status = f"Paper data: {current}/{total} ({pct:.1f}%)"
                            safe_draw()
                            pygame.event.pump()
                            plt.pause(0.001)

                        export_result = export_demo_tracking_dataset(
                            robots,
                            paper_algorithm_name,
                            demo_state["paths"],
                            demo_state["selected_idx"],
                            duration_s=PAPER_VISIBILITY_STRESS_DURATION_S,
                            profile_name=paper_profile_name,
                            look_target_x_cm=ball["x_cm"],
                            look_target_y_cm=ball["y_cm"],
                            progress_callback=paper_progress,
                        )
                        converge_test_status = f"Paper data saved: {export_result['steps']} rows"
                        last_history = []
                        demo_state["playing"] = False
                        demo_state["pending_reset"] = True
                        safe_draw()
                        pygame.event.pump()
                        plt.pause(0.001)
                        continue
                    if reset_belief_btn.collidepoint(mx, my):
                        for axis in ("x", "y", "theta"):
                            robots["belief"][axis] = list(robots["actual"][axis])
                        last_history = []
                        continue
                    if convdynamic_btn.collidepoint(mx, my) or convdynamic_near_btn.collidepoint(mx, my):
                        converge_algorithm_name = algorithm_names[selected_algorithm_idx]
                        demo_path_name = demo_state["paths"][demo_state["selected_idx"]]["name"]
                        dynamic_start_mode = "nearby" if convdynamic_near_btn.collidepoint(mx, my) else "random"
                        status_label = f"Dynamic {dynamic_start_mode} {demo_path_name}"
                        near_limits = None
                        if dynamic_start_mode == "nearby":
                            settings = experiment_settings(converge_algorithm_name)
                            settings.update(algorithm_name=converge_algorithm_name,
                                            profile_name=MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx],
                                            demo_path=demo_state["paths"][demo_state["selected_idx"]], ball_fcs=dict(ball))
                            near_limits = convergence_setup_dialog(screen, dict(NEAR_START_LIMITS), settings, near_test="D-NEAR")
                            if near_limits is False:
                                running = False
                                continue
                            if near_limits is None:
                                continue

                        def progress_update(current, total, conv, nonconv):
                            nonlocal converge_test_status
                            if current % 10 != 0 and current != total:
                                return
                            converge_test_status = f"{status_label}: {current}/{total}"
                            safe_draw()
                            pygame.event.pump()
                            plt.pause(0.001)

                        start_cases = PROFILE_REGRESSION_START_POSES if CONVERGE_TEST_USE_REGRESSION_SET else None
                        total_cases = len(start_cases) if start_cases is not None else CONVERGE_TEST_RUNS
                        converge_test_status = f"{status_label}: starting"
                        safe_draw()
                        pygame.event.pump()
                        plt.pause(0.001)

                        converge_test_report = run_dynamic_converge_test(
                            robots,
                            converge_algorithm_name,
                            demo_state["paths"],
                            demo_state["selected_idx"],
                            runs=total_cases,
                            progress_callback=progress_update,
                            start_mode=dynamic_start_mode,
                            start_cases=start_cases,
                            look_target_x_cm=ball["x_cm"],
                            look_target_y_cm=ball["y_cm"],
                            profile_name=MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx],
                            near_start_limits=near_limits,
                        )
                        converge_test_status = None
                        last_history = converge_test_report["last_history"]
                        demo_state["playing"] = False
                        demo_state["pending_reset"] = True
                        plot_converge_test_report(converge_test_report, converge_algorithm_name)
                        continue
                    if convtest_btn.collidepoint(mx, my) or convnear_btn.collidepoint(mx, my) or convcorner_btn.collidepoint(mx, my):
                        converge_algorithm_name = LANDMARK_ONLY_ALGORITHM_NAME
                        if convnear_btn.collidepoint(mx, my):
                            start_mode = "nearby"
                            status_label = "Landmark converge nearby"
                        elif convcorner_btn.collidepoint(mx, my):
                            start_mode = "outside_corner"
                            status_label = "Landmark converge outside corner"
                        else:
                            start_mode = "random"
                            status_label = "Landmark converge test"
                        near_limits = None
                        if start_mode == "nearby":
                            settings = experiment_settings(converge_algorithm_name)
                            settings.update(algorithm_name=converge_algorithm_name,
                                            profile_name=MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx],
                                            demo_path=demo_state["paths"][demo_state["selected_idx"]], ball_fcs=dict(ball))
                            near_limits = convergence_setup_dialog(screen, dict(NEAR_START_LIMITS), settings, near_test="S-NEAR")
                            if near_limits is False:
                                running = False
                                continue
                            if near_limits is None:
                                continue

                        def progress_update(current, total, conv, nonconv):
                            nonlocal converge_test_status
                            if current % 10 != 0 and current != total:
                                return
                            converge_test_status = f"{status_label}: {current}/{total}"
                            safe_draw()
                            pygame.event.pump()
                            plt.pause(0.001)

                        start_cases = PROFILE_REGRESSION_START_POSES if CONVERGE_TEST_USE_REGRESSION_SET else None
                        total_cases = len(start_cases) if start_cases is not None else CONVERGE_TEST_RUNS
                        converge_test_status = f"{status_label}: starting"
                        safe_draw()
                        pygame.event.pump()
                        plt.pause(0.001)

                        converge_test_report = run_converge_test(
                            robots,
                            converge_algorithm_name,
                            runs=total_cases,
                            progress_callback=progress_update,
                            start_mode=start_mode,
                            start_cases=start_cases,
                            near_start_limits=near_limits,
                        )
                        converge_test_status = None
                        last_history = converge_test_report['last_history']
                        plot_converge_test_report(converge_test_report, converge_algorithm_name)
                        continue
                    if mx >= world_w:
                        continue

                    # Select robot automatically when clicking near it; only then start dragging
                    actual_pose = robot_pose_view(robots["actual"])
                    belief_pose = robot_pose_view(robots["belief"])
                    actual_px, actual_py = fcs_to_screen(actual_pose["x_cm"], actual_pose["y_cm"], world_w, world_h, scale)
                    belief_px, belief_py = fcs_to_screen(belief_pose["x_cm"], belief_pose["y_cm"], world_w, world_h, scale)
                    ball_px, ball_py = fcs_to_screen(ball["x_cm"], ball["y_cm"], world_w, world_h, scale)

                    d_actual2 = (mx - actual_px) ** 2 + (my - actual_py) ** 2
                    d_belief2 = (mx - belief_px) ** 2 + (my - belief_py) ** 2
                    d_ball2 = (mx - ball_px) ** 2 + (my - ball_py) ** 2
                    robot_pick_r2 = ROBOT_PICK_RADIUS_PX ** 2
                    ball_pick_r2 = BALL_PICK_RADIUS_PX ** 2

                    if d_actual2 <= robot_pick_r2 or d_belief2 <= robot_pick_r2 or d_ball2 <= ball_pick_r2:
                        candidates = []
                        if d_actual2 <= robot_pick_r2:
                            candidates.append(("actual", d_actual2))
                        if d_belief2 <= robot_pick_r2:
                            candidates.append(("belief", d_belief2))
                        if d_ball2 <= ball_pick_r2:
                            candidates.append(("ball", d_ball2))
                        selected = min(candidates, key=lambda item: item[1])[0]
                        dragging = True
                        x_cm, y_cm = screen_to_fcs(mx, my, world_w, world_h, scale)
                        if selected == "ball":
                            rx = ball["x_cm"]
                            ry = ball["y_cm"]
                        else:
                            selected_pose = robot_pose_view(robots[selected])
                            rx = selected_pose["x_cm"]
                            ry = selected_pose["y_cm"]
                        drag_offset_cm = (rx - x_cm, ry - y_cm)
                    else:
                        dragging = False
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                if mx < world_w:
                    x_cm, y_cm = screen_to_fcs(mx, my, world_w, world_h, scale)
                    x_cm += drag_offset_cm[0]
                    y_cm += drag_offset_cm[1]
                    x_cm, y_cm = clamp_world(x_cm, y_cm)
                    if selected == "ball":
                        ball["x_cm"] = x_cm
                        ball["y_cm"] = y_cm
                        demo_state["look_target_x_cm"] = x_cm
                        demo_state["look_target_y_cm"] = y_cm
                    else:
                        if selected == "actual":
                            demo_state["playing"] = False
                            demo_state["pending_reset"] = True
                            demo_state["trace"] = [(x_cm, y_cm)]
                        scalar_2_gc(robots[selected], x_cm=x_cm, y_cm=y_cm)
                        zero_robot_generalized_motion(robots[selected])

        if demo_state["look_target_x_cm"] != ball["x_cm"] or demo_state["look_target_y_cm"] != ball["y_cm"]:
            demo_state["look_target_x_cm"] = ball["x_cm"]
            demo_state["look_target_y_cm"] = ball["y_cm"]

        if demo_state["playing"] and DT > 0.0:
            demo_state["sim_accumulator_s"] += frame_dt_s
            sim_steps = 0
            while (
                demo_state["sim_accumulator_s"] >= DT
                and sim_steps < MAX_REALTIME_SIM_STEPS_PER_FRAME
            ):
                step_demo_playback(robots["actual"], demo_state, DT)
                simulation_inference_step(robots, algorithm_names[selected_algorithm_idx])
                demo_state["sim_accumulator_s"] -= DT
                sim_steps += 1
        else:
            demo_state["sim_accumulator_s"] = 0.0

        keys = pygame.key.get_pressed()
        if selected != "ball" and keys[pygame.K_q]:
            if selected == "actual":
                actual_pose = robot_pose_view(robots["actual"])
                demo_state["playing"] = False
                demo_state["pending_reset"] = True
                demo_state["trace"] = [(actual_pose["x_cm"], actual_pose["y_cm"])]
            selected_pose = robot_pose_view(robots[selected])
            scalar_2_gc(robots[selected], theta_rad=selected_pose["theta_rad"] + ROTATE_STEP_RAD)
            zero_robot_generalized_motion(robots[selected])
        if selected != "ball" and keys[pygame.K_e]:
            if selected == "actual":
                actual_pose = robot_pose_view(robots["actual"])
                demo_state["playing"] = False
                demo_state["pending_reset"] = True
                demo_state["trace"] = [(actual_pose["x_cm"], actual_pose["y_cm"])]
            selected_pose = robot_pose_view(robots[selected])
            scalar_2_gc(robots[selected], theta_rad=selected_pose["theta_rad"] - ROTATE_STEP_RAD)
            zero_robot_generalized_motion(robots[selected])

        safe_draw()
        plt.pause(0.001)

    pygame.quit()


def main_cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Active-inference self-localization POC")
    parser.add_argument(
        "--export-paper-data",
        action="store_true",
        help="Run the demo tracking exporter headlessly instead of opening the UI.",
    )
    parser.add_argument(
        "--paper-data-steps",
        type=int,
        default=None,
        help="Number of headless export steps. Overrides --paper-data-duration-s.",
    )
    parser.add_argument(
        "--paper-data-duration-s",
        type=float,
        default=None,
        help="Headless export duration in seconds. Defaults to the configured 2-hour paper duration.",
    )
    parser.add_argument(
        "--paper-data-profile",
        default="40Hz",
        help="Masterdata profile name for headless export, e.g. 40Hz or 200Hz.",
    )
    parser.add_argument(
        "--paper-data-distance-precision-modifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable distance-based landmark precision for headless export (default: enabled).",
    )
    parser.add_argument(
        "--paper-data-path",
        default="Circle",
        help="Demo path name for headless export, e.g. Circle, Accel slalom, or Random targets.",
    )
    parser.add_argument(
        "--paper-data-stress",
        action="store_true",
        help="Apply the 5.3 visibility-stress settings: noise, 110-degree FOV, dropout, obstruction, and unique-ID association.",
    )
    parser.add_argument(
        "--paper-data-association",
        choices=configured_association_names(),
        default=None,
        help="Landmark association mode for headless export. Defaults to Unique ID when --paper-data-stress is used.",
    )
    parser.add_argument(
        "--paper-data-start",
        choices=["fixed_offset", "nearby", "random", "converged"],
        default="fixed_offset",
        help="Headless export start condition. Use nearby to avoid a long initial convergence transient.",
    )
    parser.add_argument(
        "--paper-data-filename-prefix",
        default="paper_visibility_stress_dataset_FOR_PAPER",
        help="Filename prefix for the exported CSV/JSON files.",
    )
    args = parser.parse_args(argv)

    if args.export_paper_data:
        result = run_headless_paper_dataset_export(
            steps=args.paper_data_steps,
            duration_s=args.paper_data_duration_s,
            profile_name=args.paper_data_profile,
            demo_path_name=args.paper_data_path,
            apply_paper_stress=args.paper_data_stress,
            start_mode=args.paper_data_start,
            filename_prefix=args.paper_data_filename_prefix,
            association_name=args.paper_data_association,
            precision_distance_modifier_enabled=args.paper_data_distance_precision_modifier,
        )
        print(f"CSV: {result['csv_path']}")
        print(f"JSON: {result['json_path']}")
        print(f"Steps: {result['steps']}")
        print(f"Duration_s: {result['duration_s']:.3f}")
        return 0

    main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
