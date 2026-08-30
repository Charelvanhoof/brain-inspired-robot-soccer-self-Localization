# Code for the Robot_Soccer_Self_Localization_with_Active_Inference
# Based on version POC V28
import math
import random
import sys
import textwrap
import types

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
# CONFIG (all in cm)
# =========================
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
CONVERGE_TEST_RUNS = 100
CONVERGE_TEST_USE_REGRESSION_SET = False
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
DATA_ASSOCIATION_UNIQUE_DIRECT_NAME = "Unique direct"
DATA_ASSOCIATION_ALGORITHM_NAMES = [
    DATA_ASSOCIATION_GATED_NEAREST_NAME,
    DATA_ASSOCIATION_UNIQUE_ID_NAME,
]
ACTIVE_DATA_ASSOCIATION_NAME = DATA_ASSOCIATION_UNIQUE_ID_NAME
LANDMARK_ASSOCIATION_GATE_CM = 1000.0
GENERALIZED_COORDINATE_ORDER = 2
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
        "lr_angle": DEFAULT_ANGLE_LR * 0.8,
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
        "lr_angle": 7.000e-07,
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
        "lr_angle": 8.813e-05,
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
        "lr_angle": 7.000e-07,
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
def make_landmarks():
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    return {
        "center": (0.0, 0.0),
        "left_middle": (0.0, halfW),
        "right_middle": (0.0, -halfW),
        "corner_TR": (halfL, -halfW),
        "corner_TL": (halfL, halfW),
        "corner_BR": (-halfL, -halfW),
        "corner_BL": (-halfL, halfW),
        "goal_opponent": (halfL, 0.0),
        "goal_own": (-halfL, 0.0),
    }


LANDMARKS = make_landmarks()

LANDMARK_SCENARIO_PROFILES = [
    {
        "name": "9",
        "landmark_ids": [
            "center",
            "left_middle",
            "right_middle",
            "corner_TR",
            "corner_TL",
            "corner_BR",
            "corner_BL",
            "goal_opponent",
            "goal_own",
        ],
    },
    {
        "name": "2",
        "landmark_ids": [
            "goal_opponent",
            "goal_own",
        ],
    },
    {
        "name": "1",
        "landmark_ids": [
            "center",
        ],
    },
    {
        "name": "5",
        "landmark_ids": [
            "goal_opponent",
            "goal_own",
            "center",
            "left_middle",
            "right_middle",
        ],
    },
]
ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME = LANDMARK_SCENARIO_PROFILES[0]["name"]


def make_landmark_metadata():
    return {
        "center": {"type": "center", "unique": True},
        "left_middle": {"type": "middle", "unique": False},
        "right_middle": {"type": "middle", "unique": False},
        "corner_TR": {"type": "corner", "unique": False},
        "corner_TL": {"type": "corner", "unique": False},
        "corner_BR": {"type": "corner", "unique": False},
        "corner_BL": {"type": "corner", "unique": False},
        "goal_opponent": {"type": "goal", "unique": False},
        "goal_own": {"type": "goal", "unique": False},
    }


LANDMARK_METADATA = make_landmark_metadata()


def landmark_scenario_profile_by_name(profile_name):
    for profile in LANDMARK_SCENARIO_PROFILES:
        if profile["name"] == profile_name:
            return profile
    raise ValueError(f"Unknown landmark scenario profile: {profile_name}")


def active_landmark_scenario_profile():
    return landmark_scenario_profile_by_name(ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME)


def active_simulation_landmarks():
    active_ids = set(active_landmark_scenario_profile()["landmark_ids"])
    return {
        landmark_id: position
        for landmark_id, position in LANDMARKS.items()
        if landmark_id in active_ids
    }


def step_count_for_duration(duration_s):
    return max(1, int(round(float(duration_s) / DT)))


def landmark_type(landmark_id):
    return LANDMARK_METADATA[landmark_id]["type"]


def landmark_candidate_ids_for_type(landmark_type_name):
    return [
        landmark_id
        for landmark_id in LANDMARKS.keys()
        if landmark_type(landmark_id) == landmark_type_name
    ]


def unique_landmark_candidate_id_for_type(landmark_type_name):
    candidates = [
        landmark_id
        for landmark_id in landmark_candidate_ids_for_type(landmark_type_name)
        if LANDMARK_METADATA[landmark_id].get("unique", False)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


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
    global NOISE_ENABLED, VIEWING_ANGLE_ENABLED, LANDMARK_DROPOUT_ENABLED, LANDMARK_OBSTRUCTION_ENABLED, ACTIVE_DATA_ASSOCIATION_NAME
    NOISE_ENABLED = True
    VIEWING_ANGLE_ENABLED = True
    LANDMARK_DROPOUT_ENABLED = True
    LANDMARK_OBSTRUCTION_ENABLED = True
    ACTIVE_DATA_ASSOCIATION_NAME = association_name
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
# CORE
# =========================

def wrap_rad(angle_rad: float) -> float:
    # Input:
    #   angle_rad: angle in radians
    # Output:
    #   wrapped angle in radians in the interval [-pi, pi)
    # Function:
    #   wraps any angle value to the principal angle range
    #   used to keep angular errors and headings bounded
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def make_robot_gc_state(x_cm=0.0, y_cm=0.0, theta_rad=0.0):
    # Input:
    #   x_cm, y_cm, theta_rad: zeroth-order robot state values used to initialize generalized coordinates
    # Output:
    #   robot state in generalized-coordinate form with x, y, and theta series
    # Function:
    #   constructs the Core robot state representation from zeroth-order position/orientation inputs
    #   initializes higher-order generalized coordinates to zero
    def make_generalized_series(value=0.0, order=GENERALIZED_COORDINATE_ORDER):
        return [float(value)] + [0.0] * int(order)

    return {
        "x": make_generalized_series(x_cm),
        "y": make_generalized_series(y_cm),
        "theta": make_generalized_series(theta_rad),
    }

def scalar_2_gc(robot_fcs_gc, x_cm=None, y_cm=None, theta_rad=None):
    # Input:
    #   robot_fcs_gc: robot state in generalized-coordinate form
    #   x_cm, y_cm, theta_rad: optional scalar zeroth-order values to write into generalized coordinates
    # Output:
    #   robot state in generalized-coordinate form after the requested zeroth-order updates
    # Function:
    #   writes scalar position/orientation values into the zeroth-order generalized coordinates
    #   leaves higher-order generalized coordinates unchanged
    if x_cm is not None:
        robot_fcs_gc["x"][0] = float(x_cm)
    if y_cm is not None:
        robot_fcs_gc["y"][0] = float(y_cm)
    if theta_rad is not None:
        robot_fcs_gc["theta"][0] = wrap_rad(float(theta_rad))
    return robot_fcs_gc


def gc_2_scalar(robot_fcs_gc):
    # Input:
    #   robot_fcs_gc: robot state in generalized-coordinate form
    # Output:
    #   scalar zeroth-order state values as x_cm, y_cm, theta_rad
    # Function:
    #   collapses the generalized coordinates into a flat scalar view ( the positions without velocity and acceleration)
    #   used when code needs position/orientation values without the higher-order terms
    return (
        float(robot_fcs_gc["x"][0]),
        float(robot_fcs_gc["y"][0]),
        wrap_rad(float(robot_fcs_gc["theta"][0])),
    )


def obs_list_to_map(obs_list):
    # Input:
    #   obs_list: list of observation dicts, each carrying an "id" or "obs_id" field
    # Output:
    #   observation map keyed by landmark id or sensor observation id
    # Function:
    #   converts a list-form sensory observation set into a lookup map
    #   used so landmark predictions or typed observations can be accessed directly during inference and analysis
    return {o.get("id", o.get("obs_id")): o for o in obs_list}


def motion_prediction_error(belief_fcs_gc, y_motion_obs):
    # Input:
    #   belief_fcs_gc: belief robot state in FCS generalized coordinates
    #   y_motion_obs: sensed motion observation in RCS containing velocity and acceleration terms
    # Output:
    #   per-axis generalized motion prediction errors for x, y, and theta
    #   each axis returns the components of D*mu - f(y):
    #   vel_error, acc_error, and jerk_error
    # Function:
    #   computes the D-translated generalized-coordinate prediction error
    #   D*mu - f(y) for velocity, acceleration and jerk (is 0).
    #   The sensory motion in RCS is converted to FCS, to calculate the prediction error in the same reference frame as the belief state. 

    obs_x_vel_fcs, obs_y_vel_fcs, obs_theta_vel = vel_rcs_2_fcs(
        belief_fcs_gc,
        y_motion_obs["x_vel_cm_s"],
        y_motion_obs["y_vel_cm_s"],
        y_motion_obs["theta_vel_rad_s"],
    )
    obs_x_acc_fcs, obs_y_acc_fcs, obs_theta_acc = acc_rcs_2_fcs(
        belief_fcs_gc,
        y_motion_obs["x_acc_cm_s2"],
        y_motion_obs["y_acc_cm_s2"],
        y_motion_obs["theta_acc_rad_s2"],
    )
    return {
        "x": {
            "vel_error": belief_fcs_gc["x"][1] - obs_x_vel_fcs,
            "acc_error": belief_fcs_gc["x"][2] - obs_x_acc_fcs,
            "jerk_error": 0.0,
        },
        "y": {
            "vel_error": belief_fcs_gc["y"][1] - obs_y_vel_fcs,
            "acc_error": belief_fcs_gc["y"][2] - obs_y_acc_fcs,
            "jerk_error": 0.0,
        },
        "theta": {
            "vel_error": belief_fcs_gc["theta"][1] - obs_theta_vel,
            "acc_error": belief_fcs_gc["theta"][2] - obs_theta_acc,
            "jerk_error": 0.0,
        },
    }


def motion_inference_step(belief_fcs_gc, y_motion_obs, step_dt_s=DT):
    # Input:
    #   belief_fcs_gc: belief robot state in FCS generalized coordinates
    #   y_motion_obs: motion observation treated as the sensory input to inference
    #   step_dt_s: timestep for this motion inference update
    # Output:
    #   Updates the belief_fcs_gc state
    #   Output for reporting and visualisation / diagnostics
    #   motion_terms: D*mu - f(y) generalized motion prediction errors for x, y, and theta
    #   F: free energy = precision-weighted sum of squared motion prediction errors
    # Function:
    #   performs one active-inference motion update step and updates the belief_fcs_gc state
    #   returns also motion_terms and free_energy as diagnostic/reporting outputs

    # Calculate the D-translated motion prediction error D*mu - f(y). In FCS.
    motion_terms = motion_prediction_error(belief_fcs_gc, y_motion_obs)

    # calculate mu_dot for x, y and angle generalized coordinates
    # mu_dot = D*mu - D^T*Pi*(D*mu - f(y))
    x_mu_dot = [
        belief_fcs_gc["x"][1] - 0.0,
        belief_fcs_gc["x"][2] - MOTION_LR_VEL_XY * PI_ODO * motion_terms["x"]["vel_error"],
        0.0                   - MOTION_LR_ACC_XY * PI_IMU * motion_terms["x"]["acc_error"],
    ]
    y_mu_dot = [
        belief_fcs_gc["y"][1] - 0.0,
        belief_fcs_gc["y"][2] - MOTION_LR_VEL_XY * PI_ODO * motion_terms["y"]["vel_error"],
        0.0                   - MOTION_LR_ACC_XY * PI_IMU * motion_terms["y"]["acc_error"],
    ]
    theta_mu_dot = [
        belief_fcs_gc["theta"][1] - 0.0,
        belief_fcs_gc["theta"][2] - MOTION_LR_VEL_THETA * PI_ODO * motion_terms["theta"]["vel_error"],
        0.0                       - MOTION_LR_ACC_THETA * PI_IMU * motion_terms["theta"]["acc_error"],
    ]

    # Euler forward update rule mu(t+dt) = mu(t) + dt * mu_dot
    belief_fcs_gc["x"][0] += step_dt_s * x_mu_dot[0]
    belief_fcs_gc["x"][1] += step_dt_s * x_mu_dot[1]
    belief_fcs_gc["x"][2] += step_dt_s * x_mu_dot[2]

    belief_fcs_gc["y"][0] += step_dt_s * y_mu_dot[0]
    belief_fcs_gc["y"][1] += step_dt_s * y_mu_dot[1]
    belief_fcs_gc["y"][2] += step_dt_s * y_mu_dot[2]

    belief_fcs_gc["theta"][0] = wrap_rad(belief_fcs_gc["theta"][0] + step_dt_s * theta_mu_dot[0])
    belief_fcs_gc["theta"][1] += step_dt_s * theta_mu_dot[1]
    belief_fcs_gc["theta"][2] += step_dt_s * theta_mu_dot[2]

    # Calculate the free energy contribution
    free_energy = 0.0
    free_energy += PI_ODO * motion_terms["x"]["vel_error"] ** 2
    free_energy += PI_IMU * motion_terms["x"]["acc_error"] ** 2
    free_energy += PI_ODO * motion_terms["y"]["vel_error"] ** 2
    free_energy += PI_IMU * motion_terms["y"]["acc_error"] ** 2
    free_energy += PI_ODO * motion_terms["theta"]["vel_error"] ** 2
    free_energy += PI_IMU * motion_terms["theta"]["acc_error"] ** 2
    return motion_terms, free_energy


def precision_distance_modifier(pred_obs):
    # Input:
    #   pred_obs: predicted landmark observation in RCS with x_rcs_cm, y_rcs_cm
    # Output:
    #   multiplicative precision modifier based on predicted landmark distance
    # Function:
    #   reduces precision for farther landmarks based on predicted range
    #   range_cm = sqrt(x_rcs_cm^2 + y_rcs_cm^2)
    #   modifier = max(min_factor, 1 / (1 + (range_cm / ref_cm)^2))
    #   if disabled, landmark precision returns 1.0 (no modifier applied)
    if not PRECISION_DISTANCE_MODIFIER_ENABLED:
        return 1.0

    range_cm = math.hypot(pred_obs["x_rcs_cm"], pred_obs["y_rcs_cm"])
    range_ratio = range_cm / DISTANCE_PRECISION_REF_CM if DISTANCE_PRECISION_REF_CM > 0.0 else 0.0
    modifier = 1.0 / (1.0 + range_ratio * range_ratio)
    modifier = max(DISTANCE_PRECISION_MIN_FACTOR, modifier)
    return modifier



def landmark_fcs_2_rcs(robot_fcs_gc, x_cm, y_cm):
    # Input:
    #   robot state in generalized coordinates
    #   position landmark in FCS: x_cm, y_cm
    # Output:
    #   where the landmark is observed in robot coordinates (RCS): x_rcs_cm, y_rcs_cm, angle_rad
    # Function:
    #   FCS -> RCS translation
    #   converts a field position into the robot reference frame
    #   Is the g(mu) function of sensory mapping in active inference

    rx, ry, theta = gc_2_scalar(robot_fcs_gc)
    c = math.cos(theta)
    s = math.sin(theta)

    dx = x_cm - rx
    dy = y_cm - ry
    x_rcs = c * dx + s * dy
    y_rcs = -s * dx + c * dy
    angle_rad = math.atan2(y_rcs, x_rcs)
    return x_rcs, y_rcs, angle_rad


def vel_fcs_2_rcs(robot_fcs_gc, x_vel_fcs, y_vel_fcs, theta_vel):
    # Input:
    #   robot state in generalized coordinates
    #   translational velocity in FCS (cm/s): x_vel_fcs, y_vel_fcs
    #   angular velocity (rad/s): theta_vel
    # Output:
    #   velocity in robot coordinates (RCS, cm/s and rad/s): x_vel_rcs, y_vel_rcs, theta_vel
    # Function:
    #   FCS -> RCS translation for planar velocity vectors
    #   rotates the field-frame velocity into the robot body frame

    # Extract heading from the generalized-coordinate state.
    _, _, theta = gc_2_scalar(robot_fcs_gc)
    c = math.cos(theta)
    s = math.sin(theta)

    # Rotate the FCS velocity vector into the robot frame.
    x_vel_rcs = c * x_vel_fcs + s * y_vel_fcs
    y_vel_rcs = -s * x_vel_fcs + c * y_vel_fcs

    # Angular velocity is already aligned with the planar rotation axis.
    return x_vel_rcs, y_vel_rcs, theta_vel


def acc_fcs_2_rcs(robot_fcs_gc, x_acc_fcs, y_acc_fcs, theta_acc):
    # Input:
    #   robot state in generalized coordinates
    #   translational acceleration in FCS (cm/s^2): x_acc_fcs, y_acc_fcs
    #   angular acceleration (rad/s^2): theta_acc
    # Output:
    #   acceleration in robot coordinates (RCS, cm/s^2 and rad/s^2): x_acc_rcs, y_acc_rcs, theta_acc
    # Function:
    #   FCS -> RCS translation for planar acceleration vectors
    #   rotates the field-frame acceleration into the robot body frame

    # Extract heading from the generalized-coordinate state.
    _, _, theta = gc_2_scalar(robot_fcs_gc)
    c = math.cos(theta)
    s = math.sin(theta)

    # Rotate the FCS acceleration vector into the robot frame.
    x_acc_rcs = c * x_acc_fcs + s * y_acc_fcs
    y_acc_rcs = -s * x_acc_fcs + c * y_acc_fcs

    # Angular acceleration is already aligned with the planar rotation axis.
    return x_acc_rcs, y_acc_rcs, theta_acc


def vel_rcs_2_fcs(robot_fcs_gc, x_vel_rcs, y_vel_rcs, theta_vel):
    # Input:
    #   robot state in generalized coordinates
    #   translational velocity in RCS (cm/s): x_vel_rcs, y_vel_rcs
    #   angular velocity (rad/s): theta_vel
    # Output:
    #   velocity in field coordinates (FCS, cm/s and rad/s): x_vel_fcs, y_vel_fcs, theta_vel
    # Function:
    #   RCS -> FCS translation for planar velocity vectors
    #   rotates the robot-frame velocity into the field frame

    # Extract heading from the generalized-coordinate state.
    _, _, theta = gc_2_scalar(robot_fcs_gc)
    c = math.cos(theta)
    s = math.sin(theta)

    # Rotate the RCS velocity vector into the field frame.
    x_vel_fcs = c * x_vel_rcs - s * y_vel_rcs
    y_vel_fcs = s * x_vel_rcs + c * y_vel_rcs

    # Angular velocity is already aligned with the planar rotation axis.
    return x_vel_fcs, y_vel_fcs, theta_vel


def acc_rcs_2_fcs(robot_fcs_gc, x_acc_rcs, y_acc_rcs, theta_acc):
    # Input:
    #   robot state in generalized coordinates
    #   translational acceleration in RCS (cm/s^2): x_acc_rcs, y_acc_rcs
    #   angular acceleration (rad/s^2): theta_acc
    # Output:
    #   acceleration in field coordinates (FCS, cm/s^2 and rad/s^2): x_acc_fcs, y_acc_fcs, theta_acc
    # Function:
    #   RCS -> FCS translation for planar acceleration vectors
    #   rotates the robot-frame acceleration into the field frame

    # Extract heading from the generalized-coordinate state.
    _, _, theta = gc_2_scalar(robot_fcs_gc)
    c = math.cos(theta)
    s = math.sin(theta)

    # Rotate the RCS acceleration vector into the field frame.
    x_acc_fcs = c * x_acc_rcs - s * y_acc_rcs
    y_acc_fcs = s * x_acc_rcs + c * y_acc_rcs

    # Angular acceleration is already aligned with the planar rotation axis.
    return x_acc_fcs, y_acc_fcs, theta_acc


# NOTE:
# `motion_jacobian_fcs_2_rcs(...)` is currently not used in V22.
# Keeping the implementation commented out here as reference code rather than
# deleting it, so it stays easy to restore if this Jacobian becomes relevant again.
#
# def motion_jacobian_fcs_2_rcs(robot_fcs_gc):
#     # Input:
#     #   robot state in generalized coordinates
#     # Output:
#     #   3x3 Jacobian matrix that maps [x_motion_fcs, y_motion_fcs, theta_motion]
#     #   to [x_motion_rcs, y_motion_rcs, theta_motion]
#     # Function:
#     #   computes the FCS -> RCS local linearization for planar motion vectors
#     #   uses theta only as a fixed rotation parameter
#     #   shared by both the velocity and acceleration motion transforms
#
#     # Extract heading from the generalized-coordinate state.
#     _, _, theta = gc_2_scalar(robot_fcs_gc)
#     c = math.cos(theta)
#     s = math.sin(theta)
#
#     # Build the planar rotation-plus-pass-through Jacobian.
#     J = [
#         [c, s, 0.0],
#         [-s, c, 0.0],
#         [0.0, 0.0, 1.0],
#     ]
#     return J


def landmark_jacobian_fcs_2_rcs(robot_fcs_gc, x_cm, y_cm):
    # Input:
    #   robot state in generalized coordinates
    #   landmark position in FCS: x_cm, y_cm
    # Output:
    #   Jacobian matrix of the sensory mapping g(mu) from FCS to RCS
    #   rows correspond to x_rcs, y_rcs, angle_rad
    #   columns correspond to robot x_cm, y_cm, theta_rad
    # Function:
    #   computes the local linearization of the landmark observation model
    #   used to translate sensory prediction errors into belief updates
    rx, ry, theta = gc_2_scalar(robot_fcs_gc)
    c = math.cos(theta)
    s = math.sin(theta)

    dx = x_cm - rx
    dy = y_cm - ry

    x_rcs = c * dx + s * dy
    y_rcs = -s * dx + c * dy
    r2 = max(x_rcs * x_rcs + y_rcs * y_rcs, 1e-9) # protect against division by 0 error

    dxrcs_dx = -c
    dxrcs_dy = -s
    dxrcs_dtheta = y_rcs

    dyrcs_dx = s
    dyrcs_dy = -c
    dyrcs_dtheta = -x_rcs

    dalpha_dx = dy / r2
    dalpha_dy = -dx / r2
    dalpha_dtheta = -1.0

    J = [
        [dxrcs_dx, dxrcs_dy, dxrcs_dtheta],
        [dyrcs_dx, dyrcs_dy, dyrcs_dtheta],
        [dalpha_dx, dalpha_dy, dalpha_dtheta],
    ]
    return J


def JT_P_eps_single(J, eps, precision=None):
    # Input:
    #   J: Jacobian matrix of the sensory mapping for one landmark
    #   eps: sensory prediction error vector [ex, ey, ea]
    #   precision: scalar precision weight for this landmark contribution
    # Output:
    #   contribution to the belief update [gx, gy, gt]
    # Function:
    #   computes J^T P eps for a single landmark
    #   converts one landmark's weighted sensory prediction error into an
    #   update contribution for robot x, y, and theta
    if precision is None:
        precision = PI_CAM

    we0 = precision * eps[0]
    we1 = precision * eps[1]
    we2 = precision * eps[2]
    gx = J[0][0] * we0 + J[1][0] * we1 + J[2][0] * we2
    gy = J[0][1] * we0 + J[1][1] * we1 + J[2][1] * we2
    gt = J[0][2] * we0 + J[1][2] * we1 + J[2][2] * we2
    return [gx, gy, gt]


def landmarks_fcs_2_rcs(robot_fcs_gc, landmarks_fcs):
    # Input:
    #   robot state in generalized coordinates
    #   landmark positions in FCS: {id: (x_cm, y_cm)}
    # Output:
    #   landmark predictions in robot coordinates (RCS): id, type, x_rcs_cm, y_rcs_cm, angle_rad
    # Function:
    #   FCS -> RCS translation
    #   converts field landmarks into expected sensory observations for the robot
    obs = []
    for name, (lx, ly) in landmarks_fcs.items():
        x_rcs, y_rcs, angle_rad = landmark_fcs_2_rcs(robot_fcs_gc, lx, ly)
        obs.append({
            "id": name,
            "type": landmark_type(name),
            "x_rcs_cm": x_rcs,
            "y_rcs_cm": y_rcs,
            "angle_rad": angle_rad,
        })
    return obs


def landmark_is_within_viewing_angle(robot_fcs_gc, landmark_x_cm, landmark_y_cm, viewing_half_angle_rad):
    # Input:
    #   robot_fcs_gc: robot state in generalized coordinates
    #   landmark_x_cm, landmark_y_cm: landmark position in FCS
    #   viewing_half_angle_rad: half-width of the viewing cone around the robot forward axis
    # Output:
    #   True if the landmark lies inside the viewing cone, else False
    # Function:
    #   derives the landmark bearing in robot coordinates
    #   checks whether the landmark is visible within the configured camera field of view
    _, _, angle_rad = landmark_fcs_2_rcs(robot_fcs_gc, landmark_x_cm, landmark_y_cm)
    if viewing_half_angle_rad >= math.pi:
        return True
    return abs(angle_rad) <= viewing_half_angle_rad


def landmark_prediction_error(y_obs, yhat):
    # Input:
    #   y_obs: prepared typed landmark observation in RCS with x_rcs_cm, y_rcs_cm, and calculated angle_rad
    #   yhat: predicted candidate landmark observation in RCS
    # Output:
    #   prediction error terms and position-only gate distance
    # Function:
    #   computes the sensory prediction error for one candidate association
    ex = y_obs["x_rcs_cm"] - yhat["x_rcs_cm"]
    ey = y_obs["y_rcs_cm"] - yhat["y_rcs_cm"]
    ea = wrap_rad(y_obs["angle_rad"] - yhat["angle_rad"])
    # Angle_weight can be used to normalize the angle error to the same scale as the position errors.
    eps = [ex, ey, ANGLE_WEIGHT * ea]
    position_error_cm = math.hypot(ex, ey)
    error_score = math.sqrt(eps[0] * eps[0] + eps[1] * eps[1] + eps[2] * eps[2])
    return eps, position_error_cm, error_score


def calculate_angle(y_obs_map):
    # Input:
    #   y_obs_map: observed typed landmark map in RCS with x_rcs_cm and y_rcs_cm
    # Output:
    #   copied observation map with angle_rad calculated from x/y for every observation
    # Function:
    #   prepares landmark observations for the core x/y/angle update vector
    #   automatically calculates angle_rad = atan2(y_rcs_cm, x_rcs_cm)
    #   treats x_rcs_cm and y_rcs_cm as the source of truth for bearing
    prepared = {}
    for obs_id, y_obs in y_obs_map.items():
        x_rcs_cm = float(y_obs["x_rcs_cm"])
        y_rcs_cm = float(y_obs["y_rcs_cm"])
        prepared_obs = dict(y_obs)
        prepared_obs["x_rcs_cm"] = x_rcs_cm
        prepared_obs["y_rcs_cm"] = y_rcs_cm
        prepared_obs["angle_rad"] = math.atan2(y_rcs_cm, x_rcs_cm)
        prepared[obs_id] = prepared_obs
    return prepared


def associate_landmark_observation(
    y_obs,
    yhat_map,
    association_name=ACTIVE_DATA_ASSOCIATION_NAME,
    used_candidate_ids=None,
):
    # Input:
    #   y_obs: prepared typed landmark observation in RCS with calculated angle_rad
    #   optionally carrying simulator/debug "landmark_id"
    #   yhat_map: predicted candidate landmark observations in RCS, keyed by internal landmark id
    #   association_name: selected data-association strategy
    #   used_candidate_ids: candidates already consumed by this inference step
    # Output:
    #   association record with chosen candidate id, prediction errors, and gate status
    # Function:
    #   resolves one typed sensory observation to an internal landmark candidate
    #   using the selected ambiguity/data-association strategy
    used_candidate_ids = used_candidate_ids or set()
    if association_name == DATA_ASSOCIATION_UNIQUE_ID_NAME:
        candidate_id = y_obs.get("landmark_id")
        if candidate_id is None or candidate_id in used_candidate_ids or candidate_id not in yhat_map:
            return {
                "accepted": False,
                "candidate_id": None,
                "eps": [0.0, 0.0, 0.0],
                "position_error_cm": None,
                "error_score": None,
                "association_name": association_name,
            }
        eps, position_error_cm, error_score = landmark_prediction_error(y_obs, yhat_map[candidate_id])
        return {
            "accepted": True,
            "candidate_id": candidate_id,
            "eps": eps,
            "position_error_cm": position_error_cm,
            "error_score": error_score,
            "association_name": association_name,
        }

    unique_candidate_id = unique_landmark_candidate_id_for_type(y_obs["type"])
    if unique_candidate_id is not None and unique_candidate_id in yhat_map:
        eps, position_error_cm, error_score = landmark_prediction_error(y_obs, yhat_map[unique_candidate_id])
        return {
            "accepted": True,
            "candidate_id": unique_candidate_id,
            "eps": eps,
            "position_error_cm": position_error_cm,
            "error_score": error_score,
            "association_name": DATA_ASSOCIATION_UNIQUE_DIRECT_NAME,
        }

    if association_name != DATA_ASSOCIATION_GATED_NEAREST_NAME:
        raise ValueError(f"Unknown landmark association algorithm: {association_name}")

    candidates = [
        candidate_id
        for candidate_id in landmark_candidate_ids_for_type(y_obs["type"])
        if candidate_id not in used_candidate_ids
        and candidate_id in yhat_map
    ]

    best = None
    for candidate_id in candidates:
        eps, position_error_cm, error_score = landmark_prediction_error(y_obs, yhat_map[candidate_id])
        candidate = {
            "accepted": position_error_cm <= LANDMARK_ASSOCIATION_GATE_CM,
            "candidate_id": candidate_id,
            "eps": eps,
            "position_error_cm": position_error_cm,
            "error_score": error_score,
            "association_name": association_name,
        }
        if best is None or candidate["error_score"] < best["error_score"]:
            best = candidate

    if best is None:
        return {
            "accepted": False,
            "candidate_id": None,
            "eps": [0.0, 0.0, 0.0],
            "position_error_cm": None,
            "error_score": None,
            "association_name": association_name,
        }
    return best


def landmark_inference_step(belief_fcs_gc, y_obs_map, step_dt_s=DT):
    # Input:
    #   belief_fcs_gc: belief robot state in FCS generalized coordinates
    #   y_obs_map: prepared typed landmark map in RCS with x_rcs_cm, y_rcs_cm, and calculated angle_rad
    #   step_dt_s: timestep for this landmark inference update
    # Output:
    #   Output for reporting and visualisation / diagnostics
    #   yhat_map: predicted landmark map in RCS from the belief robot state
    #   F: free energy = precision-weighted sum of squared sensory prediction errors
    # Function:
    #   performs one active-inference landmark update step and updates the belief_fcs_gc state
    #   expects landmark observations to already be prepared with calculate_angle(...)
    #   returns yhat_map and free_energy as diagnostic/reporting outputs
    #   No generalized coordinates, only top row / positions (landmarks don’t move)

    belief_x_cm, belief_y_cm, belief_theta_rad = gc_2_scalar(belief_fcs_gc)

    # Landmark predictions given current belief robot position
    yhat_map = obs_list_to_map(landmarks_fcs_2_rcs(belief_fcs_gc, LANDMARKS))

    # Calculate mu_dot for x,y and angle
    mu_dot = [0.0, 0.0, 0.0]
    free_energy = 0.0
    used_candidate_ids = set()
    for _, y_obs in y_obs_map.items():  # for each observed typed place cell
        # Associate the typed observation with the best internal landmark candidate.
        association = associate_landmark_observation(
            y_obs,
            yhat_map,
            ACTIVE_DATA_ASSOCIATION_NAME,
            used_candidate_ids,
        )
        if not association["accepted"]:
            continue
        lid = association["candidate_id"]
        used_candidate_ids.add(lid)

        # calculate the prediction errors (x,y,angle)
        # = sensory observation - predicted associated candidate observation
        eps = association["eps"]
        # calculate the jacobian
        landmark_x, landmark_y = LANDMARKS[lid]
        J = landmark_jacobian_fcs_2_rcs(belief_fcs_gc, landmark_x, landmark_y)
        # calculate the precision distance modifier
        precision = PI_CAM * precision_distance_modifier(yhat_map[lid])
        # calculate J^T*P*eps for a single landmark
        g = JT_P_eps_single(J, eps, precision)
        mu_dot[0] += LR_XY * g[0]
        mu_dot[1] += LR_XY * g[1]
        mu_dot[2] += LR_ANGLE * g[2]
        # Calculate the Free energy contribution
        free_energy += precision * (eps[0] * eps[0] + eps[1] * eps[1] + eps[2] * eps[2])

    # Euler forward update rule mu(t+dt) = mu(t) + dt * mu_dot
    scalar_2_gc(
        belief_fcs_gc,
        x_cm=belief_x_cm + step_dt_s * mu_dot[0],
        y_cm=belief_y_cm + step_dt_s * mu_dot[1],
        theta_rad=belief_theta_rad + step_dt_s * mu_dot[2],
    )
    return yhat_map, free_energy


def algorithm_uses_motion(algorithm_name):
    return algorithm_name in {
        MOTION_ONLY_ALGORITHM_NAME,
        FUSED_MOTION_LANDMARK_ALGORITHM_NAME,
    }


def algorithm_uses_landmarks(algorithm_name):
    return algorithm_name in {
        LANDMARK_ONLY_ALGORITHM_NAME,
        FUSED_MOTION_LANDMARK_ALGORITHM_NAME,
    }


def step_core(request):
    # Input:
    #   request: pybind11/JSON-friendly dict with:
    #     belief_fcs_gc: generalized-coordinate belief state in FCS
    #     landmark_observations_rcs: observed landmarks in RCS as type + x_rcs_cm + y_rcs_cm
    #       angle_rad is not required from C++; it is calculated from x/y before landmark inference
    #     motion_observation: observed motion terms in RCS
    #     algorithm_name: optional inference mode
    #     step_dt_s: required inference timestep, preferably computed by C++ from frame timestamps
    #     timestamp: optional ISO 8601 / RFC 3339 observation-frame timestamp for traceability/logging
    # Output:
    #   dict containing the next belief_fcs_gc and diagnostics
    # Function:
    #   public headless core API for C++/pybind11 embedding
    #   shields the core from external mutable/request-shaped objects
    #   performs one inference step, and returns the
    #   updated generalized-coordinate belief state for the next caller iteration
    #   timestamp is not used to calculate dt here; C++ owns timestamp history
    #   and passes the already computed step_dt_s into this stateless core call
    def shield_belief_fcs_gc(belief_fcs_gc):
        shielded = {}
        for axis in ("x", "y", "theta"):
            if axis not in belief_fcs_gc:
                raise ValueError(f"belief_fcs_gc is missing axis '{axis}'")
            values = list(belief_fcs_gc[axis])
            if len(values) < 3:
                raise ValueError(
                    f"belief_fcs_gc axis '{axis}' must contain at least 3 values: "
                    "position, velocity, acceleration"
                )
            shielded[axis] = [float(value) for value in values]
        shielded["theta"][0] = wrap_rad(shielded["theta"][0])
        return shielded

    def shield_landmark_observations_rcs(landmark_observations_rcs):
        shielded = []
        for idx, obs in enumerate(landmark_observations_rcs or []):
            for key in ("type", "x_rcs_cm", "y_rcs_cm"):
                if key not in obs:
                    raise ValueError(f"landmark_observations_rcs[{idx}] is missing '{key}'")
            shielded.append({
                "obs_id": str(obs.get("obs_id", f"obs_{idx}")),
                "type": str(obs["type"]),
                "x_rcs_cm": float(obs["x_rcs_cm"]),
                "y_rcs_cm": float(obs["y_rcs_cm"]),
            })
        return shielded

    def shield_motion_observation(motion_observation):
        required_keys = (
            "x_vel_cm_s",
            "y_vel_cm_s",
            "theta_vel_rad_s",
            "x_acc_cm_s2",
            "y_acc_cm_s2",
            "theta_acc_rad_s2",
        )
        if motion_observation is None:
            return None
        shielded = {}
        for key in required_keys:
            if key not in motion_observation:
                raise ValueError(f"motion_observation is missing '{key}'")
            shielded[key] = float(motion_observation[key])
        return shielded

    def shield_algorithm_name(algorithm_name):
        if algorithm_name is None:
            return FUSED_MOTION_LANDMARK_ALGORITHM_NAME
        algorithm_name = str(algorithm_name)
        valid_algorithm_names = {
            LANDMARK_ONLY_ALGORITHM_NAME,
            MOTION_ONLY_ALGORITHM_NAME,
            FUSED_MOTION_LANDMARK_ALGORITHM_NAME,
        }
        if algorithm_name not in valid_algorithm_names:
            valid_names = ", ".join(sorted(valid_algorithm_names))
            raise ValueError(
                f"Unknown algorithm_name '{algorithm_name}'. "
                f"Use one of: {valid_names}"
            )
        return algorithm_name

    if request is None:
        raise ValueError("step_core request is required")
    if "belief_fcs_gc" not in request:
        raise ValueError("step_core request requires 'belief_fcs_gc'")
    if "step_dt_s" not in request:
        raise ValueError("step_core request requires 'step_dt_s'")

    belief_fcs_gc = shield_belief_fcs_gc(request["belief_fcs_gc"])
    landmark_observations_rcs = shield_landmark_observations_rcs(request.get("landmark_observations_rcs", []))
    algorithm_name = shield_algorithm_name(request.get("algorithm_name"))
    motion_observation = shield_motion_observation(request.get("motion_observation"))
    step_dt_s = float(request["step_dt_s"])
    timestamp = None if request.get("timestamp") is None else str(request["timestamp"])

    next_belief_fcs_gc, diagnostics = inference_step_from_observations(
        belief_fcs_gc,
        landmark_observations_rcs=landmark_observations_rcs,
        motion_observation=motion_observation,
        algorithm_name=algorithm_name,
        step_dt_s=step_dt_s,
    )
    diagnostics["timestamp"] = timestamp
    diagnostics["step_dt_s"] = step_dt_s
    return {
        "belief_fcs_gc": shield_belief_fcs_gc(next_belief_fcs_gc),
        "diagnostics": diagnostics,
    }


def inference_step_from_observations(
    belief_fcs_gc,
    landmark_observations_rcs=None,
    motion_observation=None,
    algorithm_name=ACTIVE_ALGORITHM_NAME,
    step_dt_s=DT,
):
    # Input:
    #   belief_fcs_gc: belief robot state in FCS generalized coordinates
    #   landmark_observations_rcs: observed landmarks in RCS from the caller/sensor adapter
    #   each landmark observation supplies x_rcs_cm and y_rcs_cm; angle_rad is calculated from x/y
    #   motion_observation: observed velocity/acceleration terms from odometry/IMU in RCS
    #   algorithm_name: selects the active inference mode
    #   step_dt_s: timestep for this inference update
    # Output:
    #   updated belief_fcs_gc in generalized-coordinate form and diagnostics
    # Function:
    #   performs one headless active-inference update using caller-supplied sensor data
    #   prepares landmark observations by calculating angle_rad from x_rcs_cm and y_rcs_cm
    #   this is the core boundary intended for real robot / external software integration
    free_energy = 0.0
    y_obs_map = calculate_angle(obs_list_to_map(landmark_observations_rcs or []))
    yhat_map = obs_list_to_map(landmarks_fcs_2_rcs(belief_fcs_gc, LANDMARKS))
    motion_terms = None
    motion_free_energy = 0.0
    landmark_free_energy = 0.0

    # Motion prediction errors and generalized-coordinate update.
    if algorithm_uses_motion(algorithm_name):
        if motion_observation is None:
            raise ValueError(f"{algorithm_name} requires a motion_observation")
        motion_terms, motion_free_energy = motion_inference_step(
            belief_fcs_gc,
            motion_observation,
            step_dt_s=step_dt_s,
        )
        free_energy += motion_free_energy

    # Landmark prediction errors, Jacobian/update-term construction, and Euler update.
    if algorithm_uses_landmarks(algorithm_name):
        yhat_map, landmark_free_energy = landmark_inference_step(
            belief_fcs_gc,
            y_obs_map,
            step_dt_s=step_dt_s,
        )
        free_energy += landmark_free_energy

    diagnostics = {
        "free_energy": free_energy,
        "motion_free_energy": motion_free_energy,
        "landmark_free_energy": landmark_free_energy,
        "motion_terms": motion_terms,
        "y_obs": y_obs_map,
        "y_hat": yhat_map,
    }
    return belief_fcs_gc, diagnostics


# =========================
# SIMULATION EXPERIMENT
# =========================
# POC simulation layer on top of CORE.
# Builds simulated sensor observations and runs repeated simulated inference steps.

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


def simulation_sense_landmarks(robot_fcs_gc, landmarks_fcs):
    # Input:
    #   robot state in generalized coordinates
    #   landmark positions in FCS: {id: (x_cm, y_cm)}
    # Output:
    #   simulated landmark observations in robot coordinates (RCS): obs_id, type, landmark_id, x_rcs_cm, y_rcs_cm
    # Function:
    #   builds simulated sensory observations
    #   emits x/y landmark observations only; angle_rad is calculated by the core preparation step
    #   adds landmark observation noise when NOISE_ENABLED is active
    #   optionally drops individual readings
    #   optionally hides landmarks for short obstruction windows
    #   only visible landmarks if viewing angle is enabled
    obs = []
    for landmark_obs in landmarks_fcs_2_rcs(robot_fcs_gc, landmarks_fcs):
        landmark_x_cm, landmark_y_cm = landmarks_fcs[landmark_obs["id"]]
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

        obs.append({
            "obs_id": f"obs_{len(obs)}_{landmark_obs['type']}",
            "type": landmark_obs["type"],
            "landmark_id": landmark_obs["id"],
            "x_rcs_cm": x_rcs_cm,
            "y_rcs_cm": y_rcs_cm,
        })
    return obs


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
    y_obs_list = simulation_sense_landmarks(robots["actual"], active_simulation_landmarks())
    y_motion_obs = simulation_sense_motion(robots["actual"])
    _, diagnostics = inference_step_from_observations(
        robots["belief"],
        landmark_observations_rcs=y_obs_list,
        motion_observation=y_motion_obs,
        algorithm_name=algorithm_name,
        step_dt_s=DT,
    )
    return diagnostics["y_obs"], diagnostics["y_hat"], y_motion_obs, diagnostics["free_energy"]


# --- Simulated experiment runner ---
def run_experiment(
    robots,
    algorithm_name,
    max_steps,
    until_convergence=False,
    step_callback=None,
    callback_interval=20,
):
    # Input:
    #   robots: dict with actual and belief robot states
    #   algorithm_name: selects the active inference mode
    #   max_steps: maximum number of inference steps to run
    #   until_convergence: whether to stop early once the belief converges to the actual pose
    #   step_callback: optional callback invoked during the run for UI refresh / interruption handling
    #   callback_interval: callback cadence in steps
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

    return history




# =========================
# REPORT
# =========================
def robot_generalized_coordinate_view(robot_fcs_gc):
    return robot_gc_view(robot_fcs_gc)


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
        "data_association": ACTIVE_DATA_ASSOCIATION_NAME,
        "landmark_association_gate_cm": LANDMARK_ASSOCIATION_GATE_CM,
        "landmark_scenario_profile": ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME,
        "landmark_scenario_count": len(active_landmark_scenario_profile()["landmark_ids"]),
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
        f"Association={settings.get('data_association', ACTIVE_DATA_ASSOCIATION_NAME)}"
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
        actual_gc = record.get("actual_gc")
        if motion_obs is None or actual_gc is None:
            return {
                "x_vel": 0.0,
                "y_vel": 0.0,
                "theta_vel": 0.0,
                "x_acc": 0.0,
                "y_acc": 0.0,
                "theta_acc": 0.0,
            }
        obs_x_vel, obs_y_vel, obs_theta_vel = vel_rcs_2_fcs(
            actual_gc,
            motion_obs["x_vel_cm_s"],
            motion_obs["y_vel_cm_s"],
            motion_obs["theta_vel_rad_s"],
        )
        obs_x_acc, obs_y_acc, obs_theta_acc = acc_rcs_2_fcs(
            actual_gc,
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

    observed_x = [h["observed_pose"]["x_cm"] for h in history]
    observed_y = [h["observed_pose"]["y_cm"] for h in history]
    observed_theta = [h["observed_pose"]["theta_rad"] for h in history]
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
        f"Association={settings.get('data_association', ACTIVE_DATA_ASSOCIATION_NAME)} "
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
    if settings["noise_enabled"]:
        ax1.plot(steps, observed_x, label="Sensory observation x")
    ax1.plot(steps, belief_x, label="Belief x")
    ax1.plot(steps, actual_x, label="Actual x")
    ax1.set_ylabel("x forward (cm)")
    ax1.legend(loc="center right")

    ax2 = fig_pos.add_subplot(gs_pos[pos_row_offset + 1, 0], sharex=ax1)
    if settings["noise_enabled"]:
        ax2.plot(steps, observed_y, label="Sensory observation y")
    ax2.plot(steps, belief_y, label="Belief y")
    ax2.plot(steps, actual_y, label="Actual y")
    ax2.set_ylabel("y left (cm)")
    ax2.legend(loc="center right")

    ax3 = fig_pos.add_subplot(gs_pos[pos_row_offset + 2, 0], sharex=ax1)
    if settings["noise_enabled"]:
        ax3.plot(steps, observed_theta, label="Sensory observation theta")
    ax3.plot(steps, belief_theta, label="Belief theta")
    ax3.plot(steps, actual_theta, label="Actual theta")
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
        ax5.plot(steps, observed_x_vel, label="Sensory observation x dot")
    ax5.plot(steps, belief_x_vel, label="Belief x dot")
    ax5.plot(steps, actual_x_vel, label="Actual x dot")
    ax5.set_ylabel("x vel (cm/s)")
    ax5.legend(loc="center right")

    ax6 = fig_vel.add_subplot(gs_vel[1, 0], sharex=ax5)
    if settings["noise_enabled"]:
        ax6.plot(steps, observed_y_vel, label="Sensory observation y dot")
    ax6.plot(steps, belief_y_vel, label="Belief y dot")
    ax6.plot(steps, actual_y_vel, label="Actual y dot")
    ax6.set_ylabel("y vel (cm/s)")
    ax6.legend(loc="center right")

    ax7 = fig_vel.add_subplot(gs_vel[2, 0], sharex=ax5)
    if settings["noise_enabled"]:
        ax7.plot(steps, observed_theta_vel, label="Sensory observation theta dot")
    ax7.plot(steps, belief_theta_vel, label="Belief theta dot")
    ax7.plot(steps, actual_theta_vel, label="Actual theta dot")
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
        ax8.plot(steps, observed_x_acc, label="Sensory observation x ddot")
    ax8.plot(steps, belief_x_acc, label="Belief x ddot")
    ax8.plot(steps, actual_x_acc, label="Actual x ddot")
    ax8.set_ylabel("x acc (cm/s^2)")
    ax8.legend(loc="center right")

    ax9 = fig_acc.add_subplot(gs_acc[1, 0], sharex=ax8)
    if settings["noise_enabled"]:
        ax9.plot(steps, observed_y_acc, label="Sensory observation y ddot")
    ax9.plot(steps, belief_y_acc, label="Belief y ddot")
    ax9.plot(steps, actual_y_acc, label="Actual y ddot")
    ax9.set_ylabel("y acc (cm/s^2)")
    ax9.legend(loc="center right")

    ax10 = fig_acc.add_subplot(gs_acc[2, 0], sharex=ax8)
    if settings["noise_enabled"]:
        ax10.plot(steps, observed_theta_acc, label="Sensory observation theta ddot")
    ax10.plot(steps, belief_theta_acc, label="Belief theta ddot")
    ax10.plot(steps, actual_theta_acc, label="Actual theta ddot")
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


# =========================
# CONVERGENCE TESTS
# Standalone block on top of the core experiment runner.
# Uses the core `run_experiment(...)` in background mode without UI updates.
# =========================
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


def random_nearby_belief_pose(actual_pose, max_offset_cm=70.0, max_angle_offset_deg=20.0, rng=None):
    rng = random if rng is None else rng
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2

    dx = rng.uniform(-max_offset_cm, max_offset_cm)
    dy = rng.uniform(-max_offset_cm, max_offset_cm)
    x = min(halfL, max(-halfL, actual_pose["x"] + dx))
    y = min(halfW, max(-halfW, actual_pose["y"] + dy))
    dtheta = math.radians(rng.uniform(-max_angle_offset_deg, max_angle_offset_deg))
    theta = wrap_rad(actual_pose["theta"] + dtheta)
    return {"x": x, "y": y, "theta": theta}


def is_pose_outside_field(pose):
    halfL = FIELD_LENGTH_CM / 2
    halfW = FIELD_WIDTH_CM / 2
    return (
        pose["x_cm"] < -halfL or pose["x_cm"] > halfL or
        pose["y_cm"] < -halfW or pose["y_cm"] > halfW
    )


def make_converge_start_pair(start_mode, rng=None):
    rng = random if rng is None else rng
    if start_mode == "nearby":
        actual_start = random_pose_inside_field(rng)
        belief_start = random_nearby_belief_pose(actual_start, max_offset_cm=50.0, rng=rng)
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


def history_has_sequential_convergence(history):
    convergence_hits = 0
    for record in history:
        if pose_converged_ros(
            record["actual_pose"],
            record["belief_pose"],
            CONV_TOL_X_CM,
            CONV_TOL_Y_CM,
            CONV_TOL_ANGLE_RAD,
        ):
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


def sampled_history_for_report(history, max_points=240):
    if len(history) <= max_points:
        return list(history)
    stride = max(1, math.ceil(len(history) / max_points))
    sampled = history[::stride]
    if sampled[-1] is not history[-1]:
        sampled.append(history[-1])
    return sampled


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
):
    converged = 0
    non_converged = 0
    failures = []
    last_history = []
    trial_steps = []
    converged_steps = []
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
            actual_start, belief_start = make_converge_start_pair(start_mode)
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
            step_callback=(lambda: False if stop_check is not None and stop_check() else None) if stop_check is not None else None,
            callback_interval=25,
        )
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
            non_converged += 1
            if collect_failures:
                actual_end = robot_pose_view(robots["actual"])
                belief_end = robot_pose_view(robots["belief"])
                visible_landmark_ids = sorted(obs["obs_id"] for obs in simulation_sense_landmarks(robots["actual"], active_simulation_landmarks()))
                failures.append({
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
            keep_running = progress_callback(i + 1, runs, converged, non_converged)
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
    non_converged = report["non_converged"]
    mean_steps = report.get("mean_steps_to_converge")
    sd_steps = report.get("sd_steps_to_converge")
    max_steps = report.get("max_steps_to_converge")
    p95_steps = report.get("p95_steps_to_converge")
    p99_steps = report.get("p99_steps_to_converge")
    failures = report["failures"]
    last_history = report.get("last_history", [])
    if last_history:
        settings = last_history[0].get("settings", experiment_settings(algorithm_name))
    else:
        settings = experiment_settings(algorithm_name)

    failures_per_page = 3
    total_failures = len(failures)
    total_pages = max(1, math.ceil(total_failures / failures_per_page))
    timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    start_mode = report.get("start_mode", "random")
    if start_mode in {"dynamic_demo_random", "dynamic_demo_nearby"}:
        mode_label = "Dynamic Demo Converge Test"
        demo_path_name = report.get("demo_path_name", "Unknown")
        if start_mode == "dynamic_demo_nearby":
            start_condition_text = f"Start condition: moving actual robot on demo path '{demo_path_name}'; belief dx,dy uniform +/-50 cm; dtheta uniform +/-20 deg"
        else:
            start_condition_text = f"Start condition: moving actual robot on demo path '{demo_path_name}'; actual and belief independently random near field"
    elif start_mode == "nearby":
        mode_label = "Landmark Converge Nearby"
        start_condition_text = "Start condition: landmark-only static actual pose; belief dx,dy uniform +/-50 cm; dtheta uniform +/-20 deg"
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

    if settings["noise_enabled"]:
        noise_text = (
            f"Noise: YES   landmarks x={settings['noise_x_cm']:.2f} cm   "
            f"y={settings['noise_y_cm']:.2f} cm   angle={settings['noise_angle_rad']:.4f} rad   "
            f"motion={settings.get('motion_noise_ratio', MOTION_NOISE_RATIO) * 100.0:.1f}%"
        )
    else:
        noise_text = "Noise: NO"

    if mean_steps is None or sd_steps is None or max_steps is None or p95_steps is None or p99_steps is None:
        step_stats_text = "Steps to converge: n/a"
    else:
        step_stats_text = f"Steps to converge: mean={mean_steps:.1f}   sd={sd_steps:.1f}   p95={p95_steps}   p99={p99_steps}   max={max_steps}   n={converged}"

    halfW = FIELD_WIDTH_CM / 2
    halfL = FIELD_LENGTH_CM / 2
    (
        odometry_bias_text,
        slip_text,
        dropout_text,
        obstruction_text,
        viewing_text,
        association_text,
        landmark_profile_text,
    ) = simulation_settings_report_lines(settings)

    def draw_field_snapshot(ax, actual_pose, belief_pose, title, history=None):
        ax.set_title(title, fontsize=12)
        ax.set_xlim(halfW + 80, -halfW - 80)
        ax.set_ylim(-halfL - 80, halfL + 80)
        ax.set_aspect("equal", adjustable="box")

        rect = Rectangle((halfW, -halfL), -FIELD_WIDTH_CM, FIELD_LENGTH_CM, fill=False, linewidth=1.8)
        ax.add_patch(rect)
        ax.plot([halfW, -halfW], [0, 0], linewidth=1.4)

        if history:
            actual_y = [record["actual_pose"]["y_cm"] for record in history]
            actual_x = [record["actual_pose"]["x_cm"] for record in history]
            belief_y = [record["belief_pose"]["y_cm"] for record in history]
            belief_x = [record["belief_pose"]["x_cm"] for record in history]
            ax.plot(actual_y, actual_x, color="green", linewidth=1.2, alpha=0.75)
            ax.plot(belief_y, belief_x, color="red", linewidth=1.2, alpha=0.75)

        ax.scatter([actual_pose["y_cm"]], [actual_pose["x_cm"]], marker="x", s=110, color="green", linewidths=2.0)
        ax.scatter([belief_pose["y_cm"]], [belief_pose["x_cm"]], marker="o", s=120, facecolors="none", edgecolors="red", linewidths=2.0)

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
                linewidth=1.6,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    with PdfPages(pdf_path) as pdf:
        for page_idx in range(total_pages):
            page_failures = failures[page_idx * failures_per_page:(page_idx + 1) * failures_per_page]
            n_rows = max(1, len(page_failures))
            is_first_page = page_idx == 0
            header_height = 2.9 if is_first_page else 1.2

            fig = plt.figure(figsize=(16, 3.2 + header_height + 6.8 * n_rows))
            gs = GridSpec(
                n_rows + 1,
                3,
                figure=fig,
                height_ratios=[header_height] + [5.4] * n_rows,
                width_ratios=[2.5, 2.5, 3.2],
            )
            fig.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.04, hspace=0.32, wspace=0.18)

            ax0 = fig.add_subplot(gs[0, :])
            ax0.axis("off")
            if is_first_page:
                header = (
                    f"{mode_label}   |   Time: {timestamp}   |   Algorithm: {algorithm_name}   |   Page {page_idx + 1}/{total_pages}\n"
                    f"Converged: {converged}    Non-converged: {non_converged}\n"
                    f"{step_stats_text}\n"
                    f"{start_condition_text}\n"
                    f"{noise_text}\n"
                    f"{odometry_bias_text}\n"
                    f"{slip_text}\n"
                    f"{dropout_text}   |   {obstruction_text}\n"
                    f"{viewing_text}   |   {association_text}   |   {landmark_profile_text}\n"
                    f"DT={settings['dt']:.4f}\n"
                    f"{precision_report_row(settings)}\n"
                    f"{precision_distance_modifier_report_row(settings)}\n"
                    f"{learning_rate_report_row(settings)}\n"
                    f"Convergence condition: |dx|<={CONV_TOL_X_CM:.1f} cm   |dy|<={CONV_TOL_Y_CM:.1f} cm   |dtheta|<={CONV_TOL_ANGLE_RAD:.4f} rad   sequential hits>={CONV_REQUIRED_SEQUENTIAL_HITS}"
                )
            else:
                header = (
                    f"{mode_label}   |   Algorithm: {algorithm_name}   |   Page {page_idx + 1}/{total_pages}\n"
                    f"Converged: {converged}    Non-converged: {non_converged}   |   {step_stats_text}"
                )
            header = "\n".join(
                textwrap.fill(
                    line,
                    width=120,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                for line in header.splitlines()
            )
            ax0.text(0.01, 0.98, header, fontsize=12, family="monospace", va="top")

            if not page_failures:
                ax_no = fig.add_subplot(gs[1, :])
                ax_no.axis("off")
                ax_no.text(0.01, 0.98, "No failed cases.", fontsize=12, family="monospace", va="top")
            else:
                for row_idx, fail in enumerate(page_failures, start=1):
                    fail_number = page_idx * failures_per_page + row_idx
                    a0 = fail["actual_start"]
                    b0 = fail["belief_start"]
                    a1 = fail["actual_end"]
                    b1 = fail["belief_end"]
                    ax_start = fig.add_subplot(gs[row_idx, 0])
                    draw_field_snapshot(ax_start, a0, b0, f"Fail {fail_number} START")

                    ax_end = fig.add_subplot(gs[row_idx, 1])
                    draw_field_snapshot(ax_end, a1, b1, f"Fail {fail_number} END", history=fail.get("path_history"))

                    ax_txt = fig.add_subplot(gs[row_idx, 2])
                    ax_txt.axis("off")
                    txt = (
                        f"Fail {fail_number}\n"
                        f"{format_ros_pose_line('Start Actual', a0, decimals=0, angle_decimals=3)}"
                        f"   outside={fail['actual_start_outside']}\n"
                        f"{format_ros_pose_line('Start Belief', b0, decimals=0, angle_decimals=3)}"
                        f"   outside={fail['belief_start_outside']}\n"
                        f"Visible landmarks at fail end: {fail['visible_landmark_count']}\n"
                        f"{format_ros_pose_line('End   Actual', a1, decimals=0, angle_decimals=3)}\n"
                        f"{format_ros_pose_line('End   Belief', b1, decimals=0, angle_decimals=3)}"
                    )
                    ax_txt.text(0.01, 0.98, txt, fontsize=12, family="monospace", va="top")

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


def landmark_analysis_data(actual_fcs_gc, belief_fcs_gc):
    y_obs_map = calculate_angle(obs_list_to_map(simulation_sense_landmarks(actual_fcs_gc, active_simulation_landmarks())))
    yhat_map = obs_list_to_map(landmarks_fcs_2_rcs(belief_fcs_gc, LANDMARKS))
    observed_landmark_ids = list(y_obs_map.keys())

    contributions = {}
    associations = {}
    used_candidate_ids = set()
    for obs_id in observed_landmark_ids:
        association = associate_landmark_observation(
            y_obs_map[obs_id],
            yhat_map,
            ACTIVE_DATA_ASSOCIATION_NAME,
            used_candidate_ids,
        )
        associations[obs_id] = association
        if not association["accepted"]:
            continue
        lid = association["candidate_id"]
        used_candidate_ids.add(lid)
        eps = association["eps"]
        landmark_x, landmark_y = LANDMARKS[lid]
        J = landmark_jacobian_fcs_2_rcs(belief_fcs_gc, landmark_x, landmark_y)
        precision = PI_CAM * precision_distance_modifier(yhat_map[lid])
        contributions[obs_id] = JT_P_eps_single(J, eps, precision)

    details = []
    for obs_id in observed_landmark_ids:
        association = associations[obs_id]
        if not association["accepted"]:
            continue
        lid = association["candidate_id"]
        actual_landmark_id = y_obs_map[obs_id].get("landmark_id")
        association_mismatch = (
            association["association_name"] == DATA_ASSOCIATION_GATED_NEAREST_NAME
            and actual_landmark_id is not None
            and actual_landmark_id != lid
        )
        ex = association["eps"][0]
        ey = association["eps"][1]
        ea = association["eps"][2] / ANGLE_WEIGHT if ANGLE_WEIGHT != 0.0 else 0.0
        obs_fcs = rcs_to_fcs(belief_fcs_gc, y_obs_map[obs_id])
        pred_fcs = rcs_to_fcs(belief_fcs_gc, yhat_map[lid])
        contrib = contributions[obs_id]
        contrib_mag = math.sqrt(contrib[0] * contrib[0] + contrib[1] * contrib[1] + contrib[2] * contrib[2])
        error_mag = math.sqrt(ex * ex + ey * ey + ea * ea)
        details.append({
            "id": lid,
            "obs_id": obs_id,
            "type": y_obs_map[obs_id]["type"],
            "landmark_id": actual_landmark_id,
            "actual_landmark_id": actual_landmark_id,
            "associated_landmark_id": lid,
            "association_mismatch": association_mismatch,
            "obs_fcs": obs_fcs,
            "pred_fcs": pred_fcs,
            "ex": ex,
            "ey": ey,
            "ea": ea,
            "precision": PI_CAM * precision_distance_modifier(yhat_map[lid]),
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
        "hidden_landmark_ids": [lid for lid in LANDMARKS.keys() if lid not in used_candidate_ids],
        "association_mismatch_count": sum(1 for d in details if d["association_mismatch"]),
    }


def motion_analysis_details(belief_fcs_gc, y_motion_obs):
    motion_terms = motion_prediction_error(belief_fcs_gc, y_motion_obs)
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

    if algorithm_name == MOTION_ONLY_ALGORITHM_NAME:
        details = motion_analysis_details(robots["belief"], y_motion_obs)
        return {
            "mode": "motion",
            "actual_pose": actual_pose,
            "belief_pose": belief_pose,
            "details": details,
        }

    landmark_data = landmark_analysis_data(robots["actual"], robots["belief"])

    if algorithm_name == FUSED_MOTION_LANDMARK_ALGORITHM_NAME:
        motion_details = motion_analysis_details(robots["belief"], y_motion_obs)
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
            current_x_cm = target_x_cm
            current_y_cm = target_y_cm
            path_state["x_cm"] = current_x_cm
            path_state["y_cm"] = current_y_cm
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
    global ACTIVE_DATA_ASSOCIATION_NAME, PRECISION_DISTANCE_MODIFIER_ENABLED
    PRECISION_DISTANCE_MODIFIER_ENABLED = bool(precision_distance_modifier_enabled)
    apply_masterdata_setting_profile(
        masterdata_setting_profile_by_name(profile_name, PRECISION_DISTANCE_MODIFIER_ENABLED)
    )
    if association_name is None:
        association_name = DATA_ASSOCIATION_UNIQUE_ID_NAME if apply_paper_stress else ACTIVE_DATA_ASSOCIATION_NAME
    if apply_paper_stress:
        apply_visibility_stress_paper_conditions(association_name)
    else:
        ACTIVE_DATA_ASSOCIATION_NAME = association_name
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
):
    converged = 0
    non_converged = 0
    failures = []
    last_history = []
    trial_steps = []
    converged_steps = []
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
            actual_start, belief_start = make_converge_start_pair(start_mode)
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
        if profile_name is not None:
            settings["profile_name"] = profile_name
        settings["demo_path_name"] = demo_path_name

        history = []
        convergence_hits = 0
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

        last_history = history
        steps_taken = len(history)
        trial_steps.append(steps_taken)

        if stopped:
            break

        if history and history_has_sequential_convergence(history):
            converged += 1
            converged_steps.append(steps_taken)
        else:
            non_converged += 1
            actual_end = robot_pose_view(robots["actual"])
            belief_end = robot_pose_view(robots["belief"])
            visible_landmark_ids = sorted(obs["obs_id"] for obs in simulation_sense_landmarks(robots["actual"], active_simulation_landmarks()))
            failures.append({
                "actual_start": start_actual,
                "belief_start": start_belief,
                "actual_end": actual_end,
                "belief_end": belief_end,
                "actual_start_outside": is_pose_outside_field(start_actual),
                "belief_start_outside": is_pose_outside_field(start_belief),
                "visible_landmark_count": len(visible_landmark_ids),
                "visible_landmark_ids": visible_landmark_ids,
                "path_history": sampled_history_for_report(history),
            })

        if progress_callback is not None:
            keep_running = progress_callback(i + 1, runs, converged, non_converged)
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
        "demo_path_name": demo_path_name,
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


# --- Main UI entrypoint ---
def main():
    global NOISE_ENABLED, ODOMETRY_BIAS_ENABLED, VIEWING_ANGLE_ENABLED, MOTION_SLIP_ENABLED, LANDMARK_DROPOUT_ENABLED, LANDMARK_OBSTRUCTION_ENABLED, PRECISION_DISTANCE_MODIFIER_ENABLED, ACTIVE_DATA_ASSOCIATION_NAME, ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME
    if getattr(pygame, "_is_stub", False) or getattr(plt, "_is_stub", False):
        raise ModuleNotFoundError("Interactive mode requires both pygame and matplotlib to be installed")
    pygame.init()
    pygame.key.start_text_input()
    info = pygame.display.Info()
    screen_w = info.current_w
    screen_h = info.current_h

    win_w = min(1380, int(screen_w * 0.95))
    win_h = min(880, int(screen_h * 0.90))

    panel_w = min(PANEL_W_PX, max(340, win_w // 4))
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
    data_association_names = DATA_ASSOCIATION_ALGORITHM_NAMES
    selected_data_association_idx = data_association_names.index(ACTIVE_DATA_ASSOCIATION_NAME)
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
    legend_h = legend_padding * 2 + legend_line_h * 2

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
    algo_btn = pygame.Rect(panel_left, row_y, panel_inner_w, btn_h)
    row_y += btn_h + row_gap
    data_assoc_btn = pygame.Rect(panel_left, row_y, panel_inner_w, btn_h)
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
    viewing_angle_input_rect = pygame.Rect(panel_left + viewing_toggle_w + viewing_gap, row_y, panel_inner_w - viewing_toggle_w - viewing_gap, btn_h)
    row_y += btn_h + row_gap
    compact_btn_gap = 6
    compact_btn_w = (panel_inner_w - 3 * compact_btn_gap) // 4
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
    paper_data_btn = pygame.Rect(convdynamic_near_btn.right + conv_btn_gap, row_y, panel_left + panel_inner_w - convdynamic_near_btn.right - conv_btn_gap, btn_h)
    row_y += btn_h + row_gap
    info_top = row_y + 8

    settings_w = 180
    settings_x = world_w - settings_w - 14
    settings_y = 14 + legend_h + 10
    settings_profile_btn = pygame.Rect(settings_x, settings_y, settings_w, btn_h)
    landmark_scenario_profile_btn = pygame.Rect(settings_x, settings_y + btn_h + row_gap, settings_w, btn_h)
    settings_fields_y = settings_y + 2 * (btn_h + row_gap)
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
            "Run 2 hours of demo steps headlessly and export compact visibility-stress CSV/JSON metadata",
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
            lambda: f"Landmark scenario profile: {ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME} active simulated landmarks",
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
            "data_association",
            data_assoc_btn,
            "Choose how landmark observations are matched to known field landmarks before the update",
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
            "Landmark-only static convergence: actual inside field; belief dx,dy uniform +/-50 cm; dtheta uniform +/-20 deg",
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
            "Run convergence cases while the actual robot follows the demo path from nearby belief starts",
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

        draw_button(
            screen,
            settings_profile_btn,
            fit_button_label(f"PROFILE {MASTERDATA_FREQUENCY_NAMES[selected_settings_profile_idx]}", font_b, settings_profile_btn.w - 16),
            font_b,
            (85, 75, 135),
            (255, 255, 255),
        )
        draw_button(
            screen,
            landmark_scenario_profile_btn,
            fit_button_label(f"LM PROFILE {ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME}", font_b, landmark_scenario_profile_btn.w - 16),
            font_b,
            (75, 105, 120),
            (255, 255, 255),
        )
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
            data_assoc_btn,
            fit_button_label(f"ASSOC: {data_association_names[selected_data_association_idx]}", font_b, data_assoc_btn.w - 16),
            font_b,
            (85, 75, 135),
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
        draw_button(screen, paper_data_btn, "PAPER 2HRS", font_b, (105, 45, 150), (255, 255, 255))

        lines = []
        lines.append("Rotate robot: hold Q / E")
        if converge_test_status is not None:
            lines.append(converge_test_status)
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
                        reset_landmark_obstruction_state()
                        continue
                    if analysis_btn.collidepoint(mx, my):
                        analysis_on = not analysis_on
                        continue
                    if algo_btn.collidepoint(mx, my):
                        selected_algorithm_idx = (selected_algorithm_idx + 1) % len(algorithm_names)
                        continue
                    if data_assoc_btn.collidepoint(mx, my):
                        selected_data_association_idx = (selected_data_association_idx + 1) % len(data_association_names)
                        ACTIVE_DATA_ASSOCIATION_NAME = data_association_names[selected_data_association_idx]
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
                    if paper_data_btn.collidepoint(mx, my):
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
        choices=DATA_ASSOCIATION_ALGORITHM_NAMES,
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



