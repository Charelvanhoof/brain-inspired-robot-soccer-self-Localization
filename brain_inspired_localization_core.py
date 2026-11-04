# Code for the Robot_Soccer_Self_Localization_with_Active_Inference
# Based on version POC V29
import math
from copy import deepcopy
from collections import Counter


# =========================
# CORE
# =========================
# Algorithm settings and built-in landmark profiles. Belief remains caller-owned.
DT = 0.020
PI_CAM = PI_ODO = PI_IMU = 1.0
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
DATA_ASSOCIATION_ALGORITHM_NAMES = [DATA_ASSOCIATION_GATED_NEAREST_NAME, DATA_ASSOCIATION_UNIQUE_ID_NAME]
ACTIVE_DATA_ASSOCIATION_NAME = DATA_ASSOCIATION_GATED_NEAREST_NAME
LANDMARK_ASSOCIATION_GATE_CM = 1000.0
GENERALIZED_COORDINATE_ORDER = 2
DISTANCE_PRECISION_REF_CM = 600.0
DISTANCE_PRECISION_MIN_FACTOR = 0.15
PRECISION_DISTANCE_MODIFIER_ENABLED = True
MOTION_LR_VEL_XY = MOTION_LR_VEL_THETA = 5.0
MOTION_LR_ACC_XY = MOTION_LR_ACC_THETA = 50.0
LANDMARK_MAP = []
LANDMARK_POINT_IDS = ()
LANDMARKS = {}
LANDMARK_METADATA = {}
ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME = None


CORE_SETTING_DEFAULTS = {
    name: globals()[name] for name in (
        "PI_CAM", "PI_ODO", "PI_IMU", "LR_XY", "LR_ANGLE", "ANGLE_WEIGHT",
        "MOTION_LR_VEL_XY", "MOTION_LR_ACC_XY", "MOTION_LR_VEL_THETA", "MOTION_LR_ACC_THETA",
        "PRECISION_DISTANCE_MODIFIER_ENABLED", "DISTANCE_PRECISION_REF_CM",
        "DISTANCE_PRECISION_MIN_FACTOR", "LANDMARK_ASSOCIATION_GATE_CM", "ACTIVE_DATA_ASSOCIATION_NAME",
    )
}


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


def landmark_profile_association_name(profile):
    """Describe recognition ambiguity for reports; matching uses each observed name."""
    names = [landmark["name"] for landmark in profile["landmarks"]]
    return (DATA_ASSOCIATION_UNIQUE_ID_NAME if len(set(names)) == len(names)
            else DATA_ASSOCIATION_GATED_NEAREST_NAME)


def set_landmark_profile(name):
    # Input: name of a built-in, self-contained landmark profile.
    # Output: selected map and association mode in the core configuration.
    # Function: select the map without loading files or changing belief/settings.
    profile = next(profile for profile in LANDMARK_SCENARIO_PROFILES if profile["name"] == name)
    landmarks = profile["landmarks"]
    point_ids = tuple(str(index) for index in range(len(landmarks)))
    counts = Counter(landmark["name"] for landmark in landmarks)
    globals().update(
        ACTIVE_LANDMARK_SCENARIO_PROFILE_NAME=name,
        ACTIVE_DATA_ASSOCIATION_NAME=landmark_profile_association_name(profile),
        LANDMARK_MAP=landmarks,
        LANDMARK_POINT_IDS=point_ids,
        LANDMARKS={key: (landmark["x_cm"], landmark["y_cm"])
                   for key, landmark in zip(point_ids, landmarks)},
        LANDMARK_METADATA={key: dict(name=landmark["name"], type=landmark["type"],
                                     unique=counts[landmark["name"]] == 1)
                           for key, landmark in zip(point_ids, landmarks)},
    )


set_landmark_profile(LANDMARK_SCENARIO_PROFILES[0]["name"])


def landmark_name(landmark_id):
    return LANDMARK_METADATA[landmark_id]["name"]


def landmark_candidate_ids_for_name(name):
    return [key for key in LANDMARKS if landmark_name(key) == name]


def unique_landmark_candidate_id_for_name(name):
    candidates = landmark_candidate_ids_for_name(name)
    return candidates[0] if len(candidates) == 1 else None


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
    #   used so landmark predictions or named observations can be accessed directly during inference and analysis
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
    #   landmark predictions in robot coordinates (RCS): id, name, x_rcs_cm, y_rcs_cm, angle_rad
    # Function:
    #   FCS -> RCS translation
    #   converts field landmarks into expected sensory observations for the robot
    obs = []
    for name, (lx, ly) in landmarks_fcs.items():
        x_rcs, y_rcs, angle_rad = landmark_fcs_2_rcs(robot_fcs_gc, lx, ly)
        obs.append({
            "id": name,
            "name": landmark_name(name),
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
    #   y_obs: prepared named landmark observation in RCS with x_rcs_cm, y_rcs_cm, and calculated angle_rad
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
    #   y_obs_map: observed named landmark map in RCS with x_rcs_cm and y_rcs_cm
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
    #   y_obs: prepared named landmark observation in RCS with calculated angle_rad
    #   yhat_map: predicted candidate landmark observations in RCS, keyed by internal landmark id
    #   association_name: selected data-association strategy
    #   used_candidate_ids: candidates already consumed by this inference step
    # Output:
    #   association record with chosen candidate id, prediction errors, and gate status
    # Function:
    #   resolves one named sensory observation to an internal landmark candidate
    #   using the selected ambiguity/data-association strategy
    # Find candidates solely by the recognition name; never read simulator identity.
    used_candidate_ids = used_candidate_ids or set()
    if association_name not in {DATA_ASSOCIATION_GATED_NEAREST_NAME, DATA_ASSOCIATION_UNIQUE_ID_NAME}:
        raise ValueError(f"Unknown landmark association algorithm: {association_name}")
    unique_candidate_id = unique_landmark_candidate_id_for_name(y_obs["name"])
    if unique_candidate_id is not None and unique_candidate_id in yhat_map and unique_candidate_id not in used_candidate_ids:
        eps, position_error_cm, error_score = landmark_prediction_error(y_obs, yhat_map[unique_candidate_id])
        return {
            "accepted": True,
            "candidate_id": unique_candidate_id,
            "eps": eps,
            "position_error_cm": position_error_cm,
            "error_score": error_score,
            "association_name": (DATA_ASSOCIATION_UNIQUE_ID_NAME if association_name == DATA_ASSOCIATION_UNIQUE_ID_NAME
                                 else DATA_ASSOCIATION_UNIQUE_DIRECT_NAME),
        }

    candidates = [
        candidate_id
        for candidate_id in landmark_candidate_ids_for_name(y_obs["name"])
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


def landmark_inference_step(belief_fcs_gc, y_obs_map, step_dt_s=DT, diagnostics=None):
    # Input:
    #   belief_fcs_gc: belief robot state in FCS generalized coordinates
    #   y_obs_map: prepared named landmark map in RCS with x_rcs_cm, y_rcs_cm, and calculated angle_rad
    #   step_dt_s: timestep for this landmark inference update
    #   diagnostics: optional list receiving actual per-observation update terms
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
    for obs_id, y_obs in y_obs_map.items():  # for each observed named place cell
        # Associate the named observation with the best internal landmark candidate.
        association = associate_landmark_observation(
            y_obs,
            yhat_map,
            ACTIVE_DATA_ASSOCIATION_NAME,
            used_candidate_ids,
        )
        term = {"obs_id": obs_id, **association, "used_for_update": False}
        if diagnostics is not None:
            diagnostics.append(term)
        if not association["accepted"]:
            term["skip_reason"] = "association_rejected"
            continue
        lid = association["candidate_id"]
        # Normal tracking uses pose landmarks only.
        if LANDMARK_METADATA[lid]["type"] != "pose":
            term["skip_reason"] = "orientation_landmark"
            continue
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
        term.update(used_for_update=True, precision=precision, jacobian=J,
                    jt_p_eps=g, mu_dot_contribution=[LR_XY * g[0], LR_XY * g[1], LR_ANGLE * g[2]],
                    free_energy_contribution=precision * sum(value * value for value in eps))
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
    #     landmark_observations_rcs: observed landmarks in RCS as name + x_rcs_cm + y_rcs_cm
    #       angle_rad is not required from C++; it is calculated from x/y before landmark inference
    #     motion_observation: observed motion terms in RCS
    #     algorithm_name: optional inference mode
    #     step_dt_s: required inference timestep, preferably computed by C++ from frame timestamps
    #     timestamp: required string for traceability/logging; no specific format required
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
            if not all(math.isfinite(value) for value in shielded[axis]):
                raise ValueError(f"belief_fcs_gc axis '{axis}' must contain finite values")
        shielded["theta"][0] = wrap_rad(shielded["theta"][0])
        return shielded

    def shield_landmark_observations_rcs(landmark_observations_rcs):
        shielded = []
        for idx, obs in enumerate(landmark_observations_rcs or []):
            for key in ("name", "x_rcs_cm", "y_rcs_cm"):
                if key not in obs:
                    raise ValueError(f"landmark_observations_rcs[{idx}] is missing '{key}'")
            shielded.append({
                "obs_id": str(obs.get("obs_id", f"obs_{idx}")),
                "name": str(obs["name"]),
                "x_rcs_cm": float(obs["x_rcs_cm"]),
                "y_rcs_cm": float(obs["y_rcs_cm"]),
            })
            if not all(math.isfinite(shielded[-1][key]) for key in ("x_rcs_cm", "y_rcs_cm")):
                raise ValueError(f"landmark_observations_rcs[{idx}] coordinates must be finite")
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
            if not math.isfinite(shielded[key]):
                raise ValueError(f"motion_observation '{key}' must be finite")
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
    timestamp = request.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValueError("timestamp must be a string")

    belief_fcs_gc = shield_belief_fcs_gc(request["belief_fcs_gc"])
    landmark_observations_rcs = shield_landmark_observations_rcs(request.get("landmark_observations_rcs", []))
    algorithm_name = shield_algorithm_name(request.get("algorithm_name"))
    motion_observation = shield_motion_observation(request.get("motion_observation"))
    step_dt_s = float(request["step_dt_s"])
    if not math.isfinite(step_dt_s) or step_dt_s <= 0:
        raise ValueError("step_dt_s must be finite and positive")

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
        "diagnostics": deepcopy(diagnostics),
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
    landmark_terms = []

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

    landmark_prediction_gc = deepcopy(belief_fcs_gc)

    # Landmark prediction errors, Jacobian/update-term construction, and Euler update.
    if algorithm_uses_landmarks(algorithm_name):
        yhat_map, landmark_free_energy = landmark_inference_step(
            belief_fcs_gc,
            y_obs_map,
            step_dt_s=step_dt_s,
            diagnostics=landmark_terms,
        )
        free_energy += landmark_free_energy

    diagnostics = {
        "free_energy": free_energy,
        "motion_free_energy": motion_free_energy,
        "landmark_free_energy": landmark_free_energy,
        "motion_terms": motion_terms,
        "y_obs": y_obs_map,
        "y_hat": yhat_map,
        "landmark_terms": landmark_terms,
        "landmark_prediction_gc": landmark_prediction_gc,
        "settings": {name: globals()[name] for name in CORE_SETTING_DEFAULTS},
    }
    return belief_fcs_gc, diagnostics


