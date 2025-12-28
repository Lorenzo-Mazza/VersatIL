"""Data constants for the data package."""
import enum


class Cameras(enum.Enum):
    """Enum for camera image keys both in the zarr and in the raw datasets."""
    # TSO datasets
    LEFT = 'left'
    RIGHT = 'right'
    DEPTH = 'depth'
    # LIBERO
    AGENTVIEW = 'agentview_rgb'
    EYE_IN_HAND = 'eye_in_hand_rgb'



class CoordinateSystem(str, enum.Enum):
    """Enum for different coordinate systems supported by the codebase"""
    ROBOT_BASE = "robot_base"
    ROBOT_EE = "robot_ee"
    CAMERA = "camera"
    UNKNOWN = "unknown"


class ProprioKey(str, enum.Enum):
    """Enum for proprioceptive observation keys."""
    # TSO datasets proprioceptive keys
    ROBOT_FRAME_CARTESIAN_TIP_POS = "proprio_robot_frame"
    ROBOT_FRAME_CARTESIAN_TIP_ORI= "tip_ori_robot_frame"
    CAMERA_FRAME_CARTESIAN_TIP_POS = "proprio_camera_frame"
    CAMERA_FRAME_CARTESIAN_TIP_ORI = "tip_ori_camera_frame"
    # LIBERO proprioceptive keys
    EE_POS = "ee_pos"
    EE_ORI = "ee_ori"
    EE_STATES = "ee_states"
    JOINT_STATES = "joint_states"
    EE_POS_ACTION = "ee_pos_action"
    EE_ORI_ACTION = "ee_ori_action"

    GRIPPER_STATE = "gripper_state_obs"
    GRIPPER_STATE_ACTION = "gripper_state_action"


class ObsKey(str, enum.Enum):
    """Enum for non-proprioceptive observation keys."""
    LANGUAGE = "language_instruction"
    PHASE_LABEL = "phase_label"


class ImageNormalizationType(str, enum.Enum):
    """Enum for image normalization types."""
    MINUS_ONE_TO_ONE = 'minus_one_to_one'
    ZERO_TO_ONE = 'zero_to_one'
    IMAGENET = 'imagenet'


class KinematicsNormalizationType(str, enum.Enum):
    """Enum for kinematics normalization types."""
    MIN_MAX = 'min_max'
    GAUSSIAN = 'gaussian'
    DEMEAN = 'demean'


class OrientationRepresentation(str, enum.Enum):
    """Enum for orientation representation types used in on-the-fly action computations."""
    ROLL = 'roll'  # roll of the end-effector around the tool axis for a robot controlled with a Remote Center of Motion constraint
    EULER = 'euler' # (roll, pitch, yaw)
    QUATERNION = 'quaternion' # (w, x, y, z)
    # TODO: add LIBERO orientation representation


class ActionComputationMethod(str, enum.Enum):
    """Enumerates the computation methods for obtaining an action from state data."""
    NEXT_TIMESTEP = 'next_timestep' # Use the next timestep state as the action target
    DELTA = 'delta' # Subtraction between the current and next timestep


class ProprioceptiveType(str, enum.Enum):
    """Enum for proprioceptive data types. Add here custom proprioceptive types as needed."""
    POSITION = 'position'
    ORIENTATION = 'orientation'
    GRIPPER = 'gripper' # gripper open/close or continuous control action
    CUSTOM = 'custom' # for any custom action types


class GripperType(str, enum.Enum):
    """Enum for gripper action types."""
    BINARY = 'binary'
    CONTINUOUS = 'continuous'


class BinaryGripperRange(str, enum.Enum):
    """Enum for binary gripper value ranges.

    Different datasets use different conventions for binary gripper states:
    """
    ZERO_ONE = 'zero_one'
    MINUS_ONE_ONE = 'minus_one_one'


class TokenizerType(str, enum.Enum):
    """Enum for tokenizer types in tokenization chains."""
    FAST = 'fast'  # FAST action tokenizer
    LANGUAGE = 'language'  # Language model tokenizer (BERT, GPT, Gemma, etc.)


#: Observation keys
OBSERVATION_KEY = "observation"
PROPRIO_OBS_ROBOT_FRAME_KEY = "proprio_robot_frame"
PROPRIO_OBS_CAMERA_FRAME_KEY = "proprio_camera_frame"
LANGUAGE_KEY = "language_instruction"

#: Action keys
ACTION_KEY = "action"
POSITION_ACTION_KEY = "position_action"
ORIENTATION_ACTION_KEY = "orientation_action"
GRIPPER_ACTION_KEY = "gripper_action"
IS_PAD_ACTION_KEY = "is_pad"
IS_PAD_OBSERVATION_KEY = "is_pad_observation"
TOKENIZED_OBSERVATIONS_KEY = "tokenized_observations"
TOKENIZED_ACTIONS_KEY = "tokenized_actions"



VALID_CAMERAS = [cam.value for cam in Cameras]
RGB_CAMERAS = [Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.AGENTVIEW.value, Cameras.EYE_IN_HAND.value]



#: ImageNet statistics for normalization
IMAGENET_RGB_MEAN:list[float] = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD:list[float] = [0.229, 0.224, 0.225]
IMAGENET_DEPTH_MEAN: float = 0.48  # ref. https://github.com/VCIP-RGBD/RGBD-Pretrain/blob/main/data/constants.py#L3
IMAGENET_DEPTH_STD: float = 0.28
