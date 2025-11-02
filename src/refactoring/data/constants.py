"""Data constants for the codebase.

All string constants should be defined here to avoid hard-coding throughout the codebase.

Organization:
    - Enums for type-safe constants
    - Observation/Action keys for data pipeline
    - Normalization constants
"""
import enum


class Cameras(enum.Enum):
    """Enum for camera names."""
    LEFT = 'left'
    RIGHT = 'right'
    DEPTH = 'depth'

VALID_CAMERAS = [cam.value for cam in Cameras]

class ImageNormalizationType(str, enum.Enum):
    """Enum for image normalization types."""
    MINUS_ONE_TO_ONE = 'minus_one_to_one'
    ZERO_TO_ONE = 'zero_to_one'
    IMAGENET = 'imagenet'


class KinematicsNormalizationType(str, enum.Enum):
    """Enum for kinematics normalization types."""
    MIN_MAX = 'min_max'
    GAUSSIAN = 'gaussian'


class SamplingMode(str, enum.Enum):
    """Enum for sampling mode of the dataset."""
    OVERLAPPING = 'overlapping'
    RANDOM_CHUNK = 'random_chunk'


class OrientationRepresentation(str, enum.Enum):
    """Enum for orientation representation types."""
    ROLL = 'roll'  # roll of the end-effector around the tool axis for a robot controlled with a Remote Center of Motion constraint
    EULER = 'euler' # (roll, pitch, yaw)
    QUATERNION = 'quaternion' # (w, action_embedding, y, z)



class GripperType(str, enum.Enum):
    """Enum for gripper action types."""
    BINARY = 'binary'
    CONTINUOUS = 'continuous'


#: Observation keys
PROPRIO_OBS_ROBOT_FRAME_KEY = "proprio_robot_frame"
PROPRIO_OBS_CAMERA_FRAME_KEY = "proprio_camera_frame"
GRIPPER_STATE_OBS_KEY = "gripper_state_obs"
PROPRIO_STATE = "robot_proprio_state"
OBSERVATION_KEY = "observation"
LANGUAGE_KEY = "language_instruction"

#: Action keys
ACTION_KEY = "action"
POSITION_ACTION_KEY = "position_action"
ORIENTATION_ACTION_KEY = "orientation_action"
GRIPPER_ACTION_KEY = "gripper_action"

#: Padding key
IS_PAD_KEY = "is_pad"

#: Shape key
SHAPE_KEY = "shape"

#: Phase label key
PHASE_LABEL_KEY = "phase_label"

#: Episode metadata
EPISODE_FILENAME = "episode.csv"





#: ImageNet statistics for normalization
IMAGENET_RGB_MEAN:list[float] = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD:list[float] = [0.229, 0.224, 0.225]
IMAGENET_DEPTH_MEAN: float = 0.48  # Cf. https://github.com/VCIP-RGBD/RGBD-Pretrain/blob/main/data/constants.py#L3
IMAGENET_DEPTH_STD: float = 0.28
