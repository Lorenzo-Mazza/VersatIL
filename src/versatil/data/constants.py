"""Data constants for the data package."""
import enum


class Cameras(enum.Enum):
    """Enum for camera image keys both in the zarr and in the raw datasets."""

    # TSO datasets
    LEFT = "left"
    RIGHT = "right"
    DEPTH = "depth"
    # LIBERO
    AGENTVIEW = "agentview_rgb"
    EYE_IN_HAND = "eye_in_hand_rgb"

    # LIBERO LEROBOT
    IMAGE = "observation.images.image"
    IMAGE_2 = "observation.images.image2"

    # LIBERO PLUS LEROBOT
    FRONT = "observation.images.front"
    WRIST = "observation.images.wrist"

    IMAGE_METAWORLD_LEROBOT = "observation.image"


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
    ROBOT_FRAME_CARTESIAN_TIP_ORI = "tip_ori_robot_frame"
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

    MINUS_ONE_TO_ONE = "minus_one_to_one"
    ZERO_TO_ONE = "zero_to_one"
    IMAGENET = "imagenet"


class KinematicsNormalizationType(str, enum.Enum):
    """Enum for kinematics normalization types."""

    MIN_MAX = "min_max"
    GAUSSIAN = "gaussian"
    DEMEAN = "demean"


class OrientationRepresentation(str, enum.Enum):
    """Enum for orientation representation types used in on-the-fly action computations."""

    ROLL = "roll"  # roll of the end-effector around the tool axis for a robot controlled with a Remote Center of Motion constraint
    EULER = "euler"  # (roll, pitch, yaw)
    QUATERNION = "quaternion"  # (w, x, y, z)
    # TODO: add LIBERO orientation representation


class ActionComputationMethod(str, enum.Enum):
    """Enumerates the computation methods for obtaining an action from state data."""

    NEXT_TIMESTEP = "next_timestep"  # Use the next timestep state as the action target
    DELTA = "delta"  # Subtraction between the current and next timestep


class ProprioceptiveType(str, enum.Enum):
    """Enum for proprioceptive data types. Add here custom proprioceptive types as needed."""

    POSITION = "position"
    ORIENTATION = "orientation"
    GRIPPER = "gripper"  # gripper open/close or continuous control action
    CUSTOM = "custom"  # for any custom action types


class GripperType(str, enum.Enum):
    """Enum for gripper action types."""

    BINARY = "binary"
    CONTINUOUS = "continuous"


class BinaryGripperRange(str, enum.Enum):
    """Enum for binary gripper value ranges.

    Different datasets use different conventions for binary gripper states:
    """

    ZERO_ONE = "zero_one"
    MINUS_ONE_ONE = "minus_one_one"


class TokenizerType(str, enum.Enum):
    """Enum for tokenizer types in tokenization chains."""

    FAST = "fast"  # FAST action tokenizer
    LANGUAGE = "language"  # Language model tokenizer (BERT, GPT, Gemma, etc.)


class DatasetType(str, enum.Enum):
    """Enum for dataset types, independent of storage format."""

    LIBERO = "libero"
    TSO = "tso"
    METAWORLD = "metaworld"


class LeRobotPathsV30(str, enum.Enum):
    INFO_PATH = "meta/info.json"
    STATS_PATH = "meta/stats.json"

    EPISODES_DIR = "meta/episodes"
    DATA_DIR = "data"
    VIDEO_DIR = "videos"

    CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
    DEFAULT_TASKS_PATH = "meta/tasks.parquet"
    DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
    DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
    DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
    DEFAULT_IMAGE_PATH = (
        "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png"
    )

    def __str__(self):
        return self.value


class SampleKey(str, enum.Enum):
    """Keys for sample dictionary structure in data pipeline."""

    OBSERVATION = "observation"
    ACTION = "action"
    IS_PAD_ACTION = "is_pad"
    IS_PAD_OBSERVATION = "is_pad_observation"
    TOKENIZED_OBSERVATIONS = "tokenized_observations"
    TOKENIZED_ACTIONS = "tokenized_actions"


VALID_CAMERAS = [cam.value for cam in Cameras]
RGB_CAMERAS = [
    Cameras.LEFT.value,
    Cameras.RIGHT.value,
    Cameras.AGENTVIEW.value,
    Cameras.EYE_IN_HAND.value,
]


#: ImageNet statistics for normalization
IMAGENET_RGB_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD: list[float] = [0.229, 0.224, 0.225]
IMAGENET_DEPTH_MEAN: float = 0.48  # ref. https://github.com/VCIP-RGBD/RGBD-Pretrain/blob/main/data/constants.py#L3
IMAGENET_DEPTH_STD: float = 0.28
