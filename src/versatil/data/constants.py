"""Data constants for the data package.

Note:
    Wire protocol constants (observation keys, action keys, gripper types) are
    imported from ``versatil_constants`` — the single source of truth shared
    across all projects in the ecosystem.
"""

import enum

from versatil_constants.libero import LiberoCamera
from versatil_constants.shared import (  # noqa: F401
    ActionComponent,
    ActionComputationMethod,
    ActionMetadataField,
    BinaryGripperRange,
    CoordinateSystem,
    GripperType,
    ObsKey,
    OrientationRepresentation,
)
from versatil_constants.synthetic import SyntheticObsKey  # noqa: F401
from versatil_constants.tso import TSOCamera, TSOObsKey  # noqa: F401


class Cameras(enum.Enum):
    """Camera image keys used in the versatil pipeline and wire protocol."""

    LEFT = TSOCamera.LEFT.value
    RIGHT = TSOCamera.RIGHT.value
    DEPTH = TSOCamera.DEPTH.value
    AGENTVIEW = (
        LiberoCamera.AGENTVIEW.value
    )  # MetaWorldCamera.AGENTVIEW.value is identical
    EYE_IN_HAND = LiberoCamera.EYE_IN_HAND.value


class RawCameraKey(enum.StrEnum):
    """Raw dataset storage keys for camera observations.

    Each member maps to the key used in a specific raw data format.
    Remapped to ``Cameras`` keys during zarr creation via
    ``RAW_TO_CAMERA_MAPPING``.
    """

    # TSO
    LEFT = TSOCamera.LEFT.value
    RIGHT = TSOCamera.RIGHT.value
    DEPTH = TSOCamera.DEPTH.value
    # Libero Original (HDF5)
    AGENTVIEW = LiberoCamera.AGENTVIEW.value
    EYE_IN_HAND = LiberoCamera.EYE_IN_HAND.value
    # Libero LeRobot
    IMAGE = "observation.images.image"
    IMAGE_2 = "observation.images.image2"
    # Libero Plus LeRobot
    FRONT = "observation.images.front"
    WRIST = "observation.images.wrist"
    # MetaWorld LeRobot
    IMAGE_METAWORLD = "observation.image"


RAW_TO_CAMERA_MAPPING: dict[str, str] = {
    # TSO (identity)
    RawCameraKey.LEFT.value: Cameras.LEFT.value,
    RawCameraKey.RIGHT.value: Cameras.RIGHT.value,
    RawCameraKey.DEPTH.value: Cameras.DEPTH.value,
    # Libero Original (identity)
    RawCameraKey.AGENTVIEW.value: Cameras.AGENTVIEW.value,
    RawCameraKey.EYE_IN_HAND.value: Cameras.EYE_IN_HAND.value,
    # Libero LeRobot
    RawCameraKey.IMAGE.value: Cameras.AGENTVIEW.value,
    RawCameraKey.IMAGE_2.value: Cameras.EYE_IN_HAND.value,
    # Libero Plus LeRobot
    RawCameraKey.FRONT.value: Cameras.AGENTVIEW.value,
    RawCameraKey.WRIST.value: Cameras.EYE_IN_HAND.value,
    # MetaWorld LeRobot
    RawCameraKey.IMAGE_METAWORLD.value: Cameras.AGENTVIEW.value,
}


class ProprioKey(enum.StrEnum):
    """Proprioceptive observation keys for all supported environments."""

    # TSO datasets proprioceptive keys
    ROBOT_FRAME_CARTESIAN_TIP_POS = "proprio_robot_frame"
    ROBOT_FRAME_CARTESIAN_TIP_ORI = "tip_ori_robot_frame"
    CAMERA_FRAME_CARTESIAN_TIP_POS = "proprio_camera_frame"
    CAMERA_FRAME_CARTESIAN_TIP_ORI = "tip_ori_camera_frame"
    # LIBERO/Metaworld proprioceptive keys
    EE_POS = "ee_pos"
    EE_ORI = "ee_ori"
    EE_STATES = "ee_states"
    JOINT_STATES = "joint_states"
    EE_POS_ACTION = "ee_pos_action"
    EE_ORI_ACTION = "ee_ori_action"
    GRIPPER_STATE = "gripper_state_obs"
    GRIPPER_STATE_ACTION = "gripper_state_action"

    # Synthetic data
    SYNTHETIC_POSITION = "synthetic_position"
    SYNTHETIC_POSITION_ACTION = "synthetic_position_action"


class ImageNormalizationType(enum.StrEnum):
    """Image normalization types."""

    MINUS_ONE_TO_ONE = "minus_one_to_one"
    ZERO_TO_ONE = "zero_to_one"
    IMAGENET = "imagenet"


class KinematicsNormalizationType(enum.StrEnum):
    """Kinematics normalization types."""

    MIN_MAX = "min_max"
    GAUSSIAN = "gaussian"
    DEMEAN = "demean"


class TokenPaddingStrategy(enum.StrEnum):
    """Padding strategy for language tokenization."""

    MAX_LENGTH = "max_length"
    LONGEST = "longest"


class ProprioceptiveType(enum.StrEnum):
    """Proprioceptive data types."""

    POSITION = ActionComponent.POSITION.value
    ORIENTATION = ActionComponent.ORIENTATION.value
    GRIPPER = ActionComponent.GRIPPER.value
    CUSTOM = ActionComponent.CUSTOM.value


class TokenizerType(enum.StrEnum):
    """Tokenizer types in tokenization chains."""

    FAST = "fast"
    LANGUAGE = "language"


class DatasetType(enum.StrEnum):
    """Dataset types, independent of storage format."""

    LIBERO = "libero"
    TSO = "tso"
    METAWORLD = "metaworld"
    SYNTHETIC = "synthetic"


class LeRobotPathsV30(enum.StrEnum):
    """LeRobot v3.0 dataset directory layout."""

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


class SampleKey(enum.StrEnum):
    """Keys for sample dictionary structure in data pipeline."""

    OBSERVATION = "observation"
    ACTION = "action"
    IS_PAD_ACTION = "is_pad"
    IS_PAD_OBSERVATION = "is_pad_observation"
    TOKENIZED_OBSERVATIONS = "tokenized_observations"
    TOKENIZED_ACTIONS = "tokenized_actions"


VALID_CAMERAS = [cam.value for cam in Cameras]
VALID_RAW_CAMERA_KEYS = [key.value for key in RawCameraKey]

RGB_CAMERAS = [
    Cameras.LEFT.value,
    Cameras.RIGHT.value,
    Cameras.AGENTVIEW.value,
    Cameras.EYE_IN_HAND.value,
]

DEPTH_CAMERAS = [
    Cameras.DEPTH.value,
]

# ref. https://github.com/VCIP-RGBD/RGBD-Pretrain/blob/main/data/constants.py#L3
IMAGENET_RGB_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD: list[float] = [0.229, 0.224, 0.225]
IMAGENET_DEPTH_MEAN: float = 0.48
IMAGENET_DEPTH_STD: float = 0.28
