"""Data constants for the data package.

Note:
    Wire protocol constants (observation keys, action keys, gripper types) are
    imported from ``versatil_constants`` — the single source of truth shared
    across all projects in the ecosystem.
"""

import enum

from versatil_constants.blockpush import BlockPushProprioKey
from versatil_constants.kitchen import KitchenProprioKey
from versatil_constants.libero import LiberoCamera, LiberoProprioKey
from versatil_constants.multimodal_ant import MultimodalAntProprioKey
from versatil_constants.pusht import PushTProprioKey
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
from versatil_constants.tso import TSOCamera, TSOProprioKey
from versatil_constants.ur3 import UR3ProprioKey


class Cameras(enum.Enum):
    """Camera image keys used in the versatil pipeline and to wire server protocols, coming from `versatil_constants` library.

    Note:
        Cf. https://github.com/nct-tso-robotics/versatil_constants for the single source of truth for camera keys across all projects in the ecosystem.
    """

    LEFT = TSOCamera.LEFT.value
    RIGHT = TSOCamera.RIGHT.value
    DEPTH = TSOCamera.DEPTH.value
    # Libero agent-view key is intentionally reused by MetaWorld, PushT, and Kitchen simulation constants.
    AGENTVIEW = LiberoCamera.AGENTVIEW.value
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
    # PushT LeRobot
    IMAGE_PUSHT = "observation.image"
    # Kitchen LeRobot
    IMAGE_KITCHEN = "observation.image"


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
    RawCameraKey.IMAGE_METAWORLD.value: Cameras.AGENTVIEW.value,
    RawCameraKey.IMAGE_PUSHT.value: Cameras.AGENTVIEW.value,
    RawCameraKey.IMAGE_KITCHEN.value: Cameras.AGENTVIEW.value,
}


class ProprioKey(enum.StrEnum):
    """Proprioceptive observation keys for all supported environments.

    Note:
        Cf. https://github.com/nct-tso-robotics/versatil_constants for the single source of truth for proprioceptive keys across all projects in the ecosystem.
    """

    # TSO datasets proprioceptive keys
    ROBOT_FRAME_CARTESIAN_TIP_POS = TSOProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
    ROBOT_FRAME_CARTESIAN_TIP_ORI = TSOProprioKey.ROBOT_FRAME_CARTESIAN_TIP_ORI.value
    CAMERA_FRAME_CARTESIAN_TIP_POS = TSOProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value
    CAMERA_FRAME_CARTESIAN_TIP_ORI = TSOProprioKey.CAMERA_FRAME_CARTESIAN_TIP_ORI.value
    # LIBERO/Metaworld proprioceptive keys
    EE_POS = LiberoProprioKey.EE_POS.value
    EE_ORI = LiberoProprioKey.EE_ORI.value
    EE_STATES = LiberoProprioKey.EE_STATES.value
    JOINT_STATES = LiberoProprioKey.JOINT_STATES.value
    EE_POS_ACTION = LiberoProprioKey.EE_POS_ACTION.value
    EE_ORI_ACTION = LiberoProprioKey.EE_ORI_ACTION.value
    GRIPPER_STATE = LiberoProprioKey.GRIPPER_STATE.value
    GRIPPER_STATE_ACTION = LiberoProprioKey.GRIPPER_STATE_ACTION.value

    # Synthetic data, these don't live in versatil_constants because they are never passed over the wire at inference time,
    # but are only used locally within versatil for synthetic benchmarks.
    SYNTHETIC_POSITION = "synthetic_position"
    SYNTHETIC_POSITION_ACTION = "synthetic_position_action"

    # PushT data
    PUSHT_BLOCK_POS = PushTProprioKey.BLOCK_POS.value
    PUSHT_BLOCK_ANGLE = PushTProprioKey.BLOCK_ANGLE.value
    PUSHT_KEYPOINTS = PushTProprioKey.KEYPOINTS.value
    PUSHT_CONTACTS = PushTProprioKey.CONTACTS.value

    # Relay kitchen data
    KITCHEN_ARM_QPOS = KitchenProprioKey.ARM_QPOS.value
    KITCHEN_OBJECT_QPOS = KitchenProprioKey.OBJECT_QPOS.value
    KITCHEN_TASK_GOAL = KitchenProprioKey.TASK_GOAL.value
    KITCHEN_ARM_ACTION = KitchenProprioKey.ARM_ACTION.value

    # Ant maze multimodal data
    ANT_QPOS = MultimodalAntProprioKey.QPOS.value
    ANT_QVEL = MultimodalAntProprioKey.QVEL.value
    ANT_GOAL_COORDS = MultimodalAntProprioKey.GOAL_COORDS.value
    ANT_ACHIEVED = MultimodalAntProprioKey.ACHIEVED.value
    ANT_TORQUE_ACTION = MultimodalAntProprioKey.TORQUE_ACTION.value

    # UR3 block-pushing data
    UR3_EE_POS = UR3ProprioKey.EE_POS.value
    UR3_BLOCK1_POS = UR3ProprioKey.BLOCK1_POS.value
    UR3_BLOCK2_POS = UR3ProprioKey.BLOCK2_POS.value
    UR3_EE_TARGET_ACTION = UR3ProprioKey.EE_TARGET_ACTION.value

    # Block pushing data
    BLOCK_PUSH_BLOCK1_POS = BlockPushProprioKey.BLOCK1_POS.value
    BLOCK_PUSH_BLOCK1_ANGLE = BlockPushProprioKey.BLOCK1_ANGLE.value
    BLOCK_PUSH_BLOCK2_POS = BlockPushProprioKey.BLOCK2_POS.value
    BLOCK_PUSH_BLOCK2_ANGLE = BlockPushProprioKey.BLOCK2_ANGLE.value
    BLOCK_PUSH_EE_COMMANDED = BlockPushProprioKey.EE_COMMANDED.value
    BLOCK_PUSH_TARGET1_POS = BlockPushProprioKey.TARGET1_POS.value
    BLOCK_PUSH_TARGET1_ANGLE = BlockPushProprioKey.TARGET1_ANGLE.value
    BLOCK_PUSH_TARGET2_POS = BlockPushProprioKey.TARGET2_POS.value
    BLOCK_PUSH_TARGET2_ANGLE = BlockPushProprioKey.TARGET2_ANGLE.value


class SyntheticObsKey(enum.StrEnum):
    """Non-proprioceptive observation keys for synthetic benchmark tasks.

    Defined locally in versatil (not in versatil_constants) because synthetic
    keys are never passed over the client-server wire at inference time.
    """

    CONTEXT = "synthetic_context"
    MODE_ID = "synthetic_mode_id"


class ImageNormalizationType(enum.StrEnum):
    """Image normalization types."""

    MINUS_ONE_TO_ONE = "minus_one_to_one"
    ZERO_TO_ONE = "zero_to_one"
    IMAGENET = "imagenet"
    CLIP = "clip"


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
    PUSHT = "pusht"
    BLOCK_PUSHING = "block_pushing"
    KITCHEN = "kitchen"
    ANT = "ant"
    UR3 = "ur3"


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


class MetadataPassthroughSource(enum.StrEnum):
    """Source dictionaries that can be copied into training metadata."""

    OBSERVATION = SampleKey.OBSERVATION.value
    ACTION = SampleKey.ACTION.value
    PREDICTION = "prediction"


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

# Ref. https://github.com/VCIP-RGBD/RGBD-Pretrain/blob/main/data/constants.py#L3
IMAGENET_RGB_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD: list[float] = [0.229, 0.224, 0.225]
# Ref. OpenAI CLIP preprocess and HF CLIP preprocessor_config.
# https://github.com/openai/CLIP/blob/main/clip/clip.py
# https://huggingface.co/openai/clip-vit-base-patch32/blob/main/preprocessor_config.json
CLIP_RGB_MEAN: list[float] = [0.48145466, 0.4578275, 0.40821073]
CLIP_RGB_STD: list[float] = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEPTH_MEAN: float = 0.48
IMAGENET_DEPTH_STD: float = 0.28
