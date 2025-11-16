from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from refactoring.configs.decoding.decoder import (
    ACTConfig,
    DecodingNetworkConfig,
    FreeTransformerConfig,
    MixtureOfExpertsDecoderConfig,
)
from refactoring.configs.encoding.encoder import (
    DepthEncoderConfig,
    EncoderConfig,
    LanguageEncoderConfig,
    StateEncoderConfig,
)
from refactoring.configs.encoding.image import ImageEncoderConfig
from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.inference import InferenceConfig
from refactoring.configs.main import MainConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.task.dataloader import DataloaderConfig
from refactoring.configs.task.task import TaskConfig
from refactoring.configs.training import (
    OptimizerConfig,
    ParameterGroupConfig,
    TrainingConfig,
)
from refactoring.data.constants import (
    Cameras,
    GripperType,
    GRIPPER_ACTION_KEY,
    LANGUAGE_KEY,
    OrientationRepresentation,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
)
from refactoring.models.encoding.encoders.constants import RGBBackboneType, PoolingMethod, LanguageEncoderType
from refactoring.training.constants import Float32MatmulPrecision, PrecisionType


def register_resolvers():
    """Register custom OmegaConf resolvers for enum access in YAML configs.

    This allows using ${cameras:LEFT} in YAML to get Cameras.LEFT.value.
    """
    if not OmegaConf.has_resolver("cameras"):
        OmegaConf.register_new_resolver("cameras", lambda name: Cameras[name].value)
    if not OmegaConf.has_resolver("gripper"):
        OmegaConf.register_new_resolver("gripper", lambda name: GripperType[name].value)
    if not OmegaConf.has_resolver("orientation"):
        OmegaConf.register_new_resolver("orientation", lambda name: OrientationRepresentation[name].value)
    if not OmegaConf.has_resolver("rgb_backbone"):
        OmegaConf.register_new_resolver("rgb_backbone", lambda name: RGBBackboneType[name].value)
    if not OmegaConf.has_resolver("precision"):
        OmegaConf.register_new_resolver("precision", lambda name: PrecisionType[name].value)
    if not OmegaConf.has_resolver("float32_matmul"):
        OmegaConf.register_new_resolver("float32_matmul", lambda name: Float32MatmulPrecision[name].value)
    if not OmegaConf.has_resolver("pooling_method"):
        OmegaConf.register_new_resolver("pooling_method", lambda name: PoolingMethod[name].value)
    if not OmegaConf.has_resolver("language_model"):
        OmegaConf.register_new_resolver("language_model", lambda name: LanguageEncoderType[name].value)
    if not OmegaConf.has_resolver("action_key"):
        action_key_map = {
            "POSITION": POSITION_ACTION_KEY,
            "ORIENTATION": ORIENTATION_ACTION_KEY,
            "GRIPPER": GRIPPER_ACTION_KEY,
        }
        OmegaConf.register_new_resolver("action_key", lambda name: action_key_map[name])
    if not OmegaConf.has_resolver("obs_key"):
        obs_key_map = {
            "PROPRIO_CAMERA_FRAME": PROPRIO_OBS_CAMERA_FRAME_KEY,
            "PROPRIO_ROBOT_FRAME": PROPRIO_OBS_ROBOT_FRAME_KEY,
            "LANGUAGE": LANGUAGE_KEY,
        }
        OmegaConf.register_new_resolver("obs_key", lambda name: obs_key_map[name])


def register_configs():
    cs = ConfigStore.instance()

    # Register main config
    cs.store(name="config", node=MainConfig)

    # Register groups
    cs.store(group="data", name="base", node=DataloaderConfig)
    cs.store(group="experiment", name="base", node=ExperimentConfig)
    cs.store(group="inference", name="base", node=InferenceConfig)
    cs.store(group="task", name="base", node=TaskConfig)
    cs.store(group="training", name="base", node=TrainingConfig)
    cs.store(group="policy", name="base", node=PolicyConfig)

    # Register encoder variants
    cs.store(group="encoder", name="image", node=ImageEncoderConfig)
    cs.store(group="encoder", name="depth", node=DepthEncoderConfig)
    cs.store(group="encoder", name="state", node=StateEncoderConfig)
    cs.store(group="encoder", name="language", node=LanguageEncoderConfig)

    # Register decoder variants
    cs.store(group="decoder", name="base", node=DecodingNetworkConfig)
    cs.store(group="decoder", name="act", node=ACTConfig)
    cs.store(group="decoder", name="free_transformer", node=FreeTransformerConfig)
    cs.store(group="decoder", name="moe", node=MixtureOfExpertsDecoderConfig)


# Register resolvers on module import
register_resolvers()
