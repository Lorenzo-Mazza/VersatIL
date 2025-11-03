from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from refactoring.configs.decoding.decoder import (
    DecodingNetworkConfig,
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
from refactoring.data.constants import Cameras, GripperType, OrientationRepresentation


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
    cs.store(group="decoder", name="moe", node=MixtureOfExpertsDecoderConfig)


# Register resolvers on module import
register_resolvers()
