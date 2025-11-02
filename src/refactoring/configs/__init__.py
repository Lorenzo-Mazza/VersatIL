from hydra.core.config_store import ConfigStore

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
