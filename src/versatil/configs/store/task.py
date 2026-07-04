"""Task, dataset schema, and metadata config registrations."""

from hydra.core.config_store import ConfigStore

from versatil.configs import (
    ActionDiscretizerConfig,
    ActionSpaceConfig,
    ActionTokenIdMappingConfig,
    ActionTokenizationConfig,
    AugmentationPipelineConfig,
    CameraMetadataConfig,
    CsvDatasetSchemaConfig,
    DataLoaderConfig,
    DatasetMetadataConfig,
    DatasetSchemaConfig,
    DepthCameraMetadataConfig,
    GripperActionMetadataConfig,
    GripperObservationMetadataConfig,
    Hdf5DatasetSchemaConfig,
    LeRobotDatasetSchemaConfig,
    ObservationMetadataConfig,
    ObservationSpaceConfig,
    ObservationTokenizationConfig,
    OrientationActionMetadataConfig,
    OrientationObservationMetadataConfig,
    PositionActionMetadataConfig,
    PositionObservationMetadataConfig,
    PrecomputedActionMetadataConfig,
    RGBCameraMetadataConfig,
    SyntheticDatasetSchemaConfig,
    TaskSpaceConfig,
    TokenizationConfig,
)


def register(cs: ConfigStore) -> None:
    """Store this domain's config nodes.

    Args:
        cs: The global Hydra config store.
    """
    cs.store(group="task", name="base", node=TaskSpaceConfig)
    cs.store(
        group="task/dataset_schema/zarr_meta", name="base", node=DatasetMetadataConfig
    )
    cs.store(
        group="task/dataset_schema/metadata/observation",
        name="base",
        node=ObservationMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/observation",
        name="position",
        node=PositionObservationMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/observation",
        name="orientation",
        node=OrientationObservationMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/observation",
        name="gripper",
        node=GripperObservationMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/camera",
        name="base",
        node=CameraMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/camera",
        name="rgb",
        node=RGBCameraMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/camera",
        name="depth",
        node=DepthCameraMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/precomputed_action",
        name="base",
        node=PrecomputedActionMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/precomputed_action",
        name="position",
        node=PositionActionMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/precomputed_action",
        name="orientation",
        node=OrientationActionMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema/metadata/precomputed_action",
        name="gripper",
        node=GripperActionMetadataConfig,
    )
    cs.store(
        group="task/dataset_schema", name="lerobot", node=LeRobotDatasetSchemaConfig
    )
    cs.store(group="task/dataset_schema", name="base", node=DatasetSchemaConfig)
    cs.store(group="task/dataset_schema", name="hdf5", node=Hdf5DatasetSchemaConfig)
    cs.store(group="task/dataset_schema", name="csv", node=CsvDatasetSchemaConfig)
    cs.store(
        group="task/dataset_schema",
        name="synthetic_schema",
        node=SyntheticDatasetSchemaConfig,
    )
    cs.store(
        group="task/dataset_schema/synthetic",
        name="synthetic_schema",
        node=SyntheticDatasetSchemaConfig,
    )
    cs.store(group="task/dataloader", name="base", node=DataLoaderConfig)
    cs.store(
        group="task/dataloader/image_augmentations",
        name="base",
        node=AugmentationPipelineConfig,
    )
    cs.store(group="task/dataloader/tokenization", name="base", node=TokenizationConfig)
    cs.store(
        group="task/dataloader/tokenization/action",
        name="base",
        node=ActionTokenizationConfig,
    )
    cs.store(
        group="task/dataloader/tokenization/action/discretizer",
        name="base",
        node=ActionDiscretizerConfig,
    )
    cs.store(
        group="task/dataloader/tokenization/action/token_id_mapping",
        name="base",
        node=ActionTokenIdMappingConfig,
    )
    cs.store(
        group="task/dataloader/tokenization/observation",
        name="base",
        node=ObservationTokenizationConfig,
    )
    cs.store(group="task/action_space", name="base", node=ActionSpaceConfig)
    cs.store(group="task/observation_space", name="base", node=ObservationSpaceConfig)
