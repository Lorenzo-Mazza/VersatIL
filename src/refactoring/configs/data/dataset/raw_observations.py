from dataclasses import dataclass, field

from refactoring.data.constants import GripperType, OrientationRepresentation


@dataclass
class RawObservationsConfig:
    """Configuration for the raw dataset observations."""
    # CSV column names for different observation proprioceptive data
    robot_frame_proprio_keys: list[str] = field(default_factory=list)
    camera_frame_proprio_keys: list[str] = field(default_factory=list)
    gripper_state_keys: list[str] = field(default_factory=list)
    # CSV column names for camera keys (for cameras used in the dataset)
    camera_keys: list[str] = field(default_factory=list)
    # Whether to use rectified images if available (for stereo datasets)
    use_rectified_images: bool = False
    #: Image width and height for optional resizing.
    image_width: int | None = None
    image_height: int | None = None
    # Language instruction key (for language-conditioned datasets)
    language_key: str | None = None
    # Custom observation modalities (name -> list of column names).
    # NB: These are assumed to be float values.
    custom_obs_keys: dict[str, list[str]] = field(default_factory=dict)
    # Observation dimensions
    has_position: bool = True
    position_dim: int = 3
    has_orientation: bool = False
    orientation_dim: int = 0
    orientation_repr: str = OrientationRepresentation.ROLL.value
    has_gripper: bool = False
    gripper_type: str = GripperType.BINARY.value
    gripper_dim: int = 1



