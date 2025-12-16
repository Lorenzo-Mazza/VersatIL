from refactoring.configs.data.dataloader import DataLoaderConfig
from refactoring.data.constants import (
    OrientationRepresentation,
    GripperType,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    GRIPPER_STATE_OBS_KEY,
    PHASE_LABEL_KEY,
    LANGUAGE_KEY,
    VALID_CAMERAS,
    PRECOMPUTED_ACTIONS_KEY,
)
from refactoring.data.schemas.base import DatasetSchema


class ActionSpace:
    """Defines what actions the task will predict and how they are computed."""
    def __init__(
        self,
        has_position: bool = True,
        position_dim: int = 3,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        orientation_repr: str = OrientationRepresentation.ROLL.value,
        has_gripper: bool = True,
        gripper_type: str = GripperType.BINARY.value,
        gripper_dim: int = 1,
        use_gripper_class_weights: bool = False,
        predict_in_camera_frame: bool = True,
        deltas_as_actions: bool = False,
        denoise_actions: bool = True,
        custom_action_dims: dict[str, int] = None,
        task_has_phases: bool = False,
        number_of_phases: int = 5,
        use_precomputed_actions: bool = False,
    ):
        """Initialize ActionSpace.

        Args:
            has_position: Whether to include position in action space
            position_dim: Dimension of position action (usually 3)
            has_orientation: Whether to include orientation in action space
            orientation_dim: Dimension of orientation action (1, 3, or 4)
            orientation_repr: Representation of orientation (e.g., roll, euler, quaternion)
            has_gripper: Whether to include gripper state in action space
            gripper_type: Type of gripper action (e.g., binary, continuous)
            gripper_dim: Dimension of gripper action (usually 1)
            use_gripper_class_weights: Whether to use class weights for gripper loss
            predict_in_camera_frame: Whether actions are predicted in camera frame
            deltas_as_actions: Whether actions are deltas from current state
            denoise_actions: Whether to denoise actions during training
            custom_action_dims: Dictionary of custom action dimensions
            task_has_phases: Whether the task has distinct phases
            number_of_phases: Number of phases in the task
            use_precomputed_actions: Whether the actions are going to be computed on-the-fly based on observations or already stored.
        """
        self.has_position = has_position
        self.position_dim = position_dim
        self.has_orientation = has_orientation
        self.orientation_dim = orientation_dim
        self.orientation_repr = orientation_repr
        self.has_gripper = has_gripper
        self.gripper_type = gripper_type
        self.gripper_dim = gripper_dim
        self.use_gripper_class_weights = use_gripper_class_weights
        self.predict_in_camera_frame = predict_in_camera_frame
        self.deltas_as_actions = deltas_as_actions
        self.denoise_actions = denoise_actions
        self.custom_action_dims = custom_action_dims if custom_action_dims is not None else {}
        self.task_has_phases = task_has_phases
        self.number_of_phases = number_of_phases
        self.use_precomputed_actions = use_precomputed_actions




    def get_total_action_dim(self) -> int:
        """Calculate total action dimension.

        Returns:
            Total dimension of action space
        """
        total = 0
        if self.has_position:
            total += self.position_dim
        if self.has_orientation:
            total += self.orientation_dim
        if self.has_gripper:
            total += self.gripper_dim
        for dim in self.custom_action_dims.values():
            total += dim
        if self.task_has_phases:
            total += self.number_of_phases
        return total


    def get_required_zarr_keys(self) -> list[str]:
        """Get zarr keys needed for computing actions.

        Returns:
            List of keys to load from replay buffer
        """
        keys = []
        if self.use_precomputed_actions:
            keys.append(PRECOMPUTED_ACTIONS_KEY)
            if self.task_has_phases:
                keys.append(PHASE_LABEL_KEY)
            return keys
        else:
            if self.has_position or self.has_orientation:
                if self.predict_in_camera_frame:
                    keys.append(PROPRIO_OBS_CAMERA_FRAME_KEY)
                else:
                    keys.append(PROPRIO_OBS_ROBOT_FRAME_KEY)
            if self.has_gripper:
                keys.append(GRIPPER_STATE_OBS_KEY)
            if self.task_has_phases:
                keys.append(PHASE_LABEL_KEY)
            return keys


class ObservationSpace:
    """Defines what observations the task will request and how they are processed."""
    def __init__(
        self,
        use_proprio_base_frame: bool = False,
        use_proprio_camera_frame: bool = False,
        use_gripper_state: bool = False,
        gripper_type: str = GripperType.BINARY.value,
        camera_keys: list[str] = None,
        use_language: bool = False,
        custom_obs_keys: list[str] = None,
    ):
        """Initialize ObservationSpace.

        Args:
            use_proprio_base_frame: Whether to use robot base frame for proprioception
            use_proprio_camera_frame: Whether to use camera frame for proprioception
            use_gripper_state: Whether to include gripper state in observations
            gripper_type: Type of gripper (e.g., binary, continuous)
            camera_keys: List of camera keys to include in observations
            use_language: Whether to include language instructions in observations
            custom_obs_keys: List of custom observation keys to include
        """
        self.use_proprio_base_frame = use_proprio_base_frame
        self.use_proprio_camera_frame = use_proprio_camera_frame
        self.use_proprioceptive_data = use_proprio_base_frame or use_proprio_camera_frame
        self.use_gripper_state = use_gripper_state
        self.gripper_type = gripper_type
        self.camera_keys = camera_keys if camera_keys is not None else []
        self.use_language = use_language
        self.custom_obs_keys = custom_obs_keys if custom_obs_keys is not None else []


    def get_required_zarr_keys(self) -> list[str]:
        """Get all zarr keys needed for this observation space at runtime.

        Returns:
            List of keys to load from replay buffer
        """
        keys = []
        keys.extend(self.camera_keys)
        if self.use_proprio_base_frame:
            keys.append(PROPRIO_OBS_ROBOT_FRAME_KEY)
        if self.use_proprio_camera_frame:
            keys.append(PROPRIO_OBS_CAMERA_FRAME_KEY)
        if self.use_language:
            keys.append(LANGUAGE_KEY)
        if self.use_gripper_state:
            keys.append(GRIPPER_STATE_OBS_KEY)
        for key in self.custom_obs_keys:
            keys.append(key)
        return keys


class TaskSpace:
    """The task space defines what data the experiment will use at runtime."""

    def __init__(
        self,
        dataset_schema: DatasetSchema,
        dataloader: DataLoaderConfig,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        observation_horizon: int = 1,
        prediction_horizon: int = 16,
    ):
        self.dataset_schema = dataset_schema
        self.dataloader = dataloader
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon


    def __post_init__(self):
        """Validate task configuration."""
        if self.observation_horizon < 1:
            raise ValueError(f"observation_horizon must be >= 1, got {self.observation_horizon}")
        if self.prediction_horizon < 1:
            raise ValueError(f"prediction_horizon must be >= 1, got {self.prediction_horizon}")
        if self.action_space.has_orientation:
            valid_ori_dims = {1, 3, 4}
            if self.action_space.orientation_dim not in valid_ori_dims:
                raise ValueError(
                    f"orientation_dim must be one of {valid_ori_dims}:"
                    f"1 for roll of RCM-constrained EE, 3 for euler, 4 for quaternion,"
                    f"got {self.action_space.orientation_dim}"
                )
        if self.observation_space.use_proprioceptive_data and not (self.observation_space.use_proprio_base_frame or self.observation_space.use_proprio_camera_frame):
            raise ValueError(
                "If use_proprioceptive_data is True, then one of"
                "use_proprio_base_frame or use_proprio_camera_frame must be True"
                )
        if self.observation_space.camera_keys:
            for cam in self.observation_space.camera_keys:
                if cam not in VALID_CAMERAS:
                    raise ValueError(f"Invalid camera key '{cam}', must be one of {VALID_CAMERAS}."
                                     f"To add custom camera keys, add them to constants.data.Cameras enum.")
