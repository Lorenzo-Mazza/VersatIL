"""Shared structured input specifications for model modules."""

from dataclasses import dataclass, field

from versatil.data.constants import CameraModality


@dataclass
class InputSpecification:
    """Structured input contract for modules that consume observation tensors."""

    keys: str | list[str]
    #: The module needs these input observation keys.
    required: list[str] = field(default_factory=list)
    #: The module needs exactly one camera with each listed modality.
    exactly_one_camera_modality: list[CameraModality] = field(default_factory=list)
    #: The module requires at least one camera for each listed modality and
    #: accepts no camera modalities outside this list.
    required_camera_modalities: list[CameraModality] = field(default_factory=list)
    # For conditional encoders.
    conditioning_key: str | None = None
    conditioning_required: list[str] = field(default_factory=list)
    conditioning_one_of_groups: list[list[str]] = field(default_factory=list)
    # For validating the tokenizer vocabulary against language-consuming modules.
    requires_tokenized: bool = False

    def __post_init__(self) -> None:
        """Normalize keys to a list if a single string is provided."""
        if isinstance(self.keys, str):
            self.keys = [self.keys]

    def validate(self) -> None:
        """Validate structural input constraints."""
        key_set = set(self.keys)
        missing = set(self.required) - key_set
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")
        if self.conditioning_key:
            conditioning_set = {self.conditioning_key}
            missing_conditioning = set(self.conditioning_required) - conditioning_set
            if missing_conditioning:
                raise ValueError(
                    f"Missing required conditioning: {missing_conditioning}"
                )
            for group in self.conditioning_one_of_groups:
                matches = conditioning_set.intersection(group)
                if len(matches) != 1:
                    raise ValueError(
                        f"Exactly one from {group} required for conditioning"
                    )
        self._validate_unique_camera_modalities(
            field_name="exactly_one_camera_modality",
            modalities=self.exactly_one_camera_modality,
        )
        self._validate_unique_camera_modalities(
            field_name="required_camera_modalities",
            modalities=self.required_camera_modalities,
        )

    @staticmethod
    def _validate_unique_camera_modalities(
        field_name: str,
        modalities: list[CameraModality],
    ) -> None:
        """Validate that a camera-modality constraint has no duplicates."""
        modality_values = [modality.value for modality in modalities]
        duplicate_modalities = sorted(
            {
                modality_value
                for modality_value in modality_values
                if modality_values.count(modality_value) > 1
            }
        )
        if duplicate_modalities:
            raise ValueError(
                f"Camera modality constraint '{field_name}' contains duplicate "
                f"modalities: {duplicate_modalities}"
            )
