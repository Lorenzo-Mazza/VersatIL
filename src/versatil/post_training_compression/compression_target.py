"""Compression target defining preparation and pruning for a policy submodule."""

from versatil.configs.post_training_compression import PreparationConfig
from versatil.post_training_compression.pruning.base import BasePruner


class CompressionTarget:
    """Stores preparation and pruning components for one PyTorch submodule."""

    def __init__(
        self,
        module_path: str,
        preparation: PreparationConfig | None = None,
        pruning: list[BasePruner] | None = None,
    ) -> None:
        """Initialize compression components.

        Args:
            module_path: Dotted path to the target submodule,
                or empty string for the full policy.
            preparation: BN replacement and fusion settings.
            pruning: Pruning strategies to apply sequentially.
        """
        self.module_path = module_path
        self.preparation = preparation
        self.pruning: list[BasePruner] = pruning or []

    def overlaps(self, other: "CompressionTarget") -> bool:
        """Return whether two targets can select the same submodule.

        Args:
            other: Target to compare against this target.

        Returns:
            Whether either target is root, both targets are the same path, or
            one target is nested under the other.
        """
        if self.module_path == "" or other.module_path == "":
            return True
        return (
            self.module_path == other.module_path
            or self.module_path.startswith(other.module_path + ".")
            or other.module_path.startswith(self.module_path + ".")
        )
