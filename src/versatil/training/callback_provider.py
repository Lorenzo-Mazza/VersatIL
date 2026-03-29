"""Protocol for components that declare training callbacks."""

from typing import Protocol, runtime_checkable

from pytorch_lightning import Callback

from versatil.configs.experiment import ExperimentConfig


@runtime_checkable
class CallbackProvider(Protocol):
    """Components that provide training callbacks implement this protocol.

    Decoders, algorithms, and loss modules can declare callbacks by adding
    a ``get_callbacks`` method. The workspace collects these at training setup.
    Uses ``@runtime_checkable`` so components don't need to import or inherit
    this protocol — structural subtyping via duck typing.
    """

    def get_callbacks(self, experiment_config: ExperimentConfig) -> list[Callback]: ...
