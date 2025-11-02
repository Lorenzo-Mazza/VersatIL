"""Configuration for the dataset schema. The dataset schema defines the content of a dataset, but does not define
   what data is used at runtime (see TaskConfig for that).
   To add a new dataset schema, subclass data.schemas.base.DatasetSchema and implement the required methods.
 """

from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DatasetSchemaConfig:
    _target_: str = MISSING
    dataset_folders: list[str] = MISSING
    zarr_path: str = MISSING
