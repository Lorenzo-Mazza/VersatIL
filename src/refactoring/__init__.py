"""Surg-IL library."""
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
)