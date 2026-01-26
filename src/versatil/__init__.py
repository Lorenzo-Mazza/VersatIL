"""Surg-IL library."""
import warnings
import debugpy

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
)
   
# Listen on all interfaces
#debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")
#debugpy.wait_for_client()
print("Debugger attached!")
