from dataclasses import dataclass


@dataclass
class ImagePathConfig:
    """Configuration for image paths in CSV.

    This is a dataclass that can be instantiated via Hydra.
    """
    left_image_key: str = "frameLeftPath"
    right_image_key: str = "frameRightPath"
    rectified_left_image_key: str = "frameLeftRectifiedPath"
    rectified_right_image_key: str = "frameRightRectifiedPath"

    # For computing depth
    depth_dir_pattern: str = "depth"
    depth_file_pattern: str = r'depth_\1.npy'
    left_dir_pattern: str = "framesLeft"
    rectified_left_dir_pattern: str = "framesLeftRectified"
    rgb_extension: str = ".png"
