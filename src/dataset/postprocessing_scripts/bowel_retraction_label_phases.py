"""
Script for postprocessing bowel retraction episodes by labeling phases based on gripper state
and camera frame kinematics data. The phases represent different stages of the surgical procedure.
"""

import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from dataset.preprocess import (
    CAMERA_FRAME_KINEMATICS_COLS,
    GRIPPER_STATE_COL,
    PHASE_LABEL_COL,
)
from dataset.dataloader import EPISODE_FILENAME


def detect_first_desired_gripper_state(
    gripper_state_col, phases, delay_seconds=1, hz=4, phase_number=0, opened=True
):
    """
    Detects the first occurrence of a desired gripper state (opened or closed) and extends the phase list.

    Args:
        gripper_state_col: Array of gripper state values (boolean-like)
        phases: Current list of phase labels
        delay_seconds: Number of seconds to delay after detection
        hz: Sampling frequency in Hz
        phase_number: The phase number to assign
        opened: If True, detect first opened state; if False, detect first closed state

    Returns:
        Extended phases list with the new phase labels
    """
    phases = phases[:]
    start_idx = len(phases)
    if not opened:
        # Invert gripper state to detect closed state (assuming True = opened)
        gripper_state_col[start_idx:] = ~gripper_state_col[start_idx:]
    # Find first occurrence of desired state, or use end of data if none found
    first_desired_gripper_state_idx = (
        np.argmax(gripper_state_col[start_idx:])
        if np.any(gripper_state_col[start_idx:])
        else len(gripper_state_col[start_idx:])
    )
    # Extend phases with delay
    return phases + [phase_number] * (
        first_desired_gripper_state_idx + delay_seconds * hz
    )


def detect_movement_greater_or_smaller_than(
    camera_frame_kinematics_cols,
    phases,
    epsilon=1e-4,
    phase_number=None,
    greater_than=True,
):
    """
    Detects the first occurrence where movement magnitude is greater than or less than a threshold.

    Args:
        camera_frame_kinematics_cols: Array of camera frame kinematics data
        phases: Current list of phase labels
        epsilon: Movement threshold value
        phase_number: The phase number to assign
        greater_than: If True, detect when movement > epsilon; if False, detect when movement <= epsilon

    Returns:
        Extended phases list with the new phase labels
    """
    phases = phases[:]
    start_idx = len(phases)
    camera_frame_kinematics_cols = camera_frame_kinematics_cols[start_idx:]
    # Calculate movement magnitude between consecutive frames
    movement_norm = np.linalg.norm(
        camera_frame_kinematics_cols[1:, :] - camera_frame_kinematics_cols[:-1, :],
        axis=1,
    )
    # Create mask based on threshold condition
    mask = movement_norm >= epsilon if greater_than else movement_norm <= epsilon
    # Find first frame meeting the condition, or use end of data if none found
    first_positive_idx = np.argmax(mask) if np.any(mask) else len(movement_norm)
    return phases + [phase_number] * first_positive_idx


def detect_phase_0(gripper_state_col, phases):
    """
    Phase 0: Initial phase - detects first gripper opening.
    This represents the beginning of the procedure when the robot waits for the robot to point/lift where the grasp is wanted.
    """
    return detect_first_desired_gripper_state(
        gripper_state_col, phases, phase_number=0, opened=True
    )


def detect_phase_1(gripper_state_col, phases):
    """
    Phase 1: Tool preparation - The robot goes to the grasp position and closes the gripper.
    This represents the phase where the gripper is closed to grasp the tool.
    """
    return detect_first_desired_gripper_state(
        gripper_state_col, phases, phase_number=1, opened=False
    )


def detect_phase_2(camera_frame_kinematics_cols, phases, epsilon=1e-4):
    """
    Phase 2: Wait for retraction trigger - The robot holds until the surgeon grasp another end of the bowel.
    """
    return detect_movement_greater_or_smaller_than(
        camera_frame_kinematics_cols,
        phases,
        phase_number=2,
        greater_than=True,
        epsilon=epsilon,
    )


def detect_phase_3(camera_frame_kinematics_cols, phases, epsilon=1e-4):
    """
    Phase 3: Retraction phase - Once the surgeon has grasped the other end of the bowel, the robot retracts the bowel.
    """
    return detect_movement_greater_or_smaller_than(
        camera_frame_kinematics_cols,
        phases,
        phase_number=3,
        greater_than=False,
        epsilon=epsilon,
    )


def detect_phase_4(episode_length, phases):
    """
    Phase 4: Completion - Once the bowel is retracted, the robot holds the grasp without moving.
    """
    phases = phases[:]
    start_idx = len(phases)
    return phases + [4] * (episode_length - start_idx)


def add_phases_to_episode_csv(episode_path, col_name=None):
    """
    Processes a single episode CSV file and adds phase labels based on gripper state and kinematics.

    The phase detection follows this sequence:
    1. Phase 0: Hold for grasp position
    2. Phase 1: Gripper closing for tool grasp
    3. Phase 2: Hold for retraction trigger
    4. Phase 3: Retraction phase
    5. Phase 4: Hold for completion

    Args:
        episode_path: Path to the episode CSV file
        col_name: Name of the column to add phase labels to (defaults to PHASE_LABEL_COL)
    """
    col_name = PHASE_LABEL_COL if col_name is None else col_name
    episode = pd.read_csv(episode_path)
    gripper_state_col = np.array(episode[GRIPPER_STATE_COL])
    camera_frame_kinematics_cols = np.array(episode[CAMERA_FRAME_KINEMATICS_COLS])

    # Initialize empty phases list and detect each phase sequentially
    phases = []
    phases = detect_phase_0(gripper_state_col, phases)
    phases = detect_phase_1(gripper_state_col, phases)
    phases = detect_phase_2(
        camera_frame_kinematics_cols, phases, epsilon=1e-3
    )  # Higher threshold for phase 2
    phases = detect_phase_3(camera_frame_kinematics_cols, phases)
    phases = detect_phase_4(len(episode), phases)

    # Add phase labels to the episode dataframe and save
    episode[col_name] = phases
    episode.to_csv(episode_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add phase labels to bowel retraction episodes"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory containing episode folders",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    # Find all episode directories (excluding metadata files)
    episodes_paths = [
        os.path.join(dataset_path, i, EPISODE_FILENAME)
        for i in os.listdir(dataset_path)
        if i not in ["dataset.zarr", "EADME.md"]
    ]

    # Process each episode with progress bar
    for episode_path in tqdm(episodes_paths, desc="Processing episodes"):
        try:
            add_phases_to_episode_csv(episode_path)
        except Exception as e:
            print(f"Error processing: {episode_path}.\n{e}")
