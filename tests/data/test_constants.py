"""Tests for data constants shared with simulation servers."""

import pytest
from versatil_constants.kitchen import KitchenCamera
from versatil_constants.libero import LiberoCamera
from versatil_constants.metaworld import MetaWorldCamera
from versatil_constants.pusht import PushTCamera
from versatil_constants.tso import TSOCamera

from versatil.data.constants import Cameras


@pytest.mark.parametrize(
    ("constant_camera", "versatil_camera"),
    [
        (TSOCamera.LEFT, Cameras.LEFT),
        (TSOCamera.RIGHT, Cameras.RIGHT),
        (TSOCamera.DEPTH, Cameras.DEPTH),
        (LiberoCamera.AGENTVIEW, Cameras.AGENTVIEW),
        (LiberoCamera.EYE_IN_HAND, Cameras.EYE_IN_HAND),
        (MetaWorldCamera.AGENTVIEW, Cameras.AGENTVIEW),
        (PushTCamera.AGENTVIEW, Cameras.AGENTVIEW),
        (KitchenCamera.AGENTVIEW, Cameras.AGENTVIEW),
    ],
)
def test_production_camera_keys_match_simulation_constants(
    constant_camera,
    versatil_camera,
):
    assert constant_camera.value == versatil_camera.value
