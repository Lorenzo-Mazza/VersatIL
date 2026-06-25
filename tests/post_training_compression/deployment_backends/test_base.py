"""Tests for versatil.post_training_compression.deployment_backends.base module."""

import pytest

from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)


@pytest.mark.unit
def test_deployment_backend_requires_export_implementation():
    with pytest.raises(TypeError):
        DeploymentBackend()
