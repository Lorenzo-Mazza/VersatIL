"""Tests for versatil.post_training_compression.deployment_backends.base module."""

import re

import pytest

from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)


@pytest.mark.unit
def test_deployment_backend_requires_export_implementation():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Can't instantiate abstract class DeploymentBackend without an "
            "implementation for abstract method 'export'"
        ),
    ):
        DeploymentBackend()
