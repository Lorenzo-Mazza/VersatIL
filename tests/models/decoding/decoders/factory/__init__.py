"""Tests for decoder factory modules.

These tests use real components (transformers, action heads, positional encodings) rather than
mocks because the factory decoders are lightweight compositions with small dimensions and tiny
batch sizes. Forward passes take milliseconds, and using real components tests the actual wiring
that each factory is responsible for.
"""
