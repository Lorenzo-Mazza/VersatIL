"""Tests for versatil.models.layers.constants module."""
import enum

import pytest

from versatil.models.layers.constants import (
    AttentionDecompositionMode,
    AttentionType,
    Axis,
    ConditioningType,
    PositionalEncodingType,
)


EXPECTED_ATTENTION_DECOMPOSITION_MEMBERS = {
    "FULL": "full",
    "SEPARABLE": "separable",
}

EXPECTED_AXIS_MEMBERS = {
    "HEIGHT": "height",
    "WIDTH": "width",
}

EXPECTED_POSITIONAL_ENCODING_MEMBERS = {
    "SINUSOIDAL": "sinusoidal",
    "LEARNED": "learned",
    "ROPE": "rope",
}

EXPECTED_ATTENTION_TYPE_MEMBERS = {
    "MULTI_HEAD": "mha",
    "GROUPED_QUERY": "gqa",
}

EXPECTED_CONDITIONING_TYPE_MEMBERS = {
    "ADALN": "adaln",
    "FILM": "film",
}


ALL_ENUMS_WITH_EXPECTED_MEMBERS = [
    (AttentionDecompositionMode, EXPECTED_ATTENTION_DECOMPOSITION_MEMBERS),
    (Axis, EXPECTED_AXIS_MEMBERS),
    (PositionalEncodingType, EXPECTED_POSITIONAL_ENCODING_MEMBERS),
    (AttentionType, EXPECTED_ATTENTION_TYPE_MEMBERS),
    (ConditioningType, EXPECTED_CONDITIONING_TYPE_MEMBERS),
]


class TestAttentionDecompositionMode:

    def test_has_exact_expected_members(self):
        member_names = {member.name for member in AttentionDecompositionMode}
        assert member_names == set(EXPECTED_ATTENTION_DECOMPOSITION_MEMBERS.keys())

    @pytest.mark.parametrize(
        "name, value",
        list(EXPECTED_ATTENTION_DECOMPOSITION_MEMBERS.items()),
    )
    def test_member_values(self, name: str, value: str):
        member = AttentionDecompositionMode[name]
        assert member.value == value

    @pytest.mark.parametrize("member", list(AttentionDecompositionMode))
    def test_members_are_usable_as_strings(self, member: AttentionDecompositionMode):
        assert member == member.value


class TestAxis:

    def test_has_exact_expected_members(self):
        member_names = {member.name for member in Axis}
        assert member_names == set(EXPECTED_AXIS_MEMBERS.keys())

    @pytest.mark.parametrize("name, value", list(EXPECTED_AXIS_MEMBERS.items()))
    def test_member_values(self, name: str, value: str):
        member = Axis[name]
        assert member.value == value

    @pytest.mark.parametrize("member", list(Axis))
    def test_members_are_usable_as_strings(self, member: Axis):
        assert member == member.value


class TestPositionalEncodingType:

    def test_has_exact_expected_members(self):
        member_names = {member.name for member in PositionalEncodingType}
        assert member_names == set(EXPECTED_POSITIONAL_ENCODING_MEMBERS.keys())

    @pytest.mark.parametrize(
        "name, value",
        list(EXPECTED_POSITIONAL_ENCODING_MEMBERS.items()),
    )
    def test_member_values(self, name: str, value: str):
        member = PositionalEncodingType[name]
        assert member.value == value

    @pytest.mark.parametrize("member", list(PositionalEncodingType))
    def test_members_are_usable_as_strings(self, member: PositionalEncodingType):
        assert member == member.value


class TestAttentionType:

    def test_has_exact_expected_members(self):
        member_names = {member.name for member in AttentionType}
        assert member_names == set(EXPECTED_ATTENTION_TYPE_MEMBERS.keys())

    @pytest.mark.parametrize(
        "name, value",
        list(EXPECTED_ATTENTION_TYPE_MEMBERS.items()),
    )
    def test_member_values(self, name: str, value: str):
        member = AttentionType[name]
        assert member.value == value

    @pytest.mark.parametrize("member", list(AttentionType))
    def test_members_are_usable_as_strings(self, member: AttentionType):
        assert member == member.value


class TestConditioningType:

    def test_has_exact_expected_members(self):
        member_names = {member.name for member in ConditioningType}
        assert member_names == set(EXPECTED_CONDITIONING_TYPE_MEMBERS.keys())

    @pytest.mark.parametrize(
        "name, value",
        list(EXPECTED_CONDITIONING_TYPE_MEMBERS.items()),
    )
    def test_member_values(self, name: str, value: str):
        member = ConditioningType[name]
        assert member.value == value

    @pytest.mark.parametrize("member", list(ConditioningType))
    def test_members_are_usable_as_strings(self, member: ConditioningType):
        assert member == member.value


class TestAllEnumsConsistency:

    @pytest.mark.parametrize(
        "enum_class, expected_members",
        ALL_ENUMS_WITH_EXPECTED_MEMBERS,
        ids=[cls.__name__ for cls, _ in ALL_ENUMS_WITH_EXPECTED_MEMBERS],
    )
    def test_all_enums_are_str_enums(self, enum_class, expected_members):
        assert issubclass(enum_class, str)
        assert issubclass(enum_class, enum.Enum)

    @pytest.mark.parametrize(
        "enum_class, expected_members",
        ALL_ENUMS_WITH_EXPECTED_MEMBERS,
        ids=[cls.__name__ for cls, _ in ALL_ENUMS_WITH_EXPECTED_MEMBERS],
    )
    def test_lookup_by_string_value(self, enum_class, expected_members):
        for name, value in expected_members.items():
            assert enum_class(value).name == name
