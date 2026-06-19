"""Tests for versatil.training.stage module."""

import re

import pytest
from omegaconf import OmegaConf

from versatil.training.stage import TrainingStage


@pytest.mark.unit
class TestTrainingStageInit:
    def test_stores_fields_without_defaults(self) -> None:
        stage = TrainingStage(
            name="vae",
            start_epoch=0,
            end_epoch=100,
            trainable_groups=["backbone"],
            frozen_groups=["prior"],
            group_lrs={"backbone": 1e-4},
            group_weight_decays={"backbone": 1e-3},
            loss_weights={"denoising_prior": {"weight": 0.0}},
            eval_frozen_modules=False,
        )
        assert stage.name == "vae"
        assert stage.start_epoch == 0
        assert stage.end_epoch == 100
        assert stage.trainable_groups == ["backbone"]
        assert stage.frozen_groups == ["prior"]
        assert stage.group_lrs == {"backbone": 1e-4}
        assert stage.group_weight_decays == {"backbone": 1e-3}
        assert stage.loss_weights == {"denoising_prior": {"weight": 0.0}}
        assert stage.eval_frozen_modules is False

    def test_defaults_produce_empty_containers(self) -> None:
        stage = TrainingStage(name="vae", start_epoch=0)
        assert stage.end_epoch is None
        assert stage.trainable_groups == []
        assert stage.frozen_groups == []
        assert stage.group_lrs == {}
        assert stage.group_weight_decays == {}
        assert stage.loss_weights == {}
        assert stage.eval_frozen_modules is True

    def test_hydra_loss_weights_become_plain_nested_dicts(self) -> None:
        loss_weights = OmegaConf.create({"denoising_prior": {"weight": 0.0}})

        stage = TrainingStage(
            name="vae",
            start_epoch=0,
            loss_weights=loss_weights,
        )

        stage.loss_weights["denoising_prior"]["weight"] = 0.1
        assert stage.loss_weights == {"denoising_prior": {"weight": 0.1}}

    @pytest.mark.parametrize("end_epoch", [10, 5])
    def test_end_epoch_not_greater_than_start_epoch_raises(self, end_epoch) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "TrainingStage 'stage' end_epoch must be greater than "
                f"start_epoch; got {end_epoch} <= 10."
            ),
        ):
            TrainingStage(name="stage", start_epoch=10, end_epoch=end_epoch)

    def test_conflicting_trainable_and_frozen_groups_raise(self) -> None:
        conflicting_groups = ["decoder"]
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Training stage 'conflict' lists groups in both "
                f"trainable_groups and frozen_groups: {sorted(conflicting_groups)}."
            ),
        ):
            TrainingStage(
                name="conflict",
                start_epoch=0,
                trainable_groups=["prior", "decoder"],
                frozen_groups=["decoder"],
            )

    def test_int_group_weight_decay_value_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "TrainingStage.group_weight_decays values must be floats; "
                f"got {int.__name__} for group 'prior'."
            ),
        ):
            TrainingStage(
                name="stage",
                start_epoch=0,
                group_weight_decays={"prior": 1},
            )

    def test_non_mapping_loss_weights_raises(self) -> None:
        non_mapping = ["denoising_prior"]
        with pytest.raises(
            TypeError,
            match=re.escape(
                "TrainingStage mapping fields must be dictionaries; "
                f"got {type(non_mapping).__name__}."
            ),
        ):
            TrainingStage(
                name="stage",
                start_epoch=0,
                loss_weights=non_mapping,
            )


@pytest.mark.unit
class TestTrainingStageIsActiveAt:
    def test_returns_false_before_start_epoch(self) -> None:
        stage = TrainingStage(name="stage", start_epoch=5)
        assert stage.is_active_at(current_epoch=4) is False

    def test_returns_true_at_start_epoch(self) -> None:
        stage = TrainingStage(name="stage", start_epoch=5)
        assert stage.is_active_at(current_epoch=5) is True

    def test_returns_true_indefinitely_when_no_end_and_no_next(self) -> None:
        stage = TrainingStage(name="stage", start_epoch=0)
        assert stage.is_active_at(current_epoch=10_000) is True

    def test_returns_false_after_explicit_end_epoch(self) -> None:
        stage = TrainingStage(name="stage", start_epoch=0, end_epoch=5)
        assert stage.is_active_at(current_epoch=5) is False
        assert stage.is_active_at(current_epoch=4) is True

    def test_next_stage_start_epoch_acts_as_exclusive_upper_bound(self) -> None:
        stage = TrainingStage(name="vae", start_epoch=0)
        next_stage = TrainingStage(name="prior", start_epoch=400)
        assert stage.is_active_at(current_epoch=399, next_stage=next_stage) is True
        assert stage.is_active_at(current_epoch=400, next_stage=next_stage) is False

    def test_explicit_end_epoch_takes_precedence_over_next_stage(self) -> None:
        stage = TrainingStage(name="vae", start_epoch=0, end_epoch=3)
        next_stage = TrainingStage(name="prior", start_epoch=10)
        assert stage.is_active_at(current_epoch=2, next_stage=next_stage) is True
        # Explicit end_epoch is 3; stage inactive at 3 even though next starts at 10.
        assert stage.is_active_at(current_epoch=3, next_stage=next_stage) is False
