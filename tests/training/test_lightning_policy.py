"""Tests for versatil.training.lightning_policy module."""

import time
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.configs.training import (
    AdamWConfig,
    ParameterGroupConfig,
)
from versatil.training.constants import OPTIMIZER_UNMATCHED_GROUPS_NAME


@pytest.mark.unit
class TestLightningPolicyInitialization:
    def test_stores_policy_reference(
        self,
        mock_policy_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        lightning_policy = lightning_policy_factory(policy=policy)

        assert lightning_policy.policy is policy

    def test_stores_training_config(
        self,
        training_config_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        config = training_config_factory(num_epochs=42)
        lightning_policy = lightning_policy_factory(training_config=config)

        assert lightning_policy.training_config is config

    def test_stores_total_training_steps(
        self,
        lightning_policy_factory: Callable,
    ):
        lightning_policy = lightning_policy_factory(total_training_steps=5000)

        assert lightning_policy.total_training_steps == 5000

    def test_initializes_metrics_accumulators_at_zero(
        self,
        lightning_policy_factory: Callable,
    ):
        lightning_policy = lightning_policy_factory()

        assert lightning_policy.train_metrics.num_batches == 0
        assert lightning_policy.val_metrics.num_batches == 0
        assert lightning_policy.train_metrics.total_loss == 0.0
        assert lightning_policy.val_metrics.total_loss == 0.0

    def test_initializes_dataloaders_as_none(
        self,
        lightning_policy_factory: Callable,
    ):
        lightning_policy = lightning_policy_factory()

        assert lightning_policy._train_dataloader is None
        assert lightning_policy._val_dataloader is None


@pytest.mark.unit
class TestTrainingStep:
    def test_calls_policy_compute_loss_with_batch(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
        rng: np.random.Generator,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=1.5)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)
        batch = {
            "observations": {
                "left": torch.from_numpy(
                    rng.standard_normal((2, 3, 64, 64)).astype(np.float32)
                )
            }
        }

        lightning_policy.training_step(batch=batch, batch_idx=0)

        policy.compute_loss.assert_called_once_with(batch)

    def test_returns_total_loss_tensor(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        expected_loss_value = 2.3
        loss_output = loss_output_factory(total_loss_value=expected_loss_value)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)
        batch = {"observations": {}}

        result = lightning_policy.training_step(batch=batch, batch_idx=0)

        assert torch.isclose(result, torch.tensor(expected_loss_value), atol=1e-6)

    def test_accumulates_metrics_across_batches(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        lightning_policy = lightning_policy_factory(policy=policy)

        for step in range(3):
            loss_output = loss_output_factory(total_loss_value=float(step + 1))
            policy.compute_loss.return_value = loss_output
            lightning_policy.training_step(batch={}, batch_idx=step)

        assert lightning_policy.train_metrics.num_batches == 3
        # total_loss accumulates: 1.0 + 2.0 + 3.0 = 6.0
        assert abs(lightning_policy.train_metrics.total_loss - 6.0) < 1e-5

    def test_logs_train_loss_on_epoch(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=0.7)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)

        with patch.object(lightning_policy, "log") as mock_log:
            lightning_policy.training_step(batch={}, batch_idx=0)

            mock_log.assert_called_once_with(
                "train_loss",
                loss_output.total_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )


@pytest.mark.unit
class TestValidationStep:
    def test_calls_policy_compute_loss_with_batch(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
        rng: np.random.Generator,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=0.8)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)
        batch = {
            "observations": {
                "right": torch.from_numpy(
                    rng.standard_normal((2, 3, 64, 64)).astype(np.float32)
                )
            }
        }

        lightning_policy.validation_step(batch=batch, batch_idx=0)

        policy.compute_loss.assert_called_once_with(batch)

    def test_returns_total_loss_tensor(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        expected_loss_value = 1.1
        loss_output = loss_output_factory(total_loss_value=expected_loss_value)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)

        result = lightning_policy.validation_step(batch={}, batch_idx=0)

        assert torch.isclose(result, torch.tensor(expected_loss_value), atol=1e-6)

    def test_accumulates_val_metrics_not_train(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=0.5)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)
        lightning_policy.validation_step(batch={}, batch_idx=0)

        assert lightning_policy.val_metrics.num_batches == 1
        assert lightning_policy.train_metrics.num_batches == 0

    def test_logs_val_loss(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=0.42)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)

        with patch.object(lightning_policy, "log") as mock_log:
            lightning_policy.validation_step(batch={}, batch_idx=0)

            mock_log.assert_called_once_with(
                "val_loss",
                loss_output.total_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )


@pytest.mark.unit
def test_on_train_epoch_start_records_epoch_start_time(
    lightning_policy_factory: Callable,
):
    lightning_policy = lightning_policy_factory()

    before = time.monotonic()
    lightning_policy.on_train_epoch_start()
    after = time.monotonic()

    assert before <= lightning_policy._epoch_start_time <= after


@pytest.mark.unit
class TestOnTrainEpochEnd:
    def test_resets_train_metrics_after_logging(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        lightning_policy = lightning_policy_factory(policy=policy)

        for step in range(3):
            loss_output = loss_output_factory(
                total_loss_value=1.0,
                component_losses={"mse": 0.5},
            )
            policy.compute_loss.return_value = loss_output
            lightning_policy.training_step(batch={}, batch_idx=step)

        assert lightning_policy.train_metrics.num_batches == 3

        with (
            patch.object(lightning_policy, "log_dict"),
            patch.object(lightning_policy, "log"),
        ):
            lightning_policy.on_train_epoch_end()

        assert lightning_policy.train_metrics.num_batches == 0
        assert lightning_policy.train_metrics.total_loss == 0.0

    def test_logs_metrics_with_train_prefix(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(
            total_loss_value=2.0,
            component_losses={"mse": 1.0},
        )
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)
        lightning_policy.training_step(batch={}, batch_idx=0)

        with (
            patch.object(lightning_policy, "log_dict") as mock_log_dict,
            patch.object(lightning_policy, "log"),
        ):
            lightning_policy.on_train_epoch_end()

            logged_metrics = mock_log_dict.call_args[0][0]
            for key in logged_metrics:
                assert key.startswith("train/")

    def test_logs_epoch_time_seconds(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=1.0)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)
        lightning_policy.on_train_epoch_start()
        lightning_policy.training_step(batch={}, batch_idx=0)

        with (
            patch.object(lightning_policy, "log_dict"),
            patch.object(lightning_policy, "log") as mock_log,
        ):
            lightning_policy.on_train_epoch_end()

            epoch_time_calls = [
                call
                for call in mock_log.call_args_list
                if call[0][0] == "train/epoch_time_seconds"
            ]
            assert len(epoch_time_calls) == 1
            logged_duration = epoch_time_calls[0][0][1]
            assert logged_duration >= 0.0

    @pytest.mark.requires_gpu
    def test_logs_gpu_memory_peak_on_cuda(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=1.0)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)
        lightning_policy.on_train_epoch_start()
        lightning_policy.training_step(batch={}, batch_idx=0)

        cuda_device = torch.device("cuda")
        with (
            patch.object(
                type(lightning_policy),
                "device",
                new_callable=lambda: property(lambda s: cuda_device),
            ),
            patch.object(lightning_policy, "log_dict"),
            patch.object(lightning_policy, "log") as mock_log,
        ):
            lightning_policy.on_train_epoch_end()

            memory_calls = [
                call
                for call in mock_log.call_args_list
                if call[0][0] == "train/gpu_memory_peak_gb"
            ]
            assert len(memory_calls) == 1
            assert memory_calls[0][0][1] >= 0.0

    def test_skips_gpu_memory_logging_on_cpu(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=1.0)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)
        lightning_policy.on_train_epoch_start()
        lightning_policy.training_step(batch={}, batch_idx=0)

        with (
            patch.object(lightning_policy, "log_dict"),
            patch.object(lightning_policy, "log") as mock_log,
        ):
            lightning_policy.on_train_epoch_end()

            memory_calls = [
                call for call in mock_log.call_args_list if "gpu_memory" in call[0][0]
            ]
            assert len(memory_calls) == 0


@pytest.mark.unit
class TestOnValidationEpochEnd:
    def test_resets_val_metrics_after_logging(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=0.5)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)

        for step in range(2):
            lightning_policy.validation_step(batch={}, batch_idx=step)

        assert lightning_policy.val_metrics.num_batches == 2

        with patch.object(lightning_policy, "log_dict"):
            lightning_policy.on_validation_epoch_end()

        assert lightning_policy.val_metrics.num_batches == 0

    def test_logs_metrics_with_val_prefix(
        self,
        mock_policy_factory: Callable,
        loss_output_factory: Callable,
        lightning_policy_factory: Callable,
    ):
        policy = mock_policy_factory()
        loss_output = loss_output_factory(total_loss_value=0.8)
        policy.compute_loss.return_value = loss_output

        lightning_policy = lightning_policy_factory(policy=policy)
        lightning_policy.validation_step(batch={}, batch_idx=0)

        with patch.object(lightning_policy, "log_dict") as mock_log_dict:
            lightning_policy.on_validation_epoch_end()

            logged_metrics = mock_log_dict.call_args[0][0]
            for key in logged_metrics:
                assert key.startswith("val/")


@pytest.mark.unit
class TestForward:
    def test_delegates_to_policy_predict_action(
        self,
        mock_policy_factory: Callable,
        lightning_policy_factory: Callable,
        rng: np.random.Generator,
    ):
        policy = mock_policy_factory()
        expected_output = {
            "actions": torch.from_numpy(
                rng.standard_normal((2, 10, 7)).astype(np.float32)
            )
        }
        policy.predict_action.return_value = expected_output

        lightning_policy = lightning_policy_factory(policy=policy)
        obs_dict = {
            "left": torch.from_numpy(
                rng.standard_normal((2, 3, 64, 64)).astype(np.float32)
            )
        }

        result = lightning_policy(obs_dict)

        policy.predict_action.assert_called_once_with(obs_dict)
        assert result is expected_output


@pytest.mark.unit
class TestConfigureOptimizers:
    def test_returns_optimizer_without_scheduler_when_no_lr_schedule(
        self,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ):
        config = training_config_factory(lr_schedule=None)
        lightning_policy = lightning_policy_factory(training_config=config)

        result = lightning_policy.configure_optimizers()

        assert "optimizer" in result
        assert "lr_scheduler" not in result
        assert isinstance(result["optimizer"], torch.optim.AdamW)

    def test_returns_optimizer_with_scheduler_when_lr_schedule_set(
        self,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
        mock_trainer_factory: Callable,
    ):
        config = training_config_factory(lr_schedule="cosine", lr_warmup_steps=50)
        lightning_policy = lightning_policy_factory(
            training_config=config,
            total_training_steps=1000,
        )

        result = lightning_policy.configure_optimizers()

        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert "scheduler" in result["lr_scheduler"]
        assert result["lr_scheduler"]["interval"] == "step"
        assert result["lr_scheduler"]["frequency"] == 1
        assert result["lr_scheduler"]["name"] == "learning_rate"

    def test_uses_total_training_steps_when_provided(
        self,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ):
        config = training_config_factory(lr_schedule="linear", lr_warmup_steps=10)
        total_steps = 500
        lightning_policy = lightning_policy_factory(
            training_config=config,
            total_training_steps=total_steps,
        )

        result = lightning_policy.configure_optimizers()

        # The scheduler should have been created with the provided total_steps
        assert "lr_scheduler" in result

    def test_falls_back_to_trainer_estimated_steps(
        self,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
        mock_trainer_factory: Callable,
    ):
        config = training_config_factory(lr_schedule="cosine", lr_warmup_steps=10)
        lightning_policy = lightning_policy_factory(
            training_config=config,
            total_training_steps=None,
        )
        # Attach mock trainer so estimated_stepping_batches is available
        lightning_policy._trainer = mock_trainer_factory(
            estimated_stepping_batches=2000
        )

        result = lightning_policy.configure_optimizers()

        assert "lr_scheduler" in result

    def test_cosine_with_min_lr_passes_scheduler_kwargs(
        self,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ):
        min_lr = 2.5e-6
        peak_lr = 2.5e-5
        config = training_config_factory(
            lr_schedule="cosine_with_min_lr",
            lr_warmup_steps=10,
            lr_scheduler_kwargs={"min_lr": min_lr},
            optimizer=AdamWConfig(lr=peak_lr),
        )
        lightning_policy = lightning_policy_factory(
            training_config=config,
            total_training_steps=1000,
        )

        result = lightning_policy.configure_optimizers()

        assert "lr_scheduler" in result
        scheduler = result["lr_scheduler"]["scheduler"]
        # Step to the end to verify min_lr floor
        for _ in range(1000):
            scheduler.step()
        actual_lr = result["optimizer"].param_groups[0]["lr"]
        assert actual_lr == pytest.approx(min_lr, rel=0.1)

    def test_empty_scheduler_kwargs_passes_none(
        self,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ):
        config = training_config_factory(
            lr_schedule="cosine",
            lr_warmup_steps=10,
            lr_scheduler_kwargs={},
        )
        lightning_policy = lightning_policy_factory(
            training_config=config,
            total_training_steps=500,
        )

        result = lightning_policy.configure_optimizers()

        assert "lr_scheduler" in result

    def test_optimizer_receives_correct_learning_rate(
        self,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ):
        learning_rate = 3e-5
        optimizer_config = AdamWConfig(lr=learning_rate)
        config = training_config_factory(optimizer=optimizer_config)
        lightning_policy = lightning_policy_factory(training_config=config)

        result = lightning_policy.configure_optimizers()
        optimizer = result["optimizer"]

        assert optimizer.param_groups[0]["lr"] == learning_rate


@pytest.mark.unit
class TestCreateParameterGroups:
    def test_single_group_when_no_param_groups_configured(
        self,
        mock_policy_factory: Callable,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ):
        policy = mock_policy_factory()
        config = training_config_factory()
        lightning_policy = lightning_policy_factory(
            policy=policy, training_config=config
        )

        groups = lightning_policy._create_parameter_groups(config.optimizer)

        assert len(groups) == 1
        assert groups[0]["name"] == OPTIMIZER_UNMATCHED_GROUPS_NAME
        assert groups[0]["lr"] == config.optimizer.lr
        assert "params" in groups[0]

    def test_assigns_parameters_to_groups_by_pattern(
        self,
        rng: np.random.Generator,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ):
        # Create a policy mock with named parameters that match patterns
        encoder_weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4, 4)).astype(np.float32))
        )
        decoder_weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4, 4)).astype(np.float32))
        )
        other_weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        )

        policy = MagicMock()
        policy.named_parameters.return_value = iter(
            [
                ("encoder.layer.weight", encoder_weight),
                ("decoder.layer.weight", decoder_weight),
                ("head.bias", other_weight),
            ]
        )
        policy.parameters.return_value = iter(
            [encoder_weight, decoder_weight, other_weight]
        )

        param_groups = [
            ParameterGroupConfig(
                name="encoder",
                lr=1e-5,
                params_pattern=r"encoder\.",
            ),
            ParameterGroupConfig(
                name="decoder",
                lr=1e-4,
                params_pattern=r"decoder\.",
            ),
        ]
        optimizer_config = AdamWConfig(lr=1e-3, param_groups=param_groups)
        config = training_config_factory(optimizer=optimizer_config)
        lightning_policy = lightning_policy_factory(
            policy=policy, training_config=config
        )

        groups = lightning_policy._create_parameter_groups(config.optimizer)

        # Should have: default group (head.bias) + encoder group + decoder group
        assert len(groups) == 3

        # Default group has unmatched params
        default_group = groups[0]
        assert default_group["name"] == OPTIMIZER_UNMATCHED_GROUPS_NAME
        assert len(list(default_group["params"])) == 1

        # Encoder group
        encoder_group = groups[1]
        assert encoder_group["name"] == "encoder"
        assert encoder_group["lr"] == 1e-5

        # Decoder group
        decoder_group = groups[2]
        assert decoder_group["name"] == "decoder"
        assert decoder_group["lr"] == 1e-4

    def test_parameter_group_patterns_can_match_nested_substrings(
        self,
        rng: np.random.Generator,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ) -> None:
        encoder_weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        )
        nested_encoder_weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        )

        policy = MagicMock()
        policy.named_parameters.return_value = iter(
            [
                ("encoder.layer.weight", encoder_weight),
                ("nested.encoder.layer.weight", nested_encoder_weight),
            ]
        )
        policy.parameters.return_value = iter([encoder_weight, nested_encoder_weight])

        param_groups = [
            ParameterGroupConfig(
                name="encoder",
                lr=1e-5,
                params_pattern=r"encoder\.",
            ),
        ]
        optimizer_config = AdamWConfig(lr=1e-3, param_groups=param_groups)
        config = training_config_factory(optimizer=optimizer_config)
        lightning_policy = lightning_policy_factory(
            policy=policy, training_config=config
        )

        groups = lightning_policy._create_parameter_groups(config.optimizer)

        assert len(groups) == 2
        assert groups[0]["name"] == OPTIMIZER_UNMATCHED_GROUPS_NAME
        assert groups[0]["params"] == []
        assert groups[1]["name"] == "encoder"
        assert groups[1]["lr"] == 1e-5
        assert [id(parameter) for parameter in groups[1]["params"]] == [
            id(encoder_weight),
            id(nested_encoder_weight),
        ]

    def test_includes_frozen_parameters_for_later_stage_unfreezing(
        self,
        rng: np.random.Generator,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ):
        trainable = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        )
        frozen = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4,)).astype(np.float32)),
            requires_grad=False,
        )

        policy = MagicMock()
        policy.named_parameters.return_value = iter(
            [
                ("trainable.weight", trainable),
                ("frozen.weight", frozen),
            ]
        )
        policy.parameters.return_value = iter([trainable, frozen])

        param_groups = [
            ParameterGroupConfig(
                name="frozen_group",
                lr=1e-5,
                params_pattern=r"frozen\.",
            ),
        ]
        optimizer_config = AdamWConfig(lr=1e-3, param_groups=param_groups)
        config = training_config_factory(optimizer=optimizer_config)
        lightning_policy = lightning_policy_factory(
            policy=policy, training_config=config
        )

        groups = lightning_policy._create_parameter_groups(config.optimizer)

        total_params = sum(len(list(g["params"])) for g in groups)
        assert total_params == 2
        assert groups[0]["name"] == OPTIMIZER_UNMATCHED_GROUPS_NAME
        assert groups[1]["name"] == "frozen_group"
        assert groups[1]["params"] == [frozen]

    def test_includes_weight_decay_when_specified(
        self,
        rng: np.random.Generator,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ):
        weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        )

        policy = MagicMock()
        policy.named_parameters.return_value = iter(
            [
                ("encoder.weight", weight),
            ]
        )
        policy.parameters.return_value = iter([weight])

        param_groups = [
            ParameterGroupConfig(
                name="encoder",
                lr=1e-5,
                weight_decay=0.01,
                params_pattern=r"encoder\.",
            ),
        ]
        optimizer_config = AdamWConfig(lr=1e-3, param_groups=param_groups)
        config = training_config_factory(optimizer=optimizer_config)
        lightning_policy = lightning_policy_factory(
            policy=policy, training_config=config
        )

        groups = lightning_policy._create_parameter_groups(config.optimizer)

        encoder_group = next(g for g in groups if g["name"] == "encoder")
        assert encoder_group["weight_decay"] == 0.01

    def test_empty_configured_param_group_raises(
        self,
        rng: np.random.Generator,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ) -> None:
        weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        )
        policy = MagicMock()
        policy.named_parameters.return_value = iter([("encoder.weight", weight)])
        policy.parameters.return_value = iter([weight])
        optimizer_config = AdamWConfig(
            lr=1e-3,
            param_groups=[
                ParameterGroupConfig(
                    name="decoder",
                    lr=1e-5,
                    params_pattern=r"decoder\.",
                )
            ],
        )
        config = training_config_factory(optimizer=optimizer_config)
        lightning_policy = lightning_policy_factory(
            policy=policy,
            training_config=config,
        )

        with pytest.raises(ValueError, match="matched zero parameters"):
            lightning_policy._create_parameter_groups(config.optimizer)

    def test_longest_match_wins_for_overlapping_patterns(
        self,
        rng: np.random.Generator,
        lightning_policy_factory: Callable,
        training_config_factory: Callable,
    ) -> None:
        head_weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        )
        body_weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        )
        policy = MagicMock()
        policy.named_parameters.return_value = iter(
            [
                ("decoder.head.weight", head_weight),
                ("decoder.body.weight", body_weight),
            ]
        )
        policy.parameters.return_value = iter([head_weight, body_weight])
        optimizer_config = AdamWConfig(
            lr=1e-3,
            param_groups=[
                ParameterGroupConfig(
                    name="decoder",
                    lr=1e-4,
                    params_pattern=r"^decoder\.",
                ),
                ParameterGroupConfig(
                    name="head",
                    lr=1e-5,
                    params_pattern=r"^decoder\.head\.",
                ),
            ],
        )
        config = training_config_factory(optimizer=optimizer_config)
        lightning_policy = lightning_policy_factory(
            policy=policy,
            training_config=config,
        )

        groups = lightning_policy._create_parameter_groups(config.optimizer)

        decoder_group = next(g for g in groups if g["name"] == "decoder")
        head_group = next(g for g in groups if g["name"] == "head")
        assert [id(p) for p in decoder_group["params"]] == [id(body_weight)]
        assert [id(p) for p in head_group["params"]] == [id(head_weight)]


@pytest.mark.unit
class TestDataloaderAccessors:
    def test_train_dataloader_returns_stored_value(
        self,
        lightning_policy_factory: Callable,
    ):
        lightning_policy = lightning_policy_factory()
        mock_loader = MagicMock()
        lightning_policy._train_dataloader = mock_loader

        assert lightning_policy.train_dataloader() is mock_loader

    def test_val_dataloader_returns_stored_value(
        self,
        lightning_policy_factory: Callable,
    ):
        lightning_policy = lightning_policy_factory()
        mock_loader = MagicMock()
        lightning_policy._val_dataloader = mock_loader

        assert lightning_policy.val_dataloader() is mock_loader

    def test_val_dataloader_returns_none_when_not_set(
        self,
        lightning_policy_factory: Callable,
    ):
        lightning_policy = lightning_policy_factory()

        assert lightning_policy.val_dataloader() is None
