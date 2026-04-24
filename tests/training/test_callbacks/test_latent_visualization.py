"""Tests for versatil.training.callbacks.latent_visualization module."""

from collections.abc import Callable, Generator
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from versatil.metrics.accumulators import LatentVisualizationData
from versatil.metrics.constants import MetadataKey
from versatil.training.callbacks.latent_visualization import LatentVisualizationCallback


@pytest.fixture
def latent_data_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    def factory(
        num_samples: int = 12,
        latent_dimension: int = 4,
    ) -> np.ndarray:
        return rng.standard_normal((num_samples, latent_dimension)).astype(np.float32)

    return factory


@pytest.fixture
def label_array_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    def factory(
        num_samples: int = 12,
        num_labels: int = 3,
    ) -> np.ndarray:
        return rng.integers(0, num_labels, size=num_samples).astype(np.int64)

    return factory


@pytest.fixture(autouse=True)
def mock_tsne_backend() -> Generator[None]:
    class _TSNE:
        def __init__(
            self,
            n_components: int,
            random_state: int,
            perplexity: int,
        ) -> None:
            self.n_components = n_components
            self.random_state = random_state
            self.perplexity = perplexity

        def fit_transform(self, z: np.ndarray) -> np.ndarray:
            return np.zeros((z.shape[0], self.n_components), dtype=np.float32)

    with patch(
        "versatil.training.callbacks.latent_visualization.TSNE",
        _TSNE,
    ):
        yield


@pytest.fixture
def mock_latent_pl_module_factory(
    latent_data_factory: Callable[..., np.ndarray],
    label_array_factory: Callable[..., np.ndarray],
) -> Callable[..., MagicMock]:
    def factory(
        posterior_latent: np.ndarray | None = ...,
        prior_latent: np.ndarray | None = ...,
        labels: dict[str, np.ndarray] | None = ...,
        metadata: dict | None = None,
        latent_dimension: int = 4,
    ) -> MagicMock:
        if posterior_latent is ...:
            posterior_latent = latent_data_factory(latent_dimension=latent_dimension)
        if prior_latent is ...:
            prior_latent = latent_data_factory(latent_dimension=latent_dimension)
        if labels is ...:
            labels = {
                MetadataKey.LATENT_COLOR_LABEL.value: label_array_factory(),
            }
        latent_data = LatentVisualizationData(
            posterior=posterior_latent,
            prior=prior_latent,
            labels=labels or {},
        )
        pl_module = MagicMock()
        for accumulator_name in ("train_metrics", "val_metrics"):
            accumulator = getattr(pl_module, accumulator_name)
            accumulator.compute_latent_visualization_data.return_value = latent_data
            accumulator.metadata = metadata if metadata is not None else {}
        return pl_module

    return factory


@pytest.mark.unit
class TestLatentVisualizationCallback:
    @pytest.mark.parametrize("log_every_n_epochs", [1, 10])
    @pytest.mark.parametrize("max_samples", [100, 5000])
    @pytest.mark.parametrize(
        "label_keys",
        [
            [MetadataKey.LATENT_COLOR_LABEL.value],
            ["task_id", "style_id"],
        ],
    )
    def test_stores_configuration(
        self,
        log_every_n_epochs: int,
        max_samples: int,
        label_keys: list[str],
    ) -> None:
        callback = LatentVisualizationCallback(
            log_every_n_epochs=log_every_n_epochs,
            max_samples=max_samples,
            label_keys=label_keys,
        )

        assert callback.log_every_n_epochs == log_every_n_epochs
        assert callback.max_samples == max_samples
        assert callback.label_keys == label_keys

    def test_default_label_keys_include_synthetic_and_phase_labels(self) -> None:
        callback = LatentVisualizationCallback()

        assert callback.label_keys == [
            MetadataKey.LATENT_COLOR_LABEL.value,
            MetadataKey.PHASE_LABEL.value,
        ]

    @pytest.mark.parametrize(
        "hook, accumulator_name",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_skips_logging_on_non_matching_epochs(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        hook: str,
        accumulator_name: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=5)
        pl_module = MagicMock()
        trainer = mock_trainer_factory(current_epoch=3)

        getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        accumulator = getattr(pl_module, accumulator_name)
        accumulator.compute_latent_visualization_data.assert_not_called()

    @pytest.mark.parametrize(
        "hook, accumulator_name",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_skips_logging_when_no_latent_data(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        hook: str,
        accumulator_name: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=1)
        pl_module = MagicMock()
        accumulator = getattr(pl_module, accumulator_name)
        accumulator.compute_latent_visualization_data.return_value = (
            LatentVisualizationData(posterior=None, prior=None)
        )
        trainer = mock_trainer_factory(current_epoch=0)

        getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "hook, split",
    [
        ("on_train_epoch_end", "train"),
        ("on_validation_epoch_end", "val"),
    ],
)
class TestLatentVisualizationCallbackEpochEnd:
    def test_requests_configured_labels_from_accumulator(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        mock_latent_pl_module_factory: Callable[..., MagicMock],
        hook: str,
        split: str,
    ) -> None:
        label_keys = ["task_id", "style_id"]
        callback = LatentVisualizationCallback(
            log_every_n_epochs=1,
            max_samples=100,
            label_keys=label_keys,
        )
        pl_module = mock_latent_pl_module_factory(labels={})
        trainer = mock_trainer_factory(current_epoch=0)

        with patch(
            "versatil.training.callbacks.latent_visualization.figure_to_wandb_image"
        ):
            getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        accumulator_name = f"{split}_metrics"
        accumulator = getattr(pl_module, accumulator_name)
        accumulator.compute_latent_visualization_data.assert_called_once_with(
            label_keys=label_keys
        )

    def test_logs_labeled_posterior_and_prior_figures(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        mock_latent_pl_module_factory: Callable[..., MagicMock],
        hook: str,
        split: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory()
        trainer = mock_trainer_factory(current_epoch=0)

        with patch(
            "versatil.training.callbacks.latent_visualization.figure_to_wandb_image"
        ):
            getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_called_once()
        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        expected_keys = {
            f"{split}_posterior_latent_space_tsne_by_latent_color_label",
            f"{split}_posterior_latent_space_pca_by_latent_color_label",
            f"{split}_posterior_pca_explained_variance",
            f"{split}_prior_latent_space_tsne_by_latent_color_label",
            f"{split}_prior_latent_space_pca_by_latent_color_label",
            f"{split}_prior_pca_explained_variance",
        }
        assert expected_keys.issubset(set(logged_metrics.keys()))
        assert trainer.logger.log_metrics.call_args.kwargs["step"] == 0

    def test_logs_phase_labeled_figures_when_phase_label_available(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        mock_latent_pl_module_factory: Callable[..., MagicMock],
        label_array_factory: Callable[..., np.ndarray],
        hook: str,
        split: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(
            labels={MetadataKey.PHASE_LABEL.value: label_array_factory()}
        )
        trainer = mock_trainer_factory(current_epoch=0)

        with patch(
            "versatil.training.callbacks.latent_visualization.figure_to_wandb_image"
        ):
            getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        expected_keys = {
            f"{split}_posterior_latent_space_tsne_by_phase_label",
            f"{split}_posterior_latent_space_pca_by_phase_label",
            f"{split}_prior_latent_space_tsne_by_phase_label",
            f"{split}_prior_latent_space_pca_by_phase_label",
        }
        assert expected_keys.issubset(set(logged_metrics.keys()))

    def test_logs_unlabeled_figures_when_labels_are_missing(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        mock_latent_pl_module_factory: Callable[..., MagicMock],
        hook: str,
        split: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(labels={})
        trainer = mock_trainer_factory(current_epoch=0)

        with patch(
            "versatil.training.callbacks.latent_visualization.figure_to_wandb_image"
        ):
            getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        expected_keys = {
            f"{split}_posterior_latent_space_tsne",
            f"{split}_posterior_latent_space_pca",
            f"{split}_posterior_pca_explained_variance",
            f"{split}_prior_latent_space_tsne",
            f"{split}_prior_latent_space_pca",
            f"{split}_prior_pca_explained_variance",
        }
        assert expected_keys.issubset(set(logged_metrics.keys()))

    def test_handles_latent_dimension_one_without_pca_or_tsne(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        mock_latent_pl_module_factory: Callable[..., MagicMock],
        hook: str,
        split: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(latent_dimension=1)
        trainer = mock_trainer_factory(current_epoch=0)

        with patch(
            "versatil.training.callbacks.latent_visualization.figure_to_wandb_image"
        ):
            getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        expected_keys = {
            f"{split}_posterior_latent_space_histogram_by_latent_color_label",
            f"{split}_prior_latent_space_histogram_by_latent_color_label",
        }
        assert expected_keys.issubset(set(logged_metrics.keys()))
        for pca_tsne_key in (
            f"{split}_posterior_latent_space_pca_by_latent_color_label",
            f"{split}_posterior_latent_space_tsne_by_latent_color_label",
            f"{split}_posterior_pca_explained_variance",
            f"{split}_prior_latent_space_pca_by_latent_color_label",
            f"{split}_prior_latent_space_tsne_by_latent_color_label",
            f"{split}_prior_pca_explained_variance",
        ):
            assert pca_tsne_key not in logged_metrics

    def test_logs_only_prior_when_posterior_missing(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        mock_latent_pl_module_factory: Callable[..., MagicMock],
        hook: str,
        split: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(posterior_latent=None)
        trainer = mock_trainer_factory(current_epoch=0)

        with patch(
            "versatil.training.callbacks.latent_visualization.figure_to_wandb_image"
        ):
            getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        prior_keys = {
            f"{split}_prior_latent_space_tsne_by_latent_color_label",
            f"{split}_prior_latent_space_pca_by_latent_color_label",
            f"{split}_prior_pca_explained_variance",
        }
        posterior_keys = {
            f"{split}_posterior_latent_space_tsne_by_latent_color_label",
            f"{split}_posterior_latent_space_pca_by_latent_color_label",
            f"{split}_posterior_pca_explained_variance",
        }
        assert prior_keys.issubset(set(logged_metrics.keys()))
        assert posterior_keys.isdisjoint(set(logged_metrics.keys()))

    def test_does_not_convert_figures_when_logger_is_none(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        mock_latent_pl_module_factory: Callable[..., MagicMock],
        hook: str,
        split: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(prior_latent=None)
        trainer = mock_trainer_factory(current_epoch=0, logger=None)

        with patch(
            "versatil.training.callbacks.latent_visualization.figure_to_wandb_image"
        ) as mock_to_wandb:
            getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        mock_to_wandb.assert_not_called()

    def test_logs_latent_stats_table_when_metadata_present(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        mock_latent_pl_module_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
        hook: str,
        split: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        mu = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        pl_module = mock_latent_pl_module_factory(
            prior_latent=None,
            metadata={MetadataKey.POSTERIOR_MU.value: [mu]},
        )
        trainer = mock_trainer_factory(current_epoch=0)

        with patch(
            "versatil.training.callbacks.latent_visualization.figure_to_wandb_image"
        ):
            getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        assert f"{split}_latent_space_statistics" in logged_metrics

    def test_closes_figures_after_logging(
        self,
        mock_trainer_factory: Callable[..., MagicMock],
        mock_latent_pl_module_factory: Callable[..., MagicMock],
        hook: str,
        split: str,
    ) -> None:
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(prior_latent=None)
        trainer = mock_trainer_factory(current_epoch=0)

        with (
            patch(
                "versatil.training.callbacks.latent_visualization.figure_to_wandb_image"
            ),
            patch(
                "versatil.training.callbacks.latent_visualization.plt.close"
            ) as mock_close,
        ):
            getattr(callback, hook)(trainer=trainer, pl_module=pl_module)

        assert mock_close.call_count == 3


@pytest.mark.unit
class TestCreateLatentFigure:
    def test_returns_figure_with_configured_label_in_title(
        self,
        latent_data_factory: Callable[..., np.ndarray],
        label_array_factory: Callable[..., np.ndarray],
    ) -> None:
        callback = LatentVisualizationCallback(max_samples=100)
        z = latent_data_factory(num_samples=12, latent_dimension=4)
        labels = label_array_factory(num_samples=12, num_labels=3)

        with patch(
            "versatil.training.callbacks.latent_visualization.TSNE"
        ) as mock_tsne_class:
            mock_instance = MagicMock()
            mock_instance.fit_transform.return_value = np.zeros(
                (12, 2), dtype=np.float32
            )
            mock_tsne_class.return_value = mock_instance

            fig = callback._create_latent_figure(
                z=z,
                labels=labels,
                label_name="ground truth mode",
                title="Posterior latent space",
            )

        _, call_kwargs = mock_tsne_class.call_args
        assert call_kwargs["perplexity"] == 11
        assert call_kwargs["n_components"] == 2
        axes = fig.get_axes()
        assert "Posterior latent space" in axes[0].get_title()
        assert "ground truth mode" in axes[0].get_title()
        plt.close(fig)

    def test_returns_figure_without_label_annotation_when_labels_are_none(
        self,
        latent_data_factory: Callable[..., np.ndarray],
    ) -> None:
        callback = LatentVisualizationCallback(max_samples=100)
        z = latent_data_factory(num_samples=10, latent_dimension=3)

        with patch(
            "versatil.training.callbacks.latent_visualization.TSNE"
        ) as mock_tsne_class:
            mock_instance = MagicMock()
            mock_instance.fit_transform.return_value = np.zeros(
                (10, 2), dtype=np.float32
            )
            mock_tsne_class.return_value = mock_instance

            fig = callback._create_latent_figure(
                z=z,
                labels=None,
                label_name="label",
                title="Prior",
            )

        axes = fig.get_axes()
        assert len(axes) == 1
        assert "Prior" in axes[0].get_title()
        assert "colored by" not in axes[0].get_title()
        plt.close(fig)

    def test_subsamples_latents_and_labels_together(
        self,
        latent_data_factory: Callable[..., np.ndarray],
        label_array_factory: Callable[..., np.ndarray],
    ) -> None:
        max_samples = 20
        callback = LatentVisualizationCallback(max_samples=max_samples)
        z = latent_data_factory(num_samples=50, latent_dimension=4)
        labels = label_array_factory(num_samples=50, num_labels=3)

        with patch(
            "versatil.training.callbacks.latent_visualization.TSNE"
        ) as mock_tsne_class:
            mock_instance = MagicMock()
            mock_instance.fit_transform.return_value = np.zeros(
                (max_samples, 2), dtype=np.float32
            )
            mock_tsne_class.return_value = mock_instance

            fig = callback._create_latent_figure(
                z=z,
                labels=labels,
                label_name="mode",
                title="X",
            )

        fitted = mock_instance.fit_transform.call_args.args[0]
        assert fitted.shape == (max_samples, 4)
        plt.close(fig)


@pytest.mark.unit
class TestCreateHistogramFigure:
    def test_returns_figure_with_label_overlays(
        self,
        latent_data_factory: Callable[..., np.ndarray],
        label_array_factory: Callable[..., np.ndarray],
    ) -> None:
        callback = LatentVisualizationCallback(max_samples=500)
        z = latent_data_factory(num_samples=200, latent_dimension=1)
        labels = label_array_factory(num_samples=200, num_labels=3)

        fig = callback._create_histogram_figure(
            z=z,
            labels=labels,
            label_name="ground truth mode",
            title="Posterior latent space",
        )

        axes = fig.get_axes()
        assert len(axes) == 1
        assert "Posterior latent space histogram" in axes[0].get_title()
        assert "ground truth mode" in axes[0].get_title()
        assert axes[0].get_xlabel() == "Latent value"
        assert axes[0].get_ylabel() == "Density"
        plt.close(fig)

    def test_returns_figure_without_hue_when_labels_are_none(
        self,
        latent_data_factory: Callable[..., np.ndarray],
    ) -> None:
        callback = LatentVisualizationCallback(max_samples=500)
        z = latent_data_factory(num_samples=150, latent_dimension=1)

        fig = callback._create_histogram_figure(
            z=z,
            labels=None,
            label_name="label",
            title="Prior",
        )

        axes = fig.get_axes()
        assert "Prior histogram" in axes[0].get_title()
        assert "colored by" not in axes[0].get_title()
        plt.close(fig)

    def test_accepts_1d_input_shape(
        self,
        rng: np.random.Generator,
    ) -> None:
        callback = LatentVisualizationCallback(max_samples=500)
        z = rng.standard_normal(120).astype(np.float32)

        fig = callback._create_histogram_figure(
            z=z,
            labels=None,
            label_name="label",
            title="Posterior",
        )

        axes = fig.get_axes()
        assert "Posterior histogram" in axes[0].get_title()
        plt.close(fig)

    def test_subsamples_when_exceeding_max_samples(
        self,
        latent_data_factory: Callable[..., np.ndarray],
        label_array_factory: Callable[..., np.ndarray],
    ) -> None:
        max_samples = 25
        callback = LatentVisualizationCallback(max_samples=max_samples)
        z = latent_data_factory(num_samples=200, latent_dimension=1)
        labels = label_array_factory(num_samples=200, num_labels=2)

        with patch(
            "versatil.training.callbacks.latent_visualization.sns.histplot"
        ) as mock_histplot:
            callback._create_histogram_figure(
                z=z,
                labels=labels,
                label_name="mode",
                title="X",
            )

        forwarded_values = mock_histplot.call_args.kwargs["x"]
        forwarded_labels = mock_histplot.call_args.kwargs["hue"]
        assert forwarded_values.shape == (max_samples,)
        assert forwarded_labels.shape == (max_samples,)


@pytest.mark.unit
class TestBuildLatentFigures:
    def test_dispatches_to_histogram_for_1d_latent(
        self,
        latent_data_factory: Callable[..., np.ndarray],
        label_array_factory: Callable[..., np.ndarray],
    ) -> None:
        callback = LatentVisualizationCallback()
        z = latent_data_factory(num_samples=50, latent_dimension=1)
        labels = label_array_factory(num_samples=50, num_labels=2)

        figures = callback._build_latent_figures(
            z=z,
            labels_by_key={"mode_id": labels},
            prefix="posterior",
            title="Posterior",
        )

        assert set(figures.keys()) == {"posterior_latent_space_histogram_by_mode_id"}
        for fig in figures.values():
            plt.close(fig)

    @pytest.mark.parametrize("latent_dimension", [2, 8])
    def test_dispatches_to_pca_and_tsne_for_higher_dim(
        self,
        latent_data_factory: Callable[..., np.ndarray],
        label_array_factory: Callable[..., np.ndarray],
        latent_dimension: int,
    ) -> None:
        callback = LatentVisualizationCallback()
        z = latent_data_factory(num_samples=40, latent_dimension=latent_dimension)
        labels = label_array_factory(num_samples=40, num_labels=2)

        figures = callback._build_latent_figures(
            z=z,
            labels_by_key={"mode_id": labels},
            prefix="posterior",
            title="Posterior",
        )

        assert set(figures.keys()) == {
            "posterior_latent_space_tsne_by_mode_id",
            "posterior_latent_space_pca_by_mode_id",
            "posterior_pca_explained_variance",
        }
        for fig in figures.values():
            plt.close(fig)


@pytest.mark.unit
class TestCreatePcaFigure:
    def test_returns_figure_with_label_colored_scatter(
        self,
        latent_data_factory: Callable[..., np.ndarray],
        label_array_factory: Callable[..., np.ndarray],
    ) -> None:
        callback = LatentVisualizationCallback(max_samples=100)
        z = latent_data_factory(num_samples=12, latent_dimension=4)
        labels = label_array_factory(num_samples=12, num_labels=3)

        fig = callback._create_pca_figure(
            z=z,
            labels=labels,
            label_name="ground truth mode",
            title="Posterior latent space",
        )

        axes = fig.get_axes()
        assert len(axes) >= 1
        main_title = axes[0].get_title()
        assert "Posterior latent space" in main_title
        assert "ground truth mode" in main_title
        assert "PC1" in axes[0].get_xlabel()
        assert "%" in axes[0].get_xlabel()
        assert "PC2" in axes[0].get_ylabel()
        plt.close(fig)

    def test_returns_figure_without_hue_when_labels_are_none(
        self,
        latent_data_factory: Callable[..., np.ndarray],
    ) -> None:
        callback = LatentVisualizationCallback(max_samples=100)
        z = latent_data_factory(num_samples=10, latent_dimension=3)

        fig = callback._create_pca_figure(
            z=z,
            labels=None,
            label_name="label",
            title="Prior",
        )

        axes = fig.get_axes()
        assert "Prior PCA" in axes[0].get_title()
        assert "colored by" not in axes[0].get_title()
        plt.close(fig)

    def test_subsamples_when_exceeding_max_samples(
        self,
        latent_data_factory: Callable[..., np.ndarray],
        label_array_factory: Callable[..., np.ndarray],
    ) -> None:
        max_samples = 15
        callback = LatentVisualizationCallback(max_samples=max_samples)
        z = latent_data_factory(num_samples=40, latent_dimension=5)
        labels = label_array_factory(num_samples=40, num_labels=2)

        with patch(
            "versatil.training.callbacks.latent_visualization.PCA"
        ) as mock_pca_class:
            mock_instance = MagicMock()
            mock_instance.fit_transform.return_value = np.zeros(
                (max_samples, 2), dtype=np.float32
            )
            mock_instance.explained_variance_ratio_ = np.array([0.5, 0.3])
            mock_pca_class.return_value = mock_instance

            callback._create_pca_figure(
                z=z,
                labels=labels,
                label_name="mode",
                title="X",
            )

        fitted = mock_instance.fit_transform.call_args.args[0]
        assert fitted.shape == (max_samples, 5)


@pytest.mark.unit
def test_pca_variance_figure_returns_bar_chart_with_one_bar_per_component(
    latent_data_factory: Callable[..., np.ndarray],
) -> None:
    callback = LatentVisualizationCallback()
    latent_dimension = 5
    num_samples = 20
    z = latent_data_factory(num_samples=num_samples, latent_dimension=latent_dimension)

    fig = callback._create_pca_variance_figure(z, title="Posterior")

    axes = fig.get_axes()
    assert len(axes) == 1
    assert "Posterior" in axes[0].get_title()
    assert "Explained Variance" in axes[0].get_title()
    assert axes[0].get_xlabel() == "Principal Component"
    assert len(axes[0].patches) == latent_dimension
    plt.close(fig)


@pytest.mark.unit
class TestCreateLatentStatsTable:
    def test_returns_none_when_metadata_empty(self) -> None:
        callback = LatentVisualizationCallback()

        table = callback._create_latent_stats_table(metadata={})

        assert table is None

    def test_returns_wandb_table_with_expected_columns_and_rows(
        self,
        rng: np.random.Generator,
    ) -> None:
        callback = LatentVisualizationCallback()
        posterior_mu = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        prior_mu = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        metadata = {
            MetadataKey.POSTERIOR_MU.value: [posterior_mu],
            MetadataKey.PRIOR_MU.value: [prior_mu],
        }

        table = callback._create_latent_stats_table(metadata=metadata)

        assert table is not None
        expected_columns = [
            "name",
            "shape",
            "mean",
            "per_dim_std_of_mean",
            "std",
            "per_dim_mean_of_std",
            "min",
            "max",
            "collapsed_dims",
        ]
        assert list(table.columns) == expected_columns
        assert len(table.data) == 2
        row_labels = {row[0] for row in table.data}
        assert row_labels == {"mu_posterior", "mu_prior"}

    def test_flattens_three_dimensional_metadata(
        self,
        rng: np.random.Generator,
    ) -> None:
        callback = LatentVisualizationCallback()
        posterior_z = torch.from_numpy(
            rng.standard_normal((6, 2, 3)).astype(np.float32)
        )
        metadata = {MetadataKey.POSTERIOR_Z.value: [posterior_z]}

        table = callback._create_latent_stats_table(metadata=metadata)

        assert table is not None
        assert len(table.data) == 1
        shape_field = table.data[0][1]
        assert shape_field == str((6, 6))

    def test_counts_collapsed_dimensions_below_threshold(
        self,
        rng: np.random.Generator,
    ) -> None:
        callback = LatentVisualizationCallback()
        base = rng.standard_normal((20, 4)).astype(np.float32)
        base[:, 0] = 0.001
        base[:, 1] = 0.005
        tensor = torch.from_numpy(base)
        metadata = {MetadataKey.POSTERIOR_MU.value: [tensor]}

        table = callback._create_latent_stats_table(metadata=metadata)

        collapsed_dims = table.data[0][-1]
        assert collapsed_dims == 2
