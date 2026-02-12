"""Visualize latent space of a variational model on LIBERO validation data.

Quick and dirty script for:
- T-SNE and PCA projections of posterior/prior latent spaces
- K-Means clustering
- Sample image inspection per cluster
- Colored by language instruction (task label)
"""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


from hydra.utils import instantiate
from omegaconf import OmegaConf

from versatil.common.tensor_ops import to_device
from versatil.data.constants import SampleKey
from versatil.data.episodic_dataset import EpisodicDataset
from versatil.models.decoding.constants import LatentKey
from versatil.training.lightning_policy import LightningPolicy

CHECKPOINT_PATH = "/mnt/cluster/workspaces/mazzalore/libero/end_to_end_training_runs/libero_lerobot/lact/20260204_095401/best-99-2.1587.ckpt"
CONFIG_PATH = "/mnt/cluster/workspaces/mazzalore/libero/end_to_end_training_runs/libero_lerobot/lact/20260204_095401/config.yaml"
ZARR_PATH = "/mnt/cluster/datasets/robotics_zarr/libero_lerobot.zarr"
OUTPUT_DIR = Path("/mnt/cluster/workspaces/mazzalore/latent_space_visualization/lact/20260204_095401")
DEVICE = "cuda"
MAX_SAMPLES = 50000
N_CLUSTERS = 5
BATCH_SIZE = 64
NUM_WORKERS = 4

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



def setup_data_and_model():
    """Load config, create datasets, instantiate model, load checkpoint."""
    logging.info(f"Loading config from {CONFIG_PATH}")
    cfg = OmegaConf.load(CONFIG_PATH)
    action_space = instantiate(cfg.task.action_space)
    observation_space = instantiate(cfg.task.observation_space)
    dataloader_config = instantiate(cfg.task.dataloader)
    pred_horizon = cfg.task.prediction_horizon
    obs_horizon = cfg.task.observation_horizon
    seed = cfg.experiment.seed
    logging.info("Creating train dataset (for normalizer)...")
    train_dataset = EpisodicDataset(
        zarr_path=ZARR_PATH,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        dataloader_config=dataloader_config,
        train=True,
        seed=seed,
        action_space=action_space,
        observation_space=observation_space,
    )
    logging.info("Creating validation dataset...")
    val_dataset = EpisodicDataset(
        zarr_path=ZARR_PATH,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        dataloader_config=dataloader_config,
        train=False,
        seed=seed,
        action_space=action_space,
        observation_space=observation_space,
    )
    logging.info("Building normalizer and tokenizer from training data...")
    tokenization_config = dataloader_config.tokenization
    normalizer, tokenizer = train_dataset.get_normalizer_and_tokenizer(
        winsorize_depth=dataloader_config.winsorize_depth,
        depth_winsorize_quantiles=dataloader_config.depth_winsorize_quantiles,
        winsorize_kinematics=dataloader_config.winsorize_kinematics,
        kinematics_winsorize_quantiles=dataloader_config.kinematics_winsorize_quantiles,
        tokenization_config=tokenization_config,
        clamp_kinematics_range=dataloader_config.clamp_kinematics_range,
        min_kinematics_std=dataloader_config.min_kinematics_std,
        min_kinematics_range=dataloader_config.min_kinematics_range,
        device=torch.device("cpu"),
    )
    train_dataset.set_normalizer(normalizer)
    val_dataset.set_normalizer(normalizer)
    train_dataset.set_tokenizer(tokenizer)
    val_dataset.set_tokenizer(tokenizer)
    logging.info("Instantiating policy...")
    policy = instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    policy.set_tokenizer(tokenizer)
    training_config = instantiate(cfg.training)
    lightning_policy = LightningPolicy(
        policy=policy,
        training_config=training_config,
        total_training_steps=1,
    )
    device = torch.device(DEVICE)
    lightning_policy.to(device)
    lightning_policy.eval()
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    logging.info("Initializing lazy modules...")
    init_batch = next(iter(val_loader))
    init_batch = to_device(init_batch, device)
    with torch.no_grad():
        _ = policy.compute_loss(init_batch)
    logging.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    lightning_policy.load_state_dict(checkpoint["state_dict"])
    lightning_policy.eval()
    tokenizer_path = Path(CHECKPOINT_PATH).parent / "tokenizer"
    if tokenizer_path.exists():
        from versatil.data.tokenization import Tokenizer

        tok = Tokenizer.from_pretrained(tokenizer_path, device=device)
        policy.set_tokenizer(tok)
        logging.info(f"Loaded tokenizer from {tokenizer_path}")

    logging.info("Model loaded successfully.")
    return policy, val_loader, val_dataset



def get_language_labels(val_dataset, zarr_path):
    """Get language instruction for each sample in the val dataset from zarr."""
    zarr_root = zarr.open(store=zarr_path, mode="r")
    lang_data = zarr_root["data"]["language_instruction"]

    n_samples = len(val_dataset)
    languages = []
    for i in range(n_samples):
        buf_start = int(val_dataset.sampler.indices[i][0])
        lang = lang_data[buf_start]
        if isinstance(lang, np.ndarray):
            lang = lang.flat[0] if lang.size > 0 else ""
        languages.append(str(lang))

    unique = sorted(set(languages))
    logging.info(f"Found {len(unique)} unique language instructions across {n_samples} val samples")
    return languages



def collect_latents(policy, val_loader):
    """Run forward passes on validation data and collect latent vectors + images."""
    all_z_posterior = []
    all_mu_posterior = []
    all_z_prior = []
    all_mu_prior = []
    all_mean_displacement = []
    all_images = []
    all_languages = []

    policy.eval()
    device = torch.device(DEVICE)
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting latents"):
            if total >= MAX_SAMPLES:
                break

            batch = to_device(batch, device)
            output = policy.forward(batch)

            # Posterior
            z_post = output[LatentKey.POSTERIOR_LATENT.value].cpu().numpy()
            mu_post = output[LatentKey.POSTERIOR_MU.value].cpu().numpy()
            all_z_posterior.append(z_post)
            all_mu_posterior.append(mu_post)

            # Prior
            if LatentKey.PRIOR_LATENT.value in output:
                all_z_prior.append(
                    output[LatentKey.PRIOR_LATENT.value].cpu().numpy()
                )
            if LatentKey.PRIOR_MU.value in output:
                all_mu_prior.append(
                    output[LatentKey.PRIOR_MU.value].cpu().numpy()
                )

            actions = batch[SampleKey.ACTION.value]
            position_keys = list(policy.action_space.position_actions.keys())
            if total == 0 and position_keys:
                logging.info(f"Using position key: {position_keys[0]}")
                logging.info(f"Action dict keys: {list(actions.keys())}")
            if position_keys:
                position = actions[position_keys[0]]  # (B, k, dim)
                mean_disp = position.mean(dim=1).cpu().numpy()  # (B, dim)
                all_mean_displacement.append(mean_disp)

            obs = batch[SampleKey.OBSERVATION.value]
            if "agentview_rgb" in obs:
                imgs = obs["agentview_rgb"].cpu().numpy()
                if imgs.ndim == 5:
                    imgs = imgs[:, -1]  # last obs timestep
                all_images.append(imgs)
            if "language_instruction" in obs:
                batch_langs = obs["language_instruction"][0][0]
                all_languages.extend(batch_langs)

            total += z_post.shape[0]

    result = {
        "z_posterior": np.concatenate(all_z_posterior)[:MAX_SAMPLES],
        "mu_posterior": np.concatenate(all_mu_posterior)[:MAX_SAMPLES],
    }
    if all_z_prior:
        result["z_prior"] = np.concatenate(all_z_prior)[:MAX_SAMPLES]
    if all_mu_prior:
        result["mu_prior"] = np.concatenate(all_mu_prior)[:MAX_SAMPLES]
    if all_mean_displacement:
        result["mean_displacement"] = np.concatenate(all_mean_displacement)[:MAX_SAMPLES]
    if all_images:
        result["images"] = np.concatenate(all_images)[:MAX_SAMPLES]
    if all_languages:
        result["languages"] = all_languages[:MAX_SAMPLES]

    logging.info(
        f"Collected {result['z_posterior'].shape[0]} samples, "
        f"latent_dim={result['z_posterior'].shape[1]}"
    )
    return result



LIBERO_TASK_TO_SUITE = {
    "turn on the stove and put the moka pot on it": "libero_long",
    "put the black bowl in the bottom drawer of the cabinet and close it": "libero_long",
    "put the yellow and white mug in the microwave and close it": "libero_long",
    "put both moka pots on the stove": "libero_long",
    "put both the alphabet soup and the cream cheese box in the basket": "libero_long",
    "put both the alphabet soup and the tomato sauce in the basket": "libero_long",
    "put both the cream cheese box and the butter in the basket": "libero_long",
    "put the white mug on the left plate and put the yellow and white mug on the right plate": "libero_long",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": "libero_long",
    "pick up the book and place it in the back compartment of the caddy": "libero_long",
    "open the middle drawer of the cabinet": "libero_goal",
    "open the top drawer and put the bowl inside": "libero_goal",
    "push the plate to the front of the stove": "libero_goal",
    "put the bowl on the plate": "libero_goal",
    "put the bowl on the stove": "libero_goal",
    "put the bowl on top of the cabinet": "libero_goal",
    "put the cream cheese in the bowl": "libero_goal",
    "put the wine bottle on the rack": "libero_goal",
    "put the wine bottle on top of the cabinet": "libero_goal",
    "turn on the stove": "libero_goal",
    "pick up the alphabet soup and place it in the basket": "libero_object",
    "pick up the bbq sauce and place it in the basket": "libero_object",
    "pick up the butter and place it in the basket": "libero_object",
    "pick up the chocolate pudding and place it in the basket": "libero_object",
    "pick up the cream cheese and place it in the basket": "libero_object",
    "pick up the ketchup and place it in the basket": "libero_object",
    "pick up the milk and place it in the basket": "libero_object",
    "pick up the orange juice and place it in the basket": "libero_object",
    "pick up the salad dressing and place it in the basket": "libero_object",
    "pick up the tomato sauce and place it in the basket": "libero_object",
    "pick up the black bowl between the plate and the ramekin and place it on the plate": "libero_spatial",
    "pick up the black bowl from table center and place it on the plate": "libero_spatial",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": "libero_spatial",
    "pick up the black bowl next to the cookie box and place it on the plate": "libero_spatial",
    "pick up the black bowl next to the plate and place it on the plate": "libero_spatial",
    "pick up the black bowl next to the ramekin and place it on the plate": "libero_spatial",
    "pick up the black bowl on the cookie box and place it on the plate": "libero_spatial",
    "pick up the black bowl on the ramekin and place it on the plate": "libero_spatial",
    "pick up the black bowl on the stove and place it on the plate": "libero_spatial",
    "pick up the black bowl on the wooden cabinet and place it on the plate": "libero_spatial",
}


def get_suite_labels(languages: list[str]) -> list[str]:
    """Map language instructions to LIBERO suite names."""
    suites = []
    for lang in languages:
        suite = LIBERO_TASK_TO_SUITE.get(lang, "unknown")
        if suite == "unknown":
            logging.warning(f"Unknown task: {lang}")
        suites.append(suite)
    return suites


def denormalize_image(img):
    """Convert normalized image to uint8 for display."""
    if img.dtype == np.uint8:
        return img
    # Handle CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    # Handle [-1, 1] range (minus_one_to_one normalization)
    if img.min() < -0.5:
        img = (img + 1.0) / 2.0
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def assign_colors(languages):
    """Assign colors to unique language instructions."""
    unique_langs = sorted(set(languages))
    n = len(unique_langs)
    cmap = plt.cm.get_cmap("tab20", max(n, 20))
    if n > 20:
        # Use two colormaps for more than 20 classes
        cmap2 = plt.cm.get_cmap("Set3", 12)
        colors_list = [cmap(i % 20) for i in range(20)] + [
            cmap2(i % 12) for i in range(n - 20)
        ]
        lang_to_color = {lang: colors_list[i] for i, lang in enumerate(unique_langs)}
    else:
        lang_to_color = {lang: cmap(i) for i, lang in enumerate(unique_langs)}
    return lang_to_color, unique_langs


def make_legend(ax, lang_to_color, unique_langs, max_label_len=55):
    """Add a compact legend to the plot."""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=lang_to_color[lang],
            markersize=7,
            label=lang[:max_label_len],
        )
        for lang in unique_langs
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=5,
        framealpha=0.8,
    )



def plot_tsne(z, languages, title, output_path):
    """T-SNE colored by language instruction."""
    n = len(z)
    subsample_idx = None
    if n > 8000:
        subsample_idx = np.random.choice(n, 8000, replace=False)
        z = z[subsample_idx]
        languages = [languages[i] for i in subsample_idx]
        n = 8000

    perplexity = min(50, max(5, n // 10))
    logging.info(f"Running T-SNE (n={n}, perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(z)

    lang_to_color, unique_langs = assign_colors(languages)
    colors = [lang_to_color[l] for l in languages]

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, alpha=0.4, s=6, edgecolors="none")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")
    make_legend(ax, lang_to_color, unique_langs)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def plot_pca(z, languages, title, output_path):
    """PCA colored by language instruction."""
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)

    lang_to_color, unique_langs = assign_colors(languages)
    colors = [lang_to_color[l] for l in languages]

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, alpha=0.4, s=6, edgecolors="none")
    ev = pca.explained_variance_ratio_
    ax.set_title(f"{title}\nExplained variance: {ev[:2].sum():.1%}", fontsize=14)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    make_legend(ax, lang_to_color, unique_langs)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def plot_pca_continuous(z, values, title, output_path, colorbar_label="Action X", cmap="viridis"):
    """PCA colored by a continuous variable."""
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)
    explained_variance = pca.explained_variance_ratio_

    fig, axis = plt.subplots(figsize=(16, 12))
    scatter = axis.scatter(
        z_2d[:, 0], z_2d[:, 1], c=values, cmap=cmap, alpha=0.4, s=6, edgecolors="none"
    )
    plt.colorbar(scatter, ax=axis, label=colorbar_label)
    axis.set_title(
        f"{title}\nExplained variance: {explained_variance[:2].sum():.1%}", fontsize=14
    )
    axis.set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
    axis.set_ylabel(f"PC2 ({explained_variance[1]:.1%})")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def plot_pca_variance(z, title, output_path):
    """PCA explained variance spectrum."""
    pca = PCA()
    pca.fit(z)

    n_components = min(len(pca.explained_variance_ratio_), 32)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(n_components), pca.explained_variance_ratio_[:n_components])
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title(f"{title} - Individual")

    cumvar = np.cumsum(pca.explained_variance_ratio_[:n_components])
    ax2.plot(range(n_components), cumvar, "o-", markersize=4)
    ax2.axhline(y=0.90, color="orange", linestyle="--", alpha=0.7, label="90%")
    ax2.axhline(y=0.95, color="r", linestyle="--", alpha=0.7, label="95%")
    ax2.axhline(y=0.99, color="g", linestyle="--", alpha=0.7, label="99%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title(f"{title} - Cumulative")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def plot_kmeans(z, languages, n_clusters, title, output_path):
    """K-Means clustering with side-by-side cluster vs language coloring."""
    logging.info(f"Running K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(z)

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)

    fig, axes = plt.subplots(1, 2, figsize=(28, 10))

    cmap_clusters = plt.cm.get_cmap("tab10", n_clusters)
    cluster_colors = [cmap_clusters(c) for c in cluster_labels]
    axes[0].scatter(
        z_2d[:, 0], z_2d[:, 1], c=cluster_colors, alpha=0.4, s=6, edgecolors="none"
    )
    axes[0].set_title(f"{title} - K-Means (k={n_clusters})", fontsize=13)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    lang_to_color, unique_langs = assign_colors(languages)
    colors = [lang_to_color[l] for l in languages]
    axes[1].scatter(
        z_2d[:, 0], z_2d[:, 1], c=colors, alpha=0.4, s=6, edgecolors="none"
    )
    axes[1].set_title(f"{title} - By Task", fontsize=13)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    make_legend(axes[1], lang_to_color, unique_langs)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")

    logging.info(msg=f"\n{'=' * 80}")
    logging.info(msg=f"K-Means Cluster Composition ({title}, k={n_clusters})")
    logging.info(msg=f"{'=' * 80}")
    for c in range(n_clusters):
        mask = cluster_labels == c
        count = mask.sum()
        if count == 0:
            continue
        cluster_langs = [languages[i] for i in range(len(languages)) if mask[i]]
        lang_counts = defaultdict(int)
        for lang in cluster_langs:
            lang_counts[lang] += 1

        logging.info(msg=f"\nCluster {c} ({count} samples):")
        for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1])[:5]:
            pct = 100 * cnt / count
            logging.info(msg=f"  {cnt:4d} ({pct:5.1f}%) {lang[:65]}")

    return cluster_labels, kmeans


def plot_cluster_image_grid(images, cluster_labels, languages, title, output_path, n_images_per_cluster=8):
    """Grid of sample images per K-Means cluster, one row per cluster."""
    if images is None or len(images) == 0:
        logging.warning("No images available for cluster image grid")
        return

    n_clusters = len(set(cluster_labels))
    fig, axes = plt.subplots(
        n_clusters, n_images_per_cluster,
        figsize=(3 * n_images_per_cluster, 3 * n_clusters),
    )
    fig.subplots_adjust(wspace=0.05, hspace=0.4)
    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_indices = np.where(mask)[0]
        count = len(cluster_indices)

        # Pick random samples from this cluster
        n_show = min(n_images_per_cluster, count)
        chosen = np.random.choice(cluster_indices, size=n_show, replace=False)

        # Build row label with top language instructions
        cluster_langs = [languages[i] for i in cluster_indices]
        lang_counts = defaultdict(int)
        for lang in cluster_langs:
            lang_counts[lang] += 1
        top_langs = sorted(lang_counts.items(), key=lambda x: -x[1])[:2]
        label_parts = [f"Cluster {c} ({count})"]
        for lang, cnt in top_langs:
            label_parts.append(f"{lang[:50]} ({cnt})")
        row_label = "\n".join(label_parts)

        for j in range(n_images_per_cluster):
            ax = axes[c, j]
            ax.axis("off")
            if j < n_show:
                img = denormalize_image(images[chosen[j]].copy())
                ax.imshow(img)
            if j == 0:
                ax.set_title(row_label, fontsize=9, loc="left", pad=4)

    fig.suptitle(title, fontsize=16, y=1.01)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def print_latent_statistics(data):
    """Print summary statistics of latent representations."""
    logging.info(msg=f"\n{'=' * 60}")
    logging.info(msg="Latent Space Statistics")
    logging.info(msg=f"{'=' * 60}")

    for key in ["mu_posterior", "z_posterior", "mu_prior", "z_prior"]:
        if key not in data:
            continue
        z = data[key]
        logging.info(msg=f"\n{key}:")
        logging.info(msg=f"  Shape:    {z.shape}")
        logging.info(msg=f"  Mean:     {z.mean():.4f} (per-dim std: {z.mean(axis=0).std():.4f})")
        logging.info(msg=f"  Std:      {z.std():.4f} (per-dim mean: {z.std(axis=0).mean():.4f})")
        logging.info(msg=f"  Min/Max:  [{z.min():.3f}, {z.max():.3f}]")
        dim_std = z.std(axis=0)
        n_collapsed = (dim_std < 0.01).sum()
        if n_collapsed > 0:
            logging.warning(msg=f"{n_collapsed}/{z.shape[1]} dims have std < 0.01 (possibly collapsed)")




def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    policy, val_loader, val_dataset = setup_data_and_model()

    data = collect_latents(policy, val_loader)
    n = len(data["z_posterior"])
    languages = data["languages"]

    logging.info(msg=f"\nCollected {n} samples with latent_dim={data['z_posterior'].shape[1]}")
    logging.info(msg=f"Unique tasks: {len(set(languages))}")

    print_latent_statistics(data)

    plot_tsne(
        data["mu_posterior"],
        languages,
        "Posterior Mean (T-SNE) colored by task",
        OUTPUT_DIR / "tsne_posterior_mu.png",
    )
    if "mu_prior" in data:
        plot_tsne(
            data["mu_prior"],
            languages,
            "Prior Mean (T-SNE) colored by task",
            OUTPUT_DIR / "tsne_prior_mu.png",
        )
    plot_tsne(
        data["z_posterior"],
        languages,
        "Posterior z samples (T-SNE) colored by task",
        OUTPUT_DIR / "tsne_posterior_z.png",
    )
    if "z_prior" in data:
        plot_tsne(
            data["z_prior"],
            languages,
            "Prior z samples (T-SNE) colored by task",
            OUTPUT_DIR / "tsne_prior_z.png",
        )

    plot_pca(
        data["mu_posterior"],
        languages,
        "Posterior Mean (PCA)",
        OUTPUT_DIR / "pca_posterior_mu.png",
    )
    if "mu_prior" in data:
        plot_pca(
            data["mu_prior"],
            languages,
            "Prior Mean (PCA)",
            OUTPUT_DIR / "pca_prior_mu.png",
        )
    plot_pca(
        data["z_posterior"],
        languages,
        "Posterior z samples (PCA)",
        OUTPUT_DIR / "pca_posterior_z.png",
    )
    if "z_prior" in data:
        plot_pca(
            data["z_prior"],
            languages,
            "Prior z samples (PCA)",
            OUTPUT_DIR / "pca_prior_z.png",
        )

    suites = get_suite_labels(languages)
    for latent_name, latent_key in [
        ("posterior_mu", "mu_posterior"),
        ("posterior_z", "z_posterior"),
        ("prior_mu", "mu_prior"),
        ("prior_z", "z_prior"),
    ]:
        if latent_key not in data:
            continue
        plot_pca(
            data[latent_key],
            suites,
            f"{latent_name} (PCA) colored by suite",
            OUTPUT_DIR / f"pca_suite_{latent_name}.png",
        )

    mean_disp = data.get("mean_displacement")
    if mean_disp is not None:
        disp_magnitude = np.linalg.norm(mean_disp, axis=-1)
        disp_direction = np.arctan2(mean_disp[:, 1], mean_disp[:, 0])
        for latent_name, latent_key in [
            ("posterior_mu", "mu_posterior"),
            ("posterior_z", "z_posterior"),
            ("prior_mu", "mu_prior"),
            ("prior_z", "z_prior"),
        ]:
            if latent_key not in data:
                continue
            plot_pca_continuous(
                data[latent_key],
                disp_magnitude,
                f"{latent_name} (PCA) colored by action magnitude",
                OUTPUT_DIR / f"pca_action_magnitude_{latent_name}.png",
                colorbar_label="Mean displacement magnitude",
            )
            plot_pca_continuous(
                data[latent_key],
                disp_direction,
                f"{latent_name} (PCA) colored by action direction",
                OUTPUT_DIR / f"pca_action_direction_{latent_name}.png",
                colorbar_label="Mean displacement angle (rad)",
                cmap="hsv",
            )

    plot_pca_variance(
        data["mu_posterior"],
        "Posterior Mean",
        OUTPUT_DIR / "pca_variance_posterior.png",
    )
    if "mu_prior" in data:
        plot_pca_variance(
            data["mu_prior"],
            "Prior Mean",
            OUTPUT_DIR / "pca_variance_prior.png",
        )

    cluster_labels, kmeans = plot_kmeans(
        data["mu_posterior"],
        languages,
        N_CLUSTERS,
        "Posterior Mean",
        OUTPUT_DIR / "kmeans_posterior_mu.png",
    )

    if "mu_prior" in data:
        cluster_labels_prior, _ = plot_kmeans(
            data["mu_prior"],
            languages,
            N_CLUSTERS,
            "Prior Mean",
            OUTPUT_DIR / "kmeans_prior_mu.png",
        )

    # Cluster image grids: sample images per K-Means cluster
    images = data.get("images")
    if images is not None:
        plot_cluster_image_grid(
            images, cluster_labels, languages,
            "Posterior Mean - Cluster Samples",
            OUTPUT_DIR / "cluster_images_posterior_mu.png",
        )
        if "mu_prior" in data:
            plot_cluster_image_grid(
                images, cluster_labels_prior, languages,
                "Prior Mean - Cluster Samples",
                OUTPUT_DIR / "cluster_images_prior_mu.png",
            )

    logging.info(msg=f"\nAll plots saved to {OUTPUT_DIR}/")
    logging.info(msg="Done.")


if __name__ == "__main__":
    main()