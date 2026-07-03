# Supervised Mixture-of-Experts for Surgical Grasping and Retraction

Paper: [arXiv:2601.21971](https://arxiv.org/abs/2601.21971)

Dataset: [nct-tso/robotics_bowel_grasping](https://huggingface.co/datasets/nct-tso/robotics_bowel_grasping)

This note describes how to reproduce the VersatIL training runs for the fixed-
and random-viewpoint bowel grasping and retraction experiments. The main config
families are the ACT baselines and the supervised phase-aware ACT policies.

## Dataset

Download the dataset from Hugging Face:

```bash
mkdir -p /path/to/robotics_bowel_grasping
cd /path/to/robotics_bowel_grasping

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nct-tso/robotics_bowel_grasping",
    repo_type="dataset",
    local_dir=".",
)
PY
```

For the fixed-viewpoint ACT and PhaseACT training runs, extract the
fixed-viewpoint archive:

```bash
tar --use-compress-program="zstd -d" \
    -xf bowel_retraction_fixed_viewpoint.tar.zst
```

This creates:

```text
/path/to/robotics_bowel_grasping/
  v1/
  v2/
```

The released dataset also includes the random-viewpoint archive. Extract it for
the mixed fixed-plus-random viewpoint training run:

```bash
tar --use-compress-program="zstd -d" \
    -xf bowel_retraction_random_viewpoint.tar.zst
```

This adds:

```text
/path/to/robotics_bowel_grasping/
  multicam/
    multi_camera_exp/
    multi_camera_exp2/
```

The `multicam/` folders are used for the viewpoint-generalization experiments.

## Replace Image Paths

The `episode.csv` files contain the original absolute acquisition paths in the
`frameLeftRectifiedPath` and `frameRightRectifiedPath` columns. Replace those
prefixes with your local extraction path before creating the VersatIL Zarr
cache:

```bash
python - <<'PY'
from pathlib import Path

root = Path("/path/to/robotics_bowel_grasping")
replacements = {
    "/mnt/cluster/datasets/bowel_retraction/v1/": f"{root}/v1/",
    "/mnt/cluster/datasets/bowel_retraction/v2/": f"{root}/v2/",
    "/mnt/cluster/datasets/bowel_retraction/multi_camera_exp/": (
        f"{root}/multicam/multi_camera_exp/"
    ),
    "/mnt/cluster/datasets/bowel_retraction/multi_camera_exp2/": (
        f"{root}/multicam/multi_camera_exp2/"
    ),
}

for csv_path in root.rglob("episode.csv"):
    text = csv_path.read_text()
    for old, new in replacements.items():
        text = text.replace(old, new)
    csv_path.write_text(text)
PY
```

This step is required because the TSO dataset schema reads the image path
strings stored in `episode.csv`.

## Environment Variables

Create and edit `.env` in the VersatIL repository root:

```bash
cp .env.example .env
```

Set at least:

```bash
VERSATIL_CACHE_DIR=/path/to/cache
VERSATIL_CHECKPOINT_DIR=/path/to/checkpoints
VERSATIL_ZARR_DIR=/path/to/zarr
VERSATIL_BOWEL_RETRACTION_DIR=/path/to/robotics_bowel_grasping
```

Optional Weights & Biases variables:

```bash
WANDB_PROJECT=versatil
WANDB_ENTITY=your-team
```

The bowel-retraction schema used by the configs below is
`src/versatil/hydra_configs/task/dataset_schema/bowel_retraction_v2.yaml`. It expects
`VERSATIL_BOWEL_RETRACTION_DIR` to contain `v1/` and `v2/`, and writes the
preprocessed cache to:

```text
$VERSATIL_ZARR_DIR/bowel_retraction/fixed_viewpoint_phantom/dataset.zarr
```

If this Zarr store does not exist, VersatIL creates it automatically on the
first training run.

## Fixed-Viewpoint Training Configs

Run commands from the VersatIL repository root. These configs use
`src/versatil/hydra_configs/task/dataset_schema/bowel_retraction_v2.yaml`, which loads
`v1/` and `v2/`.

ACT baseline:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act
```

Supervised PhaseACT / MoE policy:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/phase_act
```

Relevant config files:

```text
src/versatil/hydra_configs/end_to_end_training_runs/bowel_retraction/act.yaml
src/versatil/hydra_configs/end_to_end_training_runs/bowel_retraction/phase_act.yaml
src/versatil/hydra_configs/task/dataset_schema/bowel_retraction_v2.yaml
src/versatil/hydra_configs/task/dataloader/bowel_retraction.yaml
```

To disable Weights & Biases for a local smoke test:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/phase_act \
    experiment.use_wandb=false
```

## Random-Viewpoint Training Configs

The random-viewpoint experiments use the fixed-viewpoint data together with the
two random-viewpoint recording folders. These configs use
`src/versatil/hydra_configs/task/dataset_schema/bowel_retraction_v3.yaml`, which expects the
public Hugging Face layout:

```text
$VERSATIL_BOWEL_RETRACTION_DIR/
  v1/
  v2/
  multicam/
    multi_camera_exp/
    multi_camera_exp2/
```

Train the PhaseACT / MoE policy on fixed-plus-random viewpoint data:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/phase_language_act
```

The ACT-style counterpart over the same v3/mixed-viewpoint data is:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act_language
```

Relevant random-viewpoint config files:

```text
src/versatil/hydra_configs/end_to_end_training_runs/bowel_retraction/phase_language_act.yaml
src/versatil/hydra_configs/end_to_end_training_runs/bowel_retraction/act_language.yaml
src/versatil/hydra_configs/task/dataset_schema/bowel_retraction_v3.yaml
src/versatil/hydra_configs/task/dataloader/bowel_retraction_language.yaml
```

For a random-viewpoint-only ablation, override
`task.dataset_schema.dataset_folders` to keep only
`$VERSATIL_BOWEL_RETRACTION_DIR/multicam/multi_camera_exp` and
`$VERSATIL_BOWEL_RETRACTION_DIR/multicam/multi_camera_exp2`, and use a separate
`task.dataset_schema.zarr_path`.

## Inference and Robot Rollouts

This page covers reproducing the VersatIL training runs and checkpoints.
Hardware inference and real robot rollouts require the surgical robot testbed
control stack used in the experiments. That code is expected to be released with
the work described at [arXiv:2603.08490](https://arxiv.org/pdf/2603.08490).
Until that control stack is public, VersatIL alone is sufficient for training
the policies but not for reproducing the physical robot rollouts.
