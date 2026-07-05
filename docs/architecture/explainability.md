# Explainability

The explainability package (`versatil.explainability`) produces visual attribution maps for trained policies: per-camera heatmaps showing which image regions drove the predicted actions. It runs post hoc on any checkpoint over offline dataset samples or live during inference.

Three attribution methods are supported, selected through `explanation_types`:

| Method | `ExplanationType` value | How it works | Paper |
|---|---|---|---|
| Grad-CAM | `gradcam` | Gradient of the prediction score w.r.t. a target activation, channel-averaged as weights | [Selvaraju et al., ICCV 2017](https://arxiv.org/abs/1610.02391) |
| Grad-CAM++ | `gradcam++` | Higher-order gradient weighting for sharper multi-instance maps | [Chattopadhay et al., WACV 2018](https://arxiv.org/abs/1710.11063) |
| Ablation-CAM | `ablation_cam` | Perturbation-based: each activation channel is zeroed and weighted by the resulting score drop, no backpropagation | [Desai & Ramaswamy, WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/html/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.html) |

CNN feature maps and ViT patch-tokens are both compatible. Token activations are reshaped to their patch grid before weighting.

## Running the endpoint

```bash
python -m versatil.endpoints.explain \
    checkpoint_path=/path/to/training/checkpoint \
    split=val \
    sample_stride=50 \
    max_samples=16
```

`checkpoint_path` is the training run directory containing `config.yaml` and the model checkpoint. Outputs are written to `checkpoint_path/explainability/<timestamp>/` unless `output_directory` is set. All settings live on [`ExplainabilityConfig`][versatil.configs.explainability.ExplainabilityConfig] and can be overridden from the CLI.

Common overrides:

```bash
# Restrict methods, cameras, or visual modules
explanation_types='[gradcam]'
target_camera_keys='[left]'
target_vision_module_names='[left_rgb_encoder]'

# Explain different data than the checkpoint was trained on
data_path_override=/path/to/other_data.zarr

# Save raw heatmap tensors in addition to overlays
writer.save_raw_heatmaps=true
```

## Pipeline

```
ExplainabilityRunner
  -> FloatCheckpointLoader                 (restore policy, normalizer, tokenizer)
  -> ExplanationSource                     (dataset windows or live inference windows)
  -> per batch, per explanation type:
       resolve_camera_explanation_targets  (discover visual modules and cameras)
       compute heatmaps                    (Grad-CAM/Grad-CAM++/Ablation-CAM)
  -> ExplanationWriter                     (overlay images, optional raw .pt tensors)
```

[`ExplainabilityRunner`][versatil.explainability.runner.ExplainabilityRunner] orchestrates the loop. Sources yield [`ExplanationBatch`][versatil.explainability.sources.typedefs.ExplanationBatch] objects carrying the observation window, optional action labels, display images for overlays, and per-batch metadata.

## Explanation sources

### Dataset source (`source=dataset`)

[`DatasetExplanationSource`][versatil.explainability.sources.dataset.DatasetExplanationSource] samples episodic windows through the same `EpisodicDataset` used in training, with the checkpoint's normalizer and tokenizer attached and augmentations disabled. Sampling is deterministic: every `sample_stride`-th window in dataset order, capped by `max_samples`.

- `split` selects `train`, `val`, or `all` (the `all` split reuses the training-side splitter with `val_ratio=0`, preserving every other sampling setting).
- `data_path_override` explains data other than the training set. A path ending in `.zarr` is sampled directly. A non-zarr path must be raw data in the same schema format as the checkpoint (CSV episode folders, HDF5 file, LeRobot root) and is converted to `offline_dataset.zarr` beside the override path. A list is only valid for raw schemas that accept multiple inputs; multiple zarr paths are rejected.

Dataset batches are already normalized and tokenized, so attribution runs with `preprocess_observation=False`.

### Online inference source (`source=online_inference`)

[`OnlineInferenceExplanationSource`][versatil.explainability.sources.online.OnlineInferenceExplanationSource] attaches to the same [`InferenceClient`][versatil.inference.inference_client.InferenceClient] loop used by the deployment endpoint. The client handles transport, preprocessing, and observation buffering; whenever a buffered observation window is ready for policy inference, the source receives the exact batch passed to the policy and explains every `sample_stride`-th timestep.

Client settings come from the shared [`InferenceClientConfig`][versatil.configs.inference_client.InferenceClientConfig] under the `online` key:

```bash
python -m versatil.endpoints.explain \
    checkpoint_path=/path/to/checkpoint \
    source=online_inference \
    online.model_server_address=10.0.0.1 \
    online.model_server_port=5556 \
    sample_stride=10 \
    max_samples=100
```


## Attribution targets

Targets are discovered automatically from the policy:

- **Encoding-pipeline encoders** (including conditional encoders) that expose `get_explainability_targets()`.
- **Decoder-owned VLM vision towers** for VLA policies: the `decoder.vlm_backbone` module itself and each entry of its `vision_encoders` list.

Each module declares its capture metadata through [`VisionExplanationTarget`][versatil.models.encoding.explainability.VisionExplanationTarget]: the target layer, whether it produces a spatial feature map (`NCHW`/`NHWC`) or a ViT token sequence (`NLC`), the tuple output index when the layer returns several tensors, and for token targets the prefix-token count and patch grid. Attribution hooks that layer, converts the captured activation to `NCHW`, computes the map, and resizes it back to the camera image with bicubic interpolation.

When one module serves several cameras, a capture mode routes the hook to the right camera: separate forward calls per camera (`per_camera_call`), a camera-stacked batch dimension (`stacked_camera_batch`), or a single call (`single_call`). When multiple visual modules can explain the same camera, their normalized maps are averaged; use `target_vision_module_names` to isolate one module.

## Prediction objective

The scalar score that is attributed depends on the decoder:

- **Continuous-action decoders**: the norm of all concatenated normalized action predictions. A custom `output_selector` callable can replace this when calling the attribution functions from Python ([`compute_gradient_maps_for_policy`][versatil.explainability.attribution.gradients.compute_gradient_maps_for_policy], [`compute_ablation_maps_for_policy`][versatil.explainability.attribution.ablation.compute_ablation_maps_for_policy]).
- **Tokenized-action decoders** (autoregressive VLAs): the teacher-forced mean log-likelihood of the action tokens, ignoring padded positions. Dataset batches use the true tokenized actions; online batches have no labels, so an unhooked inference pass generates pseudo-target tokens first.

Decoder encoder-prefix caches are disabled during attribution forwards so that hooked activations always contribute to the scored prediction.

Ablation-CAM controls its memory/compute trade-off through `channel_batch_size`: that many activation channels are ablated per policy forward by repeating the batch, so peak memory scales with `batch_size * channel_batch_size`.

## Outputs

[`ExplanationWriter`][versatil.explainability.writer.ExplanationWriter] writes under `output_directory/<source>/` (dataset runs add a `<split>/` level):

- **Overlays** (default on): one image per sample, timestep, method, and camera, named `sample_<index>_t<t>_<method>_<camera>.png` for dataset runs and `env_<index>_step_<t>_...` for online runs. `writer.image_weight` blends the original image with the JET-colormapped heatmap; `writer.overlay_image_format` selects any OpenCV-writable extension.
- **Raw heatmaps** (`writer.save_raw_heatmaps=true`): one `batch_<n>_<method>.pt` file per batch containing the batch metadata and `(B, T, H, W)` heatmap tensors keyed by camera, for downstream analysis.

## Limitations

- Each visual module must expose exactly one compatible explainability target. Per-target selection inside a module is not yet configurable.
- Token targets without an explicit `patch_grid` require a perfect-square patch count to infer the grid.
- Attribution runs full policy forwards (plus one forward per `channel_batch_size` channels for Ablation-CAM), so online mode adds latency to each explained inference step.
- There is no data format convention for explaining recorded inference rollouts, since recording formats depend on the simulator or hardware setup. The recommended workflow is currently to convert recorded rollouts into the same dataset schema used for training, then explain them through the dataset source by pointing `data_path_override` at the converted data.