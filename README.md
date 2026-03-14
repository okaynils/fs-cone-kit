<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_light.svg">
  <img alt="fs-cone-kit logo" src="/docs/logo_dark.svg" width="50%" height="50%">
</picture>

# fs-cone-kit

Train a cone detector for Formula Student Driverless.

</div>

![object detection demo](/docs/obj_detection_demo.gif)

I am building this at the FS Driverless team at Linköping University. I maintain it because useful tooling should not stay trapped inside one team.

This is a small training pipeline around Ultralytics YOLO (for now).
It downloads and preprocesses [FSOCO](https://fsoco.github.io/fsoco-dataset/) out of the box.
If your team already has a YOLO dataset, use that instead.

No notebooks. No clickops. Run the command and train the model.

## What it does

- trains a YOLO cone detector
- logs metrics and prediction images during training
- exports ONNX after training
- ships with an FSOCO pipeline so you can get a baseline fast
- uses Hydra configs, so most changes are one command-line override

## Requirements

- Python 3.12
- `uv`

## Quick start

Install dependencies:

```bash
uv sync
```

Run a small sanity-check training job:

```bash
uv run -m core.train
```

That uses FSOCO in debug mode. It is meant to prove the pipeline works.

Artifacts go to `outputs/<run_name>/`.
Weights end up in `outputs/<run_name>/ultralytics_files/weights/`.
If ONNX export is enabled, exported models are written after training finishes.

## Train for real

The default dataset config is in `configs/dataset/fsoco.yaml`.

For a full FSOCO run, set:

```yaml
debug_mode: false
```

You will probably also want to change the training knobs in `configs/trainer/ultralytics.yaml`:

```yaml
args:
  epochs: 100
  imgsz: 640
  batch: 16
```

You can also override them from the command line:

```bash
uv run -m core.train trainer.args.epochs=100 trainer.args.batch=32 model.weights=yolo11s.pt
```

This is Hydra. The command line is the UI.

## Train on your own dataset

If your dataset is already in YOLO format, you do not need to touch the code.

Put your data here, like this:

```text
data/myteam/preprocessed/
  dataset.yaml
  images/train/
  images/val/
  labels/train/
  labels/val/
```

A minimal `dataset.yaml` looks like this:

```yaml
path: data/myteam/preprocessed
train: images/train
val: images/val
names:
  0: blue_cone
  1: yellow_cone
  2: orange_cone
  3: large_orange_cone
  4: unknown_cone
```

Then point the pipeline at it:

```bash
uv run -m core.train dataset.preprocessed_dir=data/myteam/preprocessed
```

That works because the pipeline skips download and preprocessing when `preprocessed_dir` already contains:

- `dataset.yaml`
- `images/train/`

If your classes are different, change `class_map` and `class_colors` in `configs/dataset/fsoco.yaml` so they match your `dataset.yaml`.
Keep the mapping consistent. The repo is not magic.

If you want to keep your team config separate, copy `configs/dataset/fsoco.yaml` to a new file and change:

- `preprocessed_dir`
- `class_map`
- `class_colors`
- `debug_mode`

## Supported class setup

The default config uses five classes:

- blue_cone
- yellow_cone
- orange_cone
- large_orange_cone
- unknown_cone

If your team uses three classes, use three classes.
Just keep `dataset.yaml`, `class_map`, and labels aligned.

## Logging

WandB is enabled by default in `configs/config.yaml`.
If you want local-only logs, set:

```bash
$env:WANDB_MODE='offline'
uv run -m core.train
```

MLflow is optional.
Connection details live in `.env`.
Start from `.env.example`:

```env
MLFLOW_TRACKING_URI=
MLFLOW_TRACKING_TOKEN=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
```

If you enable the GitLab MLflow logger, the tracking URI is read from `MLFLOW_TRACKING_URI`.

## Outputs

After a run, look here:

- `outputs/<run_name>/train.log`
- `outputs/<run_name>/ultralytics_files/weights/best.pt`
- `outputs/<run_name>/ultralytics_files/weights/last.pt`

The WandB logger also logs side-by-side ground truth vs prediction images from validation samples.

## Project layout

```text
configs/                 Hydra configs
configs/dataset/         dataset configs
configs/trainer/         training configs
configs/logger/          logging configs
core/data/               dataset logic
core/trainers/           trainer backends
core/loggers/            logger integrations
core/metrics/            metric extraction
core/train.py            training entrypoint
```

## When you need to change code

If your data is not already in YOLO format, copy `core/data/fsoco.py` and make your own dataset adapter.
That is the place to handle download, conversion, cropping, relabeling, whatever your data needs.

## TL;DR

1. install with `uv sync`
2. run `uv run -m core.train` to make sure the pipeline works
3. point `dataset.preprocessed_dir` at your YOLO dataset
4. align `class_map` with your labels
5. train

That is it.



