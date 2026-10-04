<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_light.svg">
  <img alt="fs-cone-kit logo" src="/docs/logo_dark.svg" width="50%" height="50%">
</picture>

*Train a cone detector for Formula Student Driverless.*

</div>

I am building this at the FS Driverless team at Linköping University. I maintain it because useful tooling should not stay trapped inside one team.

This is a small training pipeline around Ultralytics YOLO (for now).
It downloads and preprocesses [FSOCO](https://fsoco.github.io/fsoco-dataset/) out of the box.
If your team already has a YOLO dataset, use that instead.

No notebooks. No clickops. Run the command and train the model.

<p align="center">
  <img alt="object detection demo" src="/docs/obj_detection_demo.gif" width="80%">
</p>

## What it does

- trains a YOLO cone detector
- logs metrics and prediction images during training
- exports ONNX after training and checks it against the checkpoint
- quantizes to FP16 and INT8 and reports what that cost
- ships with an FSOCO pipeline so you can get a baseline fast
- uses Hydra configs, so most changes are one command-line override
- records enough local metadata to reproduce and compare experiments
- evaluates final test results separately from training validation
- benchmarks inference latency and throughput

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

Every prepared FSOCO dataset uses the configured `split_seed` and writes a
`split_manifest.json`. Reusing the same dataset config reuses the same train,
validation, and test split.

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
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

A minimal `dataset.yaml` looks like this:

```yaml
path: data/myteam/preprocessed
train: images/train
val: images/val
test: images/test
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

The test split is optional while training, but required for the final-test
evaluation command below. Do not tune a model against it.

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
- `outputs/<run_name>/ultralytics_files/weights/best.onnx`
- `outputs/<run_name>/ultralytics_files/results.csv`
- `outputs/<run_name>/experiment/config.yaml`
- `outputs/<run_name>/experiment/metadata.json`
- `outputs/<run_name>/experiment/dataset.json`
- `outputs/<run_name>/experiment/splits.json`
- `outputs/<run_name>/experiment/training.json`

The WandB logger also logs side-by-side ground truth vs prediction images from validation samples.

The saved config is fully resolved. Credential-looking fields and credentials
embedded in URLs are replaced with `<redacted>`.

## Compare two models

Give both runs a stable name. Keep the dataset overrides identical:

```bash
WANDB_MODE=offline uv run -m core.train run_name=yolo11n-640 model.name=yolo11n model.weights=yolo11n.pt trainer.args.imgsz=640
WANDB_MODE=offline uv run -m core.train run_name=yolo11s-640 model.name=yolo11s model.weights=yolo11s.pt trainer.args.imgsz=640
```

Training validation chooses the checkpoint. Final test evaluation measures it:

```bash
uv run -m core.evaluate outputs/yolo11n-640 --split test --device cpu
uv run -m core.evaluate outputs/yolo11s-640 --split test --device cpu
```

The JSON files include mAP50, mAP50-95, precision, recall, per-class results,
the confusion matrix, parameter count, and checkpoint size. They also include
the dataset fingerprint. A changed split fails evaluation instead of quietly
producing a misleading number.

Benchmark both checkpoints under the same conditions:

```bash
uv run -m core.benchmark outputs/yolo11n-640 --device cpu --precision fp32 --batch-size 1 --input-size 640
uv run -m core.benchmark outputs/yolo11s-640 --device cpu --precision fp32 --batch-size 1 --input-size 640
```

This uses warm-up runs and synchronized timing where the device supports it.
It reports median latency, p95 latency, and throughput. The benchmark records
hardware and protocol details. It measures synthetic-tensor model inference,
not camera-to-actuator vehicle performance.

Build the report without WandB or MLflow:

```bash
uv run -m core.compare outputs/yolo11n-640 outputs/yolo11s-640 --output-dir comparison/yolo11-640
```

That writes `comparison.csv` and `comparison.md`. Accuracy rows from different
dataset fingerprints or evaluation settings are marked non-comparable. Timing
rows retain a benchmark context ID, and the command warns when the hardware or
protocol differs.

## Check the exported model

Training exports `best.onnx` and `last.onnx` next to the checkpoints.
A failed export fails the run. The training record is written first, so the
checkpoints stay usable.

An exported file is not proof. Check it against the checkpoint it came from:

```bash
uv run -m core.parity outputs/yolo11n-640 --device cpu
```

This runs `best.pt` through Ultralytics and `best.onnx` through onnxruntime
with the plain NumPy and OpenCV preprocessing in `core/deployment.py`. Same
test images, same thresholds. It reports box deviation in pixels, confidence
differences, class flips, and boxes only one model found. It exits non-zero
when a tolerance in `configs/release/default.yaml` is exceeded.

Tolerances are Hydra values:

```bash
uv run -m core.parity outputs/yolo11n-640 parity.image_count=64 parity.confidence=0.3
```

Evaluate and benchmark the exported file with the usual commands:

```bash
uv run -m core.evaluate outputs/yolo11n-640 --model ultralytics_files/weights/best.onnx --device cpu
uv run -m core.benchmark outputs/yolo11n-640 --model ultralytics_files/weights/best.onnx --device cpu
```

Both record the runtime they used. The evaluation lands in
`experiment/evaluations/test_best_onnx.json`. `core.compare` still reads the
checkpoint's `test.json`.

## Quantize and measure the cost

FP16 halves the file. INT8 shrinks it again. Neither is free. Measure it:

```bash
uv run -m core.quantize outputs/yolo11n-640 --precision fp16 int8 --device cpu
```

This writes `best_fp16.onnx` and `best_int8.onnx` next to `best.onnx`,
evaluates all three on the test split, and reports every metric against the
fp32 export. The report goes to `experiment/quantization/best.json` and
`best.md`.

INT8 calibration uses train images only, evenly spaced through the sorted
split. The list is saved in `best_int8.calibration.json`. Only `Conv` layers are
quantized. The YOLO head mixes pixel boxes with 0-1 scores, and one INT8 scale
for both erases the scores.

If `study/small-cones/manifest.json` exists for the same split and passed its
leakage audit, the report includes the study slices. Small cones are the ones
you can least afford to lose.

Rows from a different split or different evaluation settings are marked
non-comparable and get no delta.

## Study small-cone failures

The first failure study asks whether stronger scale augmentation helps on
images with small cones. The plan and acceptance criteria are fixed in
[`docs/failure-study-plan.md`](docs/failure-study-plan.md).

Train the baseline and intervention:

```bash
uv run -m core.train +study=small_cones_baseline
uv run -m core.train +study=small_cones_scale
```

Both runs use YOLO11n, seed 42, 50 epochs, and the same team-grouped FSOCO
split. The intervention changes only Ultralytics `scale` from `0.5` to `0.9`.
The configs disable external loggers.

Prepare the fixed slices and inspect the leakage audit:

```bash
uv run -m core.study prepare outputs/small-cones-baseline-seed42
```

This writes `study/small-cones/manifest.json`. The `small_cones` slice contains
test images with at least one ground-truth box covering at most 0.5% of the
image. `ordinary` is the disjoint remainder. Box size is a proxy for distance,
not a distance measurement.

The audit checks source-team overlap, exact duplicates, and perceptual
near-duplicates across splits. Evaluation stops if it finds leakage. FSOCO does
not expose recording or track IDs, so the audit cannot verify those groups.

Evaluate both saved checkpoints:

```bash
uv run -m core.study evaluate outputs/small-cones-baseline-seed42 --device cpu
uv run -m core.study evaluate outputs/small-cones-scale-seed42 --device cpu
```

Build the report from the saved results and predictions:

```bash
uv run -m core.study report \
  study/small-cones/results/small-cones-baseline-seed42.json \
  study/small-cones/results/small-cones-scale-seed42.json
```

The report includes full-set and slice mAP50-95, recall, supported per-class
results, annotation counts, and a deterministic false-positive and
false-negative gallery. One run per condition does not measure run-to-run
variation. Repeat both configs with new seeds before treating a delta as
stable.

Keep the dataset split fixed when you repeat training:

```bash
uv run -m core.train +study=small_cones_baseline seed=43 run_name=small-cones-baseline-seed43
uv run -m core.train +study=small_cones_scale seed=43 run_name=small-cones-scale-seed43
```

The study configs pin `dataset.split_seed=42`. The override changes the
training seed only.

## Reproduce or resume a run

Start a new experiment from the saved resolved config:

```bash
uv run -m core.reproduce outputs/yolo11n-640 --local-only
```

`--local-only` disables external loggers. It does not change the training or
dataset settings.

Resume an interrupted run in place from its `last.pt`:

```bash
uv run -m core.reproduce outputs/yolo11n-640 --resume --local-only
```

The experiment directory is the unit of work. Copy it and you keep the config,
split identity, metrics, evaluation, benchmarks, and weights together.

## Tests

Run the unit and CPU-only artifact smoke tests:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m tests.cpu_smoke
```

The pipeline smoke test builds YOLO11n from its packaged architecture, trains
for one epoch on generated images, exports ONNX, checks parity, quantizes,
evaluates, benchmarks, and exports a report.
It downloads nothing. Full FSOCO training and GPU benchmarking are separate
checks because they require the dataset, model weights, and suitable hardware.

## Project layout

```text
configs/                 Hydra configs
configs/dataset/         dataset configs
configs/trainer/         training configs
configs/logger/          logging configs
configs/release/         export checks and release settings
core/data/               dataset logic
core/trainers/           trainer backends
core/loggers/            logger integrations
core/metrics/            metric extraction
core/deployment.py       the ONNX preprocessing contract and parity checks
core/train.py            training entrypoint
```

## When you need to change code

If your data is not already in YOLO format, copy `core/data/fsoco.py` and make your own dataset adapter.
That is the place to handle download, conversion, cropping, relabeling, whatever your data needs.

## tl;dr

1. install with `uv sync`
2. run `uv run -m core.train` to make sure the pipeline works
3. point `dataset.preprocessed_dir` at your YOLO dataset
4. align `class_map` with your labels
5. train

That is it.
