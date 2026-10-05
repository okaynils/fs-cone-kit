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

- trains a YOLO cone detector from FSOCO or your own YOLO dataset
- logs metrics, prediction images, and experiment settings
- exports and checks ONNX models, then measures FP16 and INT8 quantization
- evaluates, benchmarks, and gates a release before writing a model bundle
- uses Hydra configs so you can change most settings from the command line

## Quick start

You need Python 3.12 and `uv`.

```bash
uv sync
uv run -m core.train
```

The default run uses FSOCO in debug mode. It is meant to prove the pipeline works.
Artifacts go to `outputs/<run_name>/`.

## Keep going

The [how-to guide](docs/howto.md) covers full training, your own dataset, logging,
model comparison, ONNX checks, quantization, release bundles, failure studies,
and reproducing a run. The command line is the UI.
