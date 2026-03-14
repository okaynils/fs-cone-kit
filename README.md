## Run the training

To run the training

```bash
uv run -m core.train
```

To disable debug mode to do a full training run go to `configs/dataset/fsoco.yaml` and change `debug_mode` to `false`.

## Environment variables

Copy `.env.example` to `.env` and fill in any values you want to use locally.

The GitLab MLflow logger reads its tracking URI from `MLFLOW_TRACKING_URI`, and authentication can be provided with either `MLFLOW_TRACKING_TOKEN` or the pair `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`.
