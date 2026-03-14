from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

load_dotenv()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    # prep dataset
    dataset_manager = instantiate(cfg.dataset)
    dataset_yaml_path = dataset_manager.prepare()
    val_image_dir = str(dataset_manager.prep_dir / "images" / "val")
    val_image_paths = dataset_manager.get_plot_image_paths()
    class_colors = dataset_manager.get_class_colors()

    with open_dict(cfg):
        cfg.trainer.args.data = dataset_yaml_path

    # prep metrics
    instantiated_metrics = []
    if "metrics" in cfg and cfg.metrics:
        for metric_name, metric_cfg in cfg.metrics.items():
            if metric_cfg is None:
                continue
            print(f"Preparing metric: {metric_name}")
            instantiated_metrics.append(instantiate(metric_cfg))

    # prep and setup loggers
    aggregated_callbacks = {}
    mlflow_enabled = False
    mlflow_tracking_uri = None
    logger_config_dir = Path(__file__).resolve().parents[1] / "configs" / "logger"

    if "loggers" in cfg and cfg.loggers:
        for logger_name, logger_cfg in cfg.loggers.items():
            print(f"Setting up logger: {logger_name}")
            logger_default_cfg_path = logger_config_dir / f"{logger_name}.yaml"
            effective_logger_cfg = logger_cfg

            if logger_default_cfg_path.exists():
                effective_logger_cfg = OmegaConf.merge(
                    OmegaConf.load(logger_default_cfg_path),
                    logger_cfg,
                )

            logger = instantiate(effective_logger_cfg)
            logger.set_metrics(instantiated_metrics)

            logger.setup(
                val_image_dir=val_image_dir,
                class_map=cfg.dataset.class_map,
                experiment_name=cfg.model.name,
                run_name=cfg.run_name,
                val_image_paths=val_image_paths,
                class_colors=class_colors,
            )

            mlflow_enabled = mlflow_enabled or getattr(logger, "uses_mlflow", False)
            if mlflow_tracking_uri is None:
                mlflow_tracking_uri = getattr(logger, "tracking_uri", None)

            # collect callbacks
            for event, func in logger.get_callbacks().items():
                if event not in aggregated_callbacks:
                    aggregated_callbacks[event] = []
                aggregated_callbacks[event].append(func)

    # taining!
    trainer = instantiate(cfg.trainer)
    trainer.setup(
        model_weights=cfg.model.weights,
        experiment_name=cfg.model.name,
        run_name=cfg.run_name,
        callbacks=aggregated_callbacks,
        enable_mlflow=mlflow_enabled,
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    trainer.train()


if __name__ == "__main__":
    main()
