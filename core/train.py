import hydra
from omegaconf import DictConfig, open_dict
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # prep dataset
    dataset_manager = instantiate(cfg.dataset)
    dataset_yaml_path = dataset_manager.prepare()
    val_image_dir = str(dataset_manager.prep_dir / "images" / "val")

    with open_dict(cfg):
        cfg.trainer.args.data = dataset_yaml_path

    # prep logger
    logger = None
    callbacks = {}
    if "logger" in cfg:
        print(f"Setting up logger: {cfg.logger._target_}")
        logger = instantiate(cfg.logger)
        
        logger.setup(
            val_image_dir=val_image_dir, 
            class_map=cfg.dataset.class_map,
            experiment_name=cfg.model.name,
            run_name=cfg.run_name
        )
        callbacks = logger.get_callbacks()

    # training!
    trainer = instantiate(cfg.trainer)
    trainer.setup(
        model_weights=cfg.model.weights,
        experiment_name=cfg.model.name,
        run_name=cfg.run_name,
        callbacks=callbacks
    )
    trainer.train()

if __name__ == "__main__":
    main()