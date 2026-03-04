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

    # prep and setup loggers
    aggregated_callbacks = {}
    
    if "loggers" in cfg and cfg.loggers:
        for logger_name, logger_cfg in cfg.loggers.items():
            print(f"Setting up logger: {logger_name}")
            logger = instantiate(logger_cfg)
            
            logger.setup(
                val_image_dir=val_image_dir, 
                class_map=cfg.dataset.class_map,
                experiment_name=cfg.model.name,
                run_name=cfg.run_name
            )
            
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
        callbacks=aggregated_callbacks
    )
    trainer.train()

if __name__ == "__main__":
    main()