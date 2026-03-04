import hydra
from omegaconf import DictConfig, open_dict
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # prep dataset
    print(f"Setting up dataset: {cfg.dataset._target_}")
    dataset_manager = instantiate(cfg.dataset)
    dataset_yaml_path = dataset_manager.prepare()

    # inject dataset path into trainer config
    with open_dict(cfg):
        cfg.trainer.args.data = dataset_yaml_path

    # instantiate trainer
    print(f"Instantiating trainer: {cfg.trainer._target_}")
    trainer = instantiate(cfg.trainer)
    
    # train!
    trainer.setup(model_name=cfg.model.name)
    trainer.train()

if __name__ == "__main__":
    main()