from abc import ABC, abstractmethod
from pathlib import Path
import yaml

class BaseDataset(ABC):
    def __init__(self, raw_dir: str, preprocessed_dir: str, class_map: dict, **kwargs):
        self.raw_dir = Path(raw_dir)
        self.prep_dir = Path(preprocessed_dir)
        self.class_map = class_map

    def prepare(self) -> str:
        """
        The main orchestrator. Checks if data is ready. If not, sets it up.
        Returns the path to the YOLO dataset.yaml file.
        """
        yaml_path = self.prep_dir / "dataset.yaml"
        
        if self._is_ready(yaml_path):
            print(f"[{self.__class__.__name__}] Dataset already prepared at {self.prep_dir}")
            return str(yaml_path)

        print(f"[{self.__class__.__name__}] Dataset not found or incomplete. Preparing...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.prep_dir.mkdir(parents=True, exist_ok=True)

        self._download()
        self._preprocess()
        self._create_yolo_yaml(yaml_path)
        
        return str(yaml_path)

    def _is_ready(self, yaml_path: Path) -> bool:
        """Checks if the dataset is already fully processed."""
        return yaml_path.exists() and (self.prep_dir / "images" / "train").exists()

    @abstractmethod
    def _download(self):
        """Logic to download and extract the dataset."""
        pass

    @abstractmethod
    def _preprocess(self):
        """Logic to crop, convert, and format annotations."""
        pass

    def _create_yolo_yaml(self, yaml_path: Path):
        """Automatically generates the data.yaml file YOLO expects."""
        yaml_content = {
            "path": str(self.prep_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {v: k for k, v in self.class_map.items()} 
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)