from abc import ABC, abstractmethod
from pathlib import Path

import yaml


class BaseDataset(ABC):
    def __init__(
        self,
        raw_dir: str,
        preprocessed_dir: str,
        class_map: dict,
        plot_images: list[str] | None = None,
        plot_image_count: int = 4,
        class_colors: dict[str, list[int]] | None = None,
        **kwargs,
    ):
        self.raw_dir = Path(raw_dir)
        self.prep_dir = Path(preprocessed_dir)
        self.class_map = class_map
        self.plot_images = plot_images or []
        self.plot_image_count = plot_image_count
        self.class_colors = class_colors or {}

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

    def get_plot_image_paths(self) -> list[str]:
        val_image_dir = self.prep_dir / "images" / "val"
        available_images = sorted(val_image_dir.glob("*.jpg"))
        if not available_images:
            return []

        selected_paths: list[Path] = []
        available_by_name = {path.name: path for path in available_images}
        available_by_stem = {path.stem: path for path in available_images}

        for plot_image in self.plot_images:
            candidate_path = Path(plot_image)
            if not candidate_path.is_absolute():
                candidate_path = val_image_dir / plot_image

            resolved_candidate = None
            if candidate_path.exists():
                resolved_candidate = candidate_path
            elif plot_image in available_by_name:
                resolved_candidate = available_by_name[plot_image]
            elif plot_image in available_by_stem:
                resolved_candidate = available_by_stem[plot_image]

            if resolved_candidate is None:
                print(f"[{self.__class__.__name__}] Plot image '{plot_image}' was not found in {val_image_dir}.")
                continue

            if resolved_candidate not in selected_paths:
                selected_paths.append(resolved_candidate)

        target_count = max(0, self.plot_image_count)
        for image_path in available_images:
            if target_count and len(selected_paths) >= target_count:
                break
            if image_path not in selected_paths:
                selected_paths.append(image_path)

        if target_count:
            selected_paths = selected_paths[:target_count]

        return [str(path) for path in selected_paths]

    def get_class_colors(self) -> dict[int, tuple[int, int, int]]:
        colors: dict[int, tuple[int, int, int]] = {}

        for class_name, color in self.class_colors.items():
            class_id = self.class_map.get(class_name)
            if class_id is None:
                continue
            if not isinstance(color, (list, tuple)) or len(color) != 3:
                continue
            colors[class_id] = tuple(int(channel) for channel in color)

        return colors

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
