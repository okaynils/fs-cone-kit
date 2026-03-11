import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
from tqdm import tqdm

from core.data.base import BaseDataset


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class FSOCODataset(BaseDataset):
    def __init__(
        self,
        download_url: str,
        raw_dir: str,
        preprocessed_dir: str,
        class_map: dict,
        debug_mode: bool = True,
        train_limit: int = 150,
        val_limit: int = 20,
        plot_images: list[str] | None = None,
        plot_image_count: int = 4,
        class_colors: dict[str, list[int]] | None = None,
    ):
        super().__init__(
            raw_dir=raw_dir,
            preprocessed_dir=preprocessed_dir,
            class_map=class_map,
            plot_images=plot_images,
            plot_image_count=plot_image_count,
            class_colors=class_colors,
        )

        self.url = download_url
        self.debug_mode = debug_mode
        self.train_limit = train_limit
        self.val_limit = val_limit

        random.seed(42)

    @staticmethod
    def crop_black_borders(img_path, threshold=20):
        """Removes black borders from FSOCO images using OpenCV contours."""
        img = cv2.imread(str(img_path))
        if img is None:
            return None, None

        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return img, (0, 0, w, h)

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_crop, h_crop = cv2.boundingRect(largest_contour)

        # Safety pad
        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        w_crop = min(w - x, w_crop + 2 * pad)
        h_crop = min(h - y, h_crop + 2 * pad)

        return img[y:y + h_crop, x:x + w_crop], (x, y, w_crop, h_crop)

    def _download(self):
        """Downloads and unzips the FSOCO dataset if it isn't already present."""
        if self.raw_dir.exists() and list(self.raw_dir.rglob("*.json")):
            print(f"[{self.__class__.__name__}] Dataset already found in {self.raw_dir}. Skipping download.")
            return

        print(f"[{self.__class__.__name__}] Creating directory: {self.raw_dir}")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.raw_dir / "fsoco_temp.zip"

        print(f"[{self.__class__.__name__}] Downloading dataset from {self.url}...")
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading FSOCO") as t:
            urllib.request.urlretrieve(self.url, filename=zip_path, reporthook=t.update_to)

        print(f"[{self.__class__.__name__}] Extracting dataset (this might take a moment)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.namelist(), desc="Extracting"):
                zip_ref.extract(member, self.raw_dir)

        print(f"[{self.__class__.__name__}] Cleaning up zip file...")
        zip_path.unlink()

    def _preprocess(self):
        """Crops, converts annotations, and structures the data for YOLO."""
        # Nuke existing preprocessed data to be safe and start fresh
        if self.prep_dir.exists():
            shutil.rmtree(self.prep_dir)

        for split in ['train', 'val']:
            (self.prep_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.prep_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

        json_files = list(self.raw_dir.rglob("ann/*.json"))
        random.shuffle(json_files)

        print(f"[{self.__class__.__name__}] Found {len(json_files)} annotations. Processing...")

        counts = {"train": 0, "val": 0}

        for json_file in tqdm(json_files, desc="Processing Images"):
            # 1. Determine Split (Random 80/20)
            split_name = 'train' if random.random() < 0.8 else 'val'

            # 2. Check Limits
            if self.debug_mode:
                limit = self.train_limit if split_name == 'train' else self.val_limit
                if counts[split_name] >= limit:
                    continue
                if counts['train'] >= self.train_limit and counts['val'] >= self.val_limit:
                    break

            # 3. Locate Source Image
            image_name = json_file.stem
            team_name = json_file.parent.parent.name
            src_img_path = json_file.parent.parent / "img" / image_name

            if not src_img_path.exists():
                possible = list(src_img_path.parent.glob(f"{Path(image_name).stem}.*"))
                if possible:
                    src_img_path = possible[0]
                else:
                    continue

            # 4. Load & Crop
            cropped_img, crop_rect = self.crop_black_borders(src_img_path)
            if cropped_img is None:
                continue

            # Skip empty images (e.g. completely black)
            if cropped_img.shape[0] < 10 or cropped_img.shape[1] < 10:
                continue

            crop_x, crop_y, crop_w, crop_h = crop_rect

            # 5. Process Labels
            with open(json_file, 'r') as f:
                data = json.load(f)

            yolo_lines = []
            for obj in data.get('objects', []):
                if obj['classTitle'] not in self.class_map:
                    continue
                class_id = self.class_map[obj['classTitle']]

                pts = obj['points']['exterior']
                x_min, y_min = min(pts[0][0], pts[1][0]), min(pts[0][1], pts[1][1])
                x_max, y_max = max(pts[0][0], pts[1][0]), max(pts[0][1], pts[1][1])

                # Adjust to crop
                new_x_min = max(0, x_min - crop_x)
                new_y_min = max(0, y_min - crop_y)
                new_x_max = min(crop_w, x_max - crop_x)
                new_y_max = min(crop_h, y_max - crop_y)

                # Discard invalid boxes
                if (new_x_max - new_x_min) < 1 or (new_y_max - new_y_min) < 1:
                    continue

                # Normalize
                x_center = ((new_x_min + new_x_max) / 2) / crop_w
                y_center = ((new_y_min + new_y_max) / 2) / crop_h
                w_norm = (new_x_max - new_x_min) / crop_w
                h_norm = (new_y_max - new_y_min) / crop_h

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # only save if valid labels exist
            if yolo_lines:
                new_filename = f"{team_name}_{Path(image_name).stem}.jpg"
                cv2.imwrite(str(self.prep_dir / 'images' / split_name / new_filename), cropped_img)

                label_path = self.prep_dir / 'labels' / split_name / (Path(new_filename).stem + ".txt")
                with open(label_path, 'w') as out_f:
                    out_f.write('\n'.join(yolo_lines))

                counts[split_name] += 1

        print(f"[{self.__class__.__name__}] Done! Created {counts['train']} train images and {counts['val']} val images.")
