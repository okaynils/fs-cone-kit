import json
import shutil
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import urllib.request
import zipfile

DOWNLOAD_URL = "http://fsoco.cs.uni-freiburg.de/src/download_boxes_train.php"

# --- DEBUG SETTINGS ---
# Set to True to only process a few images
DEBUG_MODE = True
TRAIN_LIMIT = 150
VAL_LIMIT = 20

random.seed(42)

CLASS_MAP = {
    "blue_cone": 0,
    "yellow_cone": 1,
    "orange_cone": 2,
    "large_orange_cone": 3,
    "unknown_cone": 4
}

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract(url, extract_to):
    """Downloads and unzips the dataset if it isn't already present."""
    extract_path = Path(extract_to)
    
    # Check if data already exists by looking for JSON files (assumes standard FSOCO structure)
    if extract_path.exists() and list(extract_path.rglob("*.json")):
        print(f"Dataset already found in {extract_path}. Skipping download.")
        return

    print(f"Creating directory: {extract_path}")
    extract_path.mkdir(parents=True, exist_ok=True)
    zip_path = extract_path / "fsoco_temp.zip"

    print(f"Downloading dataset from {url}...")
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading FSOCO") as t:
        urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)

    print("Extracting dataset (this might take a moment)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract files, wrapping in tqdm for a progress bar
        for member in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(member, extract_path)
            
    print("Cleaning up zip file...")
    zip_path.unlink()
    print("Download and extraction complete!")

def crop_black_borders(img_path, threshold=20):
    img = cv2.imread(str(img_path))
    if img is None: return None, None
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return img, (0, 0, w, h)
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w_crop, h_crop = cv2.boundingRect(largest_contour)
    
    # Safety pad
    pad = 2
    x = max(0, x - pad)
    y = max(0, y - pad)
    w_crop = min(w - x, w_crop + 2*pad)
    h_crop = min(h - y, h_crop + 2*pad)
    
    return img[y:y+h_crop, x:x+w_crop], (x, y, w_crop, h_crop)

def convert_fsoco(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Nuke existing to be safe
    if output_path.exists():
        shutil.rmtree(output_path)

    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.rglob("ann/*.json"))
    # Shuffle so our small sample has variety
    random.shuffle(json_files)
    
    print(f"Found {len(json_files)} annotations. Processing subset...")

    counts = {"train": 0, "val": 0}

    for json_file in tqdm(json_files, desc="Processing Images"):
        # 1. Determine Split (Random 80/20)
        split_name = 'train' if random.random() < 0.8 else 'val'
        
        # 2. Check Limits
        if DEBUG_MODE:
            limit = TRAIN_LIMIT if split_name == 'train' else VAL_LIMIT
            if counts[split_name] >= limit:
                continue
            # If both limits reached, stop completely
            if counts['train'] >= TRAIN_LIMIT and counts['val'] >= VAL_LIMIT:
                break

        # 3. Locate Source Image
        image_name = json_file.stem 
        team_name = json_file.parent.parent.name
        src_img_path = json_file.parent.parent / "img" / image_name
        if not src_img_path.exists():
            # Fallback for extension mismatch
            possible = list(src_img_path.parent.glob(f"{Path(image_name).stem}.*"))
            if possible: src_img_path = possible[0]
            else: continue

        # 4. Load & Crop
        cropped_img, crop_rect = crop_black_borders(src_img_path)
        if cropped_img is None: continue
        
        # Skip empty images (e.g. completely black)
        if cropped_img.shape[0] < 10 or cropped_img.shape[1] < 10:
            continue

        crop_x, crop_y, crop_w, crop_h = crop_rect
        
        # 5. Process Labels
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        yolo_lines = []
        for obj in data.get('objects', []):
            if obj['classTitle'] not in CLASS_MAP: continue
            class_id = CLASS_MAP[obj['classTitle']]
            
            pts = obj['points']['exterior']
            x_min, y_min = min(pts[0][0], pts[1][0]), min(pts[0][1], pts[1][1])
            x_max, y_max = max(pts[0][0], pts[1][0]), max(pts[0][1], pts[1][1])
            
            # Adjust to crop
            new_x_min = max(0, x_min - crop_x)
            new_y_min = max(0, y_min - crop_y)
            new_x_max = min(crop_w, x_max - crop_x)
            new_y_max = min(crop_h, y_max - crop_y)
            
            # Discard invalid boxes (width or height is effectively 0)
            if (new_x_max - new_x_min) < 1 or (new_y_max - new_y_min) < 1:
                continue

            # Normalize
            x_center = ((new_x_min + new_x_max) / 2) / crop_w
            y_center = ((new_y_min + new_y_max) / 2) / crop_h
            w_norm = (new_x_max - new_x_min) / crop_w
            h_norm = (new_y_max - new_y_min) / crop_h
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # ONLY SAVE if valid labels exist (prevents empty "background" batches which crash gradient)
        if yolo_lines:
            new_filename = f"{team_name}_{Path(image_name).stem}.jpg"
            cv2.imwrite(str(output_path / 'images' / split_name / new_filename), cropped_img)
            with open(output_path / 'labels' / split_name / (Path(new_filename).stem + ".txt"), 'w') as out_f:
                out_f.write('\n'.join(yolo_lines))
            counts[split_name] += 1

    print(f"Done! Created {counts['train']} train images and {counts['val']} val images.")

if __name__ == "__main__":
    # Current script lives in ./data/fsoco/setup.py
    CURRENT_DIR = Path(__file__).parent.resolve()
    
    # Define exact paths according to the new structure
    INPUT_RAW = CURRENT_DIR / "raw"
    OUTPUT_PREPROCESSED = CURRENT_DIR / "preprocessed"
    
    # 1. Check, Download, and Extract
    download_and_extract(DOWNLOAD_URL, INPUT_RAW)
    
    # 2. Preprocess Data into YOLO format
    convert_fsoco(INPUT_RAW, OUTPUT_PREPROCESSED)