import json
import os
import time
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

import FK.my_utils as my_utils
from sahi_setup import sahi_fun


# ==============================
# ===== KONFIGURACJA ===========
# ==============================

JSON_PATH = r"C:\Users\marcin\Desktop\SAHI_FK\FK\SODA-D\annotations\train.json"
IMAGE_DIR = r"C:\Users\marcin\Desktop\SAHI_FK\FK\SODA-D\train"

IMAGE_NAME = "00001"   # bez .jpg
MODEL_PATH = "soda_n_final.pt"


# ==============================
# ===== FUNKCJE METRYK =========
# ==============================

def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def compute_iou(box1, box2):
    b1 = box(*box1)
    b2 = box(*box2)
    inter = b1.intersection(b2).area
    union = b1.union(b2).area
    return inter / union if union > 0 else 0


def compute_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    matched_gt = set()
    tp = 0

    for pred in pred_boxes:
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if compute_iou(pred, gt) >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                break

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return tp, fp, fn, precision, recall


# ==============================
# ===== WCZYTANIE COCO =========
# ==============================

with open(JSON_PATH, "r") as f:
    coco_data = json.load(f)

image_id = None
for img in coco_data["images"]:
    if img["file_name"] == IMAGE_NAME + ".jpg":
        image_id = img["id"]
        break

gt_boxes = []
for ann in coco_data["annotations"]:
    if ann["image_id"] == image_id:
        gt_boxes.append(coco_bbox_to_xyxy(ann["bbox"]))

print(f"GT objects: {len(gt_boxes)}")


# ==============================
# ===== FUNKCJA URUCHOMIENIA ===
# ==============================

def run_variant(fk_enabled, fk_mode):

    my_utils.filtruj_puste_wycinki = fk_enabled
    my_utils.fk_mode = fk_mode
    my_utils.canny_th1 = 300
    my_utils.canny_th2 = 400
    my_utils.edgeThreshold = 300
    my_utils.podzial = 6
    my_utils.nakladanie = 0.4
    my_utils.zdjecia = IMAGE_DIR
    my_utils.imgExtension = ".jpg"

    start = time.time()

    result = sahi_fun(
        nazwa=IMAGE_NAME,
        auto_rozmiar=True,
        podzial=6,
        nakladanie=0.4,
        model=MODEL_PATH,
        zapisz=False,
        doslowna_sciezka=False,
        full_prediction=True
    )

    total_time = time.time() - start

    pred_boxes = []
    for obj in result.object_prediction_list:
        pred_boxes.append(obj.bbox.to_xyxy())

    tp, fp, fn, precision, recall = compute_metrics(gt_boxes, pred_boxes)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "time": total_time,
        "pred_count": len(pred_boxes)
    }


# ==============================
# ===== URUCHOMIENIE ===========
# ==============================

print("\n--- FULL SAHI ---")
full_results = run_variant(False, "slicing")
print(full_results)

print("\n--- SAHI + FK INTEGRAL ---")
fk_results = run_variant(True, "integral")
print(fk_results)
