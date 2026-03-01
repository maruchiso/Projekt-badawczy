import os
import json
import time
import csv
import numpy as np

import FK.my_utils as my_utils
from sahi_setup import sahi_fun


# ===============================
# KONFIGURACJA
# ===============================

DATASET_JSON = r"C:\Users\marcin\Desktop\SAHI_FK\FK\SODA-D\annotations\train.json"
IMAGE_DIR = r"C:\Users\marcin\Desktop\SAHI_FK\FK\SODA-D\train"
MODEL_PATH = "soda_n_final.pt"

NUM_IMAGES = 10
IOU_THRESHOLD = 0.5


# ===============================
# FUNKCJE POMOCNICZE
# ===============================

def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / float(area1 + area2 - inter)


def evaluate_predictions(gt_boxes, pred_boxes, iou_thr=0.5):
    matched_gt = set()

    for pred in pred_boxes:
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if compute_iou(pred, gt) >= iou_thr:
                matched_gt.add(i)
                break

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return tp, fp, fn, precision, recall


def compute_tile_metrics(gt_boxes, all_slices, kept_slices, rejected_slices):

    def slice_has_object(slice_bbox):
        x1, y1, x2, y2 = slice_bbox
        for gt in gt_boxes:
            gx1, gy1, gx2, gy2 = gt
            if not (gx2 <= x1 or gx1 >= x2 or gy2 <= y1 or gy1 >= y2):
                return True
        return False

    total_empty = 0
    total_with_obj = 0

    empty_rejected = 0
    with_obj_rejected = 0

    for s in all_slices:
        has_obj = slice_has_object(s)
        if has_obj:
            total_with_obj += 1
            if s in rejected_slices:
                with_obj_rejected += 1
        else:
            total_empty += 1
            if s in rejected_slices:
                empty_rejected += 1

    percent_empty_removed = empty_rejected / total_empty if total_empty else 0
    percent_obj_removed = with_obj_rejected / total_with_obj if total_with_obj else 0
    filter_purity = empty_rejected / len(rejected_slices) if rejected_slices else 0

    return percent_empty_removed, percent_obj_removed, filter_purity


def compute_object_loss(gt_boxes, kept_slices):

    lost_objects = 0

    for gt in gt_boxes:
        gx1, gy1, gx2, gy2 = gt

        object_kept_somewhere = False

        for s in kept_slices:
            x1, y1, x2, y2 = s
            if not (gx2 <= x1 or gx1 >= x2 or gy2 <= y1 or gy1 >= y2):
                object_kept_somewhere = True
                break

        if not object_kept_somewhere:
            lost_objects += 1

    percent_lost = lost_objects / len(gt_boxes) if gt_boxes else 0
    return percent_lost


# ===============================
# WCZYTANIE COCO
# ===============================

with open(DATASET_JSON) as f:
    coco = json.load(f)

images = coco["images"][:NUM_IMAGES]
annotations = coco["annotations"]

print(f"Testing {len(images)} images...\n")

results = []

for img in images:

    file_name = img["file_name"]
    image_id = img["id"]

    print(f"\n=== {file_name} ===")

    gt_boxes = [
        coco_bbox_to_xyxy(a["bbox"])
        for a in annotations if a["image_id"] == image_id
    ]

    # ================= FULL SAHI =================

    my_utils.filtruj_puste_wycinki = False
    my_utils.zdjecia = IMAGE_DIR
    my_utils.imgExtension = ".jpg"
    my_utils.liczba_wycinkow = 0
    my_utils.pozostale_wyc = 0

    start = time.time()
    result_full = sahi_fun(
        nazwa=file_name.replace(".jpg", ""),
        auto_rozmiar=True,
        podzial=6,
        nakladanie=0.4,
        model=MODEL_PATH,
        zapisz=False,
        doslowna_sciezka=False,
        full_prediction=True
    )
    time_full = time.time() - start

    pred_boxes_full = [
        [p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy]
        for p in result_full.object_prediction_list
    ]

    tp_f, fp_f, fn_f, prec_f, rec_f = evaluate_predictions(gt_boxes, pred_boxes_full)
    slices_full = my_utils.liczba_wycinkow

    # ================= FK INTEGRAL =================

    my_utils.filtruj_puste_wycinki = True
    my_utils.fk_mode = "integral"
    my_utils.liczba_wycinkow = 0
    my_utils.pozostale_wyc = 0

    start = time.time()
    result_fk = sahi_fun(
        nazwa=file_name.replace(".jpg", ""),
        auto_rozmiar=True,
        podzial=6,
        nakladanie=0.4,
        model=MODEL_PATH,
        zapisz=False,
        doslowna_sciezka=False,
        full_prediction=True
    )
    time_fk = time.time() - start

    pred_boxes_fk = [
        [p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy]
        for p in result_fk.object_prediction_list
    ]

    tp_fk, fp_fk, fn_fk, prec_fk, rec_fk = evaluate_predictions(gt_boxes, pred_boxes_fk)
    slices_fk = my_utils.pozostale_wyc

    # ================= METRYKI KAFELKÓW =================

    empty_removed, obj_removed, purity = compute_tile_metrics(
        gt_boxes,
        my_utils.all_slice_bboxes,
        my_utils.kept_slice_bboxes,
        my_utils.rejected_slice_bboxes
    )

    object_loss = compute_object_loss(
        gt_boxes,
        my_utils.kept_slice_bboxes
    )

    results.append([
        file_name,
        slices_full,
        slices_fk,
        time_full,
        time_fk,
        tp_f, fp_f, fn_f, prec_f, rec_f,
        tp_fk, fp_fk, fn_fk, prec_fk, rec_fk,
        empty_removed,
        obj_removed,
        purity,
        object_loss
    ])


# ===============================
# ZAPIS CSV
# ===============================

with open("results_full_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image",
        "slices_full", "slices_fk",
        "time_full", "time_fk",
        "tp_full", "fp_full", "fn_full", "precision_full", "recall_full",
        "tp_fk", "fp_fk", "fn_fk", "precision_fk", "recall_fk",
        "empty_removed",
        "obj_removed",
        "filter_purity",
        "object_loss"
    ])
    writer.writerows(results)


# ===============================
# ŚREDNIE
# ===============================

arr = np.array(results, dtype=object)

print("\n==== AVERAGE GLOBAL RESULTS ====")
print(f"Avg slices FULL: {np.mean(arr[:,1].astype(float)):.2f}")
print(f"Avg slices FK:   {np.mean(arr[:,2].astype(float)):.2f}")
print(f"Avg time FULL:   {np.mean(arr[:,3].astype(float)):.2f}s")
print(f"Avg time FK:     {np.mean(arr[:,4].astype(float)):.2f}s")
print(f"Avg recall FULL: {np.mean(arr[:,9].astype(float)):.3f}")
print(f"Avg recall FK:   {np.mean(arr[:,14].astype(float)):.3f}")

print("\n==== TILE METRICS ====")
print(f"% empty tiles removed: {np.mean(arr[:,15].astype(float)):.3f}")
print(f"% tiles with objects wrongly removed: {np.mean(arr[:,16].astype(float)):.3f}")
print(f"Filter purity: {np.mean(arr[:,17].astype(float)):.3f}")
print(f"% objects lost due to filter: {np.mean(arr[:,18].astype(float)):.3f}")
