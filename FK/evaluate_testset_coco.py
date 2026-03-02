import os
import json
import time
import argparse
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import FK.my_utils as my_utils
from sahi_setup import sahi_fun


# =========================
# Pomocnicze: kafelki
# =========================

def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def intersects(a, b):
    # a,b = [x1,y1,x2,y2]
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def tile_has_object(tile_bbox, gt_boxes):
    return any(intersects(tile_bbox, gt) for gt in gt_boxes)


def compute_tile_metrics_counts(gt_boxes, all_slices, kept_slices, rejected_slices):
    total_empty = 0
    total_with_obj = 0
    empty_rejected = 0
    with_obj_rejected = 0

    rejected_set = set(tuple(x) for x in rejected_slices)

    for s in all_slices:
        s_t = tuple(s)
        has_obj = tile_has_object(s, gt_boxes)
        if has_obj:
            total_with_obj += 1
            if s_t in rejected_set:
                with_obj_rejected += 1
        else:
            total_empty += 1
            if s_t in rejected_set:
                empty_rejected += 1

    # purity needs total rejected:
    total_rejected = len(rejected_slices)

    return {
        "total_empty": total_empty,
        "total_with_obj": total_with_obj,
        "empty_rejected": empty_rejected,
        "with_obj_rejected": with_obj_rejected,
        "total_rejected": total_rejected,
    }


def compute_object_loss_count(gt_boxes, kept_slices):
    # obiekt "utracony przez filtr" jeśli NIE przecina żadnego zachowanego kafelka
    kept = kept_slices
    lost = 0
    for gt in gt_boxes:
        if not any(intersects(gt, k) for k in kept):
            lost += 1
    return lost


# =========================
# Pomocnicze: predykcje -> COCO
# =========================

def build_cat_name_to_id(coco: COCO):
    cats = coco.loadCats(coco.getCatIds())
    return {c["name"]: c["id"] for c in cats}


def sahi_result_to_coco_dets(result, image_id):
    dets = []

    for p in result.object_prediction_list:

        bbox = p.bbox
        x1, y1, x2, y2 = float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)
        w, h = x2 - x1, y2 - y1

        score = float(p.score.value) if hasattr(p, "score") else float(p.score)

        dets.append({
            "image_id": image_id,
            "category_id": 1,   # <-- SINGLE CLASS !!!
            "bbox": [x1, y1, w, h],
            "score": score
        })

    return dets


# =========================
# COCOeval wrapper
# =========================

def run_coco_eval(coco_gt, coco_dets, img_ids, max_dets_list=(1,10,100), iou_thrs=None):
    coco_dt = coco_gt.loadRes(coco_dets) if len(coco_dets) else coco_gt.loadRes([])

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.useCats = 0
    ev.params.imgIds = img_ids
    ev.params.maxDets = list(max_dets_list)

    if iou_thrs is not None:
        ev.params.iouThrs = np.array(iou_thrs, dtype=np.float32)

    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # AP_small/AR_small dla ostatniego maxDet z listy
    area_lbls = ev.params.areaRngLbl
    a_small = area_lbls.index("small")

    precision = ev.eval["precision"]
    recall = ev.eval["recall"]

    ap_small = np.mean(precision[:, :, :, a_small, -1][precision[:, :, :, a_small, -1] > -1])
    ar_small = np.mean(recall[:, :, a_small, -1][recall[:, :, a_small, -1] > -1])

    return {
        "AP_50_95": float(ev.stats[0]),
        "AP_50": float(ev.stats[1]),
        "AR": float(ev.stats[8]),      # AR dla maxDet = max_dets_list[2] (czyli 100, jeśli standard)
        "AP_small": float(ap_small),
        "AR_small": float(ar_small)
    }


# =========================
# Główna pętla
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Ścieżka do test.json (COCO)")
    ap.add_argument("--img_dir", required=True, help="Folder ze zdjęciami dla test.json")
    ap.add_argument("--model", required=True, help="Model (np. soda_n_final.pt)")
    ap.add_argument("--limit", type=int, default=0, help="0 = wszystkie, albo np. 2000")
    ap.add_argument("--out", default="eval_testset_results.json", help="Plik wynikowy JSON")
    ap.add_argument("--edge", type=int, default=300, help="FK edgeThreshold")
    ap.add_argument("--mode", choices=["full", "fk", "both"], default="both")
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()

    coco_gt = COCO(args.ann)
    # =========================
    # CLASS-AGNOSTIC EVAL
    # =========================

    for ann in coco_gt.dataset["annotations"]:
        ann["category_id"] = 1

    coco_gt.dataset["categories"] = [{
        "id": 1,
        "name": "object"
    }]

    coco_gt.createIndex()
    # --- PATCH: some COCO-like datasets miss 'iscrowd' field required by pycocotools
    for ann_id, ann in coco_gt.anns.items():
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
    img_ids = coco_gt.getImgIds()
    if args.limit and args.limit > 0:
        img_ids = img_ids[:args.limit]
    
    baseline_full = None
    baseline_time = None

    if args.mode == "fk":
        if args.baseline is None:
            raise ValueError("FK mode requires --baseline")
        
        with open(args.baseline) as f:
            b = json.load(f)
        
        baseline_full = b["dets_full"]
        baseline_time = b["times_full"]

    # mapowanie kategorii
    cat_ids = coco_gt.getCatIds()

    # PARAMETRY (ustaw raz)
    my_utils.zdjecia = args.img_dir
    my_utils.imgExtension = ".jpg"

    # SAHI slicing params
    PODZIAL = 6
    OVERLAP = 0.4

    # FK params
    my_utils.filtruj_puste_wycinki = False  # zmieniamy w pętli
    my_utils.fk_mode = "integral"
    my_utils.canny_th1 = 300
    my_utils.canny_th2 = 400
    my_utils.edgeThreshold = args.edge

    # Gromadzenie wyników
    dets_full = []
    dets_fk = []

    times_full = []
    times_fk = []

    slices_full = []        # liczba kafelków do detektora (FULL)
    slices_fk = []          # liczba kafelków do detektora (FK)
    slices_total_fk = []    # liczba kafelków wygenerowanych (przed odrzuceniem) dla FK

    # Kafelki – zliczenia globalne (FK)
    tile_counts = {
        "total_empty": 0,
        "total_with_obj": 0,
        "empty_rejected": 0,
        "with_obj_rejected": 0,
        "total_rejected": 0,
        "gt_objects_total": 0,
        "gt_objects_lost_by_filter": 0,
    }

    for idx, image_id in enumerate(img_ids, 1):
        img_info = coco_gt.loadImgs([image_id])[0]
        file_name = img_info["file_name"]
        stem = os.path.splitext(file_name)[0]
        print(f"[{idx}/{len(img_ids)}] Processing: {file_name}")

        # GT bboxes do metryk kafelków / utraty obiektów
        ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = [coco_bbox_to_xyxy(a["bbox"]) for a in anns]
        tile_counts["gt_objects_total"] += len(gt_boxes)

        # ---------- FULL ----------
        if args.mode in ["both", "full"]:

            my_utils.filtruj_puste_wycinki = False

            my_utils.all_slice_bboxes = []
            my_utils.kept_slice_bboxes = []
            my_utils.rejected_slice_bboxes = []

            t0 = time.time()
            res_full = sahi_fun(
                nazwa=stem,
                auto_rozmiar=True,
                podzial=PODZIAL,
                nakladanie=OVERLAP,
                model=args.model,
                zapisz=False,
                doslowna_sciezka=False,
                full_prediction=True
            )
            t_full = time.time() - t0

            times_full.append(t_full)
            dets_full.extend(
                sahi_result_to_coco_dets(res_full, image_id)
            )

        # ---------- FK ----------
        if args.mode in ["both", "fk"]:
            my_utils.filtruj_puste_wycinki = True

            # wyczyść listy bboxów na obraz
            my_utils.all_slice_bboxes = []
            my_utils.kept_slice_bboxes = []
            my_utils.rejected_slice_bboxes = []

            t0 = time.time()
            res_fk = sahi_fun(
                nazwa=stem,
                auto_rozmiar=True,
                podzial=PODZIAL,
                nakladanie=OVERLAP,
                model=args.model,
                zapisz=False,
                doslowna_sciezka=False,
                full_prediction=True
            )
            t_fk = time.time() - t0

            times_fk.append(t_fk)

            dets_fk.extend(
                sahi_result_to_coco_dets(res_fk, image_id)
            )

        # kafelki FK
        # kafelki FK – TYLKO jeśli FK był liczony
        if args.mode in ["both", "fk"]:

            all_s = my_utils.all_slice_bboxes
            kept_s = my_utils.kept_slice_bboxes
            rej_s = my_utils.rejected_slice_bboxes

            slices_total_fk.append(len(all_s))
            slices_fk.append(len(kept_s))

            c = compute_tile_metrics_counts(gt_boxes, all_s, kept_s, rej_s)
            for k in ["total_empty", "total_with_obj", "empty_rejected", "with_obj_rejected", "total_rejected"]:
                tile_counts[k] += c[k]

            tile_counts["gt_objects_lost_by_filter"] += compute_object_loss_count(gt_boxes, kept_s)

        if idx % 50 == 0 or idx == len(img_ids):
            print(f"[{idx}/{len(img_ids)}] processed...")

    if args.mode == "fk":
        print("Using FULL baseline from:", args.baseline)
        dets_full = baseline_full
        times_full = baseline_time
    # =========================
    # COCO METRYKI (FULL vs FK)
    # =========================

    coco_full = None
    coco_fk = None
    coco_full_ar500 = None
    coco_fk_ar500 = None
    coco_full_ar50 = None
    coco_fk_ar50 = None

    if len(dets_full):
        coco_full = run_coco_eval(
            coco_gt,
            dets_full,
            img_ids,
            max_dets_list=(1,10,100)
        )

        coco_full_ar500 = run_coco_eval(
            coco_gt,
            dets_full,
            img_ids,
            max_dets_list=(1,10,500)
        )

        coco_full_ar50 = run_coco_eval(
            coco_gt,
            dets_full,
            img_ids,
            max_dets_list=(1,10,500),
            iou_thrs=[0.5]
        )

    if len(dets_fk):
        coco_fk = run_coco_eval(
            coco_gt,
            dets_fk,
            img_ids,
            max_dets_list=(1,10,100)
        )

        coco_fk_ar500 = run_coco_eval(
            coco_gt,
            dets_fk,
            img_ids,
            max_dets_list=(1,10,500)
        )

        coco_fk_ar50 = run_coco_eval(
            coco_gt,
            dets_fk,
            img_ids,
            max_dets_list=(1,10,500),
            iou_thrs=[0.5]
        )

    # =========================
    # Statystyki czasu / kafelków
    # =========================

    def stats(x):
        if len(x) == 0:
            return {"mean": None, "p50": None, "p95": None}

        x = np.array(x, dtype=np.float64)
        return {
            "mean": float(np.mean(x)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
        }

    time_stats_full = stats(times_full)
    time_stats_fk = stats(times_fk)

    # FULL slice stats - jeśli s_full=0 bo nie zapisujesz list bboxów dla FULL, to tu będzie 0.
    slice_stats_full = stats(slices_full) if np.sum(slices_full) > 0 else {"mean": None, "p50": None, "p95": None}
    slice_stats_fk = stats(slices_fk) if len(slices_fk) else {"mean": None, "p50": None, "p95": None}
    slice_total_stats_fk = stats(slices_total_fk) if len(slices_total_fk) else {"mean": None, "p50": None, "p95": None}

    # % odrzucenia kafelków (FK)
    total_before = float(np.sum(slices_total_fk))
    total_after = float(np.sum(slices_fk))
    pct_rejected = (total_before - total_after) / total_before if total_before > 0 else 0.0

    # metryki kafelków (FK) finalne procenty
    empty_removed = tile_counts["empty_rejected"] / tile_counts["total_empty"] if tile_counts["total_empty"] else 0.0
    obj_removed = tile_counts["with_obj_rejected"] / tile_counts["total_with_obj"] if tile_counts["total_with_obj"] else 0.0
    purity = tile_counts["empty_rejected"] / tile_counts["total_rejected"] if tile_counts["total_rejected"] else 0.0
    obj_lost_pct = tile_counts["gt_objects_lost_by_filter"] / tile_counts["gt_objects_total"] if tile_counts["gt_objects_total"] else 0.0

    out = {
        "params": {
            "podzial": 6,
            "overlap": 0.4,
            "fk_mode": "integral",
            "canny_th1": my_utils.canny_th1,
            "canny_th2": my_utils.canny_th2,
            "edgeThreshold": args.edge
        },
        "num_images": len(img_ids),

        "coco_full": coco_full,
        "coco_fk": coco_fk,
        "AR@500_full": coco_full_ar500["AR"] if coco_full_ar500 else None,
        "AR@500_fk": coco_fk_ar500["AR"] if coco_fk_ar500 else None,
        "AR@0.5@500_full": coco_full_ar50["AR"] if coco_full_ar50 else None,
        "AR@0.5@500_fk": coco_fk_ar50["AR"] if coco_fk_ar50 else None,
        "time_full": time_stats_full,
        "time_fk": time_stats_fk,

        "tiles_full": slice_stats_full,
        "tiles_fk_kept": slice_stats_fk,
        "tiles_fk_total": slice_total_stats_fk,
        "tiles_fk_pct_rejected": pct_rejected,

        "tile_metrics_fk": {
            "empty_tiles_removed": empty_removed,
            "tiles_with_objects_wrongly_removed": obj_removed,
            "filter_purity": purity,
            "objects_lost_due_to_filter": obj_lost_pct
        }
    }

    if args.mode == "full":
        baseline = {
            "dets_full": dets_full,
            "times_full": times_full
        }
        with open(args.out, "w") as f:
            json.dump(baseline, f)
        print("Saved FULL baseline:", args.out)
        return

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n==== FINAL SUMMARY (TEST.JSON) ====")
    print("\nEDGE:", args.edge)
    print(f"Images: {out['num_images']}")
    print("\nCOCO metrics (FULL):", out["coco_full"])
    print("COCO metrics (FK):  ", out["coco_fk"])
    print("\nAR@500 (FULL):", out["AR@500_full"])
    print("AR@500 (FK):  ", out["AR@500_fk"])
    print("\nAR@0.5@500 (FULL):", out["AR@0.5@500_full"])
    print("AR@0.5@500 (FK):  ", out["AR@0.5@500_fk"])
    print("\nTime FULL:", out["time_full"])
    print("Time FK:  ", out["time_fk"])
    print("\nTiles FK kept:", out["tiles_fk_kept"])
    print("Tiles FK total:", out["tiles_fk_total"])
    print(f"Tiles rejected (%): {out['tiles_fk_pct_rejected']*100:.2f}%")
    print("\nTile metrics (FK):", out["tile_metrics_fk"])
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

#python FK/evaluate_testset_coco.py --ann "C:\Users\marcin\Desktop\SODA-D\annotations\test.json" --img_dir "C:\Users\marcin\Desktop\SODA-D\images" --model "soda_n_final.pt" --out "eval_test_results.json" --limit 1 --edge 50

###### baseline
#python FK/evaluate_testset_coco.py --ann "C:\Users\marcin\Desktop\SODA-D\annotations\test.json" --img_dir "C:\Users\marcin\Desktop\SODA-D\images" --model "soda_n_final.pt" --out "baseline_full_1000.json" --limit 1000 --mode full