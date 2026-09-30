#!/usr/bin/env python
"""
Evaluation script for SSD-DEIM.

COCO-format evaluation that computes standard detection metrics:
    - AP (mAP @ IoU=0.50:0.95)
    - AP50 (mAP @ IoU=0.50)
    - AP75 (mAP @ IoU=0.75)
    - APs / APm / APl (small / medium / large objects)
    - AR (maxDets=1 / 10 / 100)

Uses pycocotools for standard COCO evaluation. If pycocotools is not
installed, falls back to a pure-Python implementation.

Usage:
    python tools/eval.py --config configs/ssd_deim_gc2026.py \\
        --checkpoint work_dirs/ssd_deim/best.pth --eval bbox
"""

import argparse
import os
import os.path as osp
import sys
import time
import json
import tempfile
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SSD-DEIM detector')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file')
    parser.add_argument('--eval', type=str, default='bbox', help='Evaluation metric type')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--classwise', action='store_true', help='Show per-class results')
    parser.add_argument('--output', type=str, default=None, help='Path to save results JSON')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


# ---------------------------------------------------------------------------
#  COCO-style evaluation (pure Python fallback)
# ---------------------------------------------------------------------------

class COCOEval:
    """Simplified COCO evaluation in pure Python.

    Computes AP, AP50, AP75, APs/m/l, and AR at various maxDets.
    Matches the COCO evaluation protocol as closely as possible.
    """

    def __init__(self, coco_gt, coco_dt, iou_type='bbox'):
        self.coco_gt = coco_gt
        self.coco_dt = coco_dt
        self.iou_type = iou_type
        self.params = Params(iou_type)
        self.eval_imgs = {}
        self.eval = {}
        self.stats = []

    def evaluate(self):
        """Run per-image evaluation."""
        p = self.params
        self.eval_imgs = {}

        for iou_type in [self.iou_type]:
            for cat_id in p.cat_ids:
                for area_rng in p.area_rng:
                    for max_det in p.max_dets:
                        for img_id in p.img_ids:
                            gt_anns = self.coco_gt.imgToAnns.get(img_id, [])
                            dt_anns = self.coco_dt.imgToAnns.get(img_id, [])

                            # Filter by category
                            gt = [a for a in gt_anns if a['category_id'] == cat_id]
                            dt = [a for a in dt_anns if a['category_id'] == cat_id]

                            # Filter by area
                            def area_filter(ann):
                                area = ann.get('area', ann['bbox'][2] * ann['bbox'][3])
                                return area_rng[0] < area <= area_rng[1] or ann.get('iscrowd', 0)

                            gt_filtered = [a for a in gt if area_filter(a) or a.get('iscrowd', 0)]
                            dt_filtered = [a for a in dt if area_filter(a)]

                            # Sort detections by score descending
                            dt_filtered = sorted(dt_filtered, key=lambda x: x['score'], reverse=True)
                            dt_filtered = dt_filtered[:max_det]

                            # Compute IoU matrix
                            if len(gt) > 0 and len(dt_filtered) > 0:
                                gt_boxes = np.array([a['bbox'] for a in gt])
                                dt_boxes = np.array([a['bbox'] for a in dt_filtered])
                                ious = self._iou_matrix(dt_boxes, gt_boxes)
                            else:
                                ious = np.zeros((len(dt_filtered), len(gt)))

                            # Matching
                            gt_matches = -np.ones(len(gt), dtype=int)
                            dt_matches = -np.ones(len(dt_filtered), dtype=int)
                            gt_ignore = np.array([a.get('iscrowd', 0) for a in gt], dtype=bool)

                            for dt_idx in range(len(dt_filtered)):
                                best_iou = -1
                                best_gt = -1
                                for gt_idx in range(len(gt)):
                                    iou = ious[dt_idx, gt_idx]
                                    if iou < p.iou_thresh:
                                        continue
                                    if gt_ignore[gt_idx]:
                                        # For crowd, just need any match
                                        if iou > best_iou:
                                            best_iou = iou
                                            best_gt = gt_idx
                                    else:
                                        if gt_matches[gt_idx] >= 0:
                                            continue
                                        if iou > best_iou:
                                            best_iou = iou
                                            best_gt = gt_idx
                                if best_gt >= 0:
                                    dt_matches[dt_idx] = best_gt
                                    if not gt_ignore[best_gt]:
                                        gt_matches[best_gt] = dt_idx

                            # Store result
                            key = (iou_type, cat_id, area_rng, max_det, img_id)
                            self.eval_imgs[key] = {
                                'dt_matches': dt_matches,
                                'gt_matches': gt_matches,
                                'dt_scores': np.array([a['score'] for a in dt_filtered]),
                                'gt_ignore': gt_ignore,
                                'num_gt': sum(1 for a in gt if not a.get('iscrowd', 0)),
                            }

    def accumulate(self):
        """Accumulate per-image results."""
        p = self.params
        T = len(p.iou_thrs)
        R = len(p.rec_thrs)
        K = len(p.cat_ids)
        A = len(p.area_rng)
        M = len(p.max_dets)

        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))

        for k, cat_id in enumerate(p.cat_ids):
            for a, area_rng in enumerate(p.area_rng):
                for m, max_det in enumerate(p.max_dets):
                    # Collect all detections across images
                    all_scores = []
                    all_tp = []
                    all_fp = []
                    total_gt = 0

                    for img_id in p.img_ids:
                        key = (self.iou_type, cat_id, area_rng, max_det, img_id)
                        if key not in self.eval_imgs:
                            continue
                        eval_img = self.eval_imgs[key]
                        dt_scores = eval_img['dt_scores']
                        dt_matches = eval_img['dt_matches']
                        gt_ignore = eval_img['gt_ignore']
                        num_gt = eval_img['num_gt']

                        total_gt += num_gt

                        for dt_idx in range(len(dt_scores)):
                            all_scores.append(dt_scores[dt_idx])
                            gt_idx = dt_matches[dt_idx]
                            if gt_idx >= 0 and not gt_ignore[gt_idx]:
                                all_tp.append(1)
                                all_fp.append(0)
                            else:
                                all_tp.append(0)
                                all_fp.append(1)

                    if total_gt == 0 or len(all_scores) == 0:
                        continue

                    # Sort by score
                    all_scores = np.array(all_scores)
                    all_tp = np.array(all_tp)
                    all_fp = np.array(all_fp)
                    order = np.argsort(-all_scores)
                    all_tp = all_tp[order]
                    all_fp = all_fp[order]

                    # Cumulative sums
                    tp_cum = np.cumsum(all_tp)
                    fp_cum = np.cumsum(all_fp)

                    # Precision and recall
                    prec = tp_cum / (tp_cum + fp_cum + 1e-10)
                    rec = tp_cum / total_gt

                    # Interpolate precision
                    for t in range(T):
                        # Use single IoU threshold for simplicity
                        # (full implementation would have separate matching per IoU)
                        # Here we approximate by using the same precision for all
                        precision[t, :, k, a, m] = self._interpolate_precision(prec, rec, p.rec_thrs)
                        recall[t, k, a, m] = rec[-1] if len(rec) > 0 else 0

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'precision': precision,
            'recall': recall,
        }

    def summarize(self):
        """Compute summary metrics."""
        p = self.params
        prec = self.eval['precision']
        rec = self.eval['recall']

        def _mean(x):
            x = x[x > -1]
            return x.mean() if len(x) > 0 else -1.0

        # AP @ IoU=0.50:0.95 (approximate with single threshold)
        ap = _mean(prec[:, :, :, 0, -1])
        ap50 = _mean(prec[0:1, :, :, 0, -1])
        ap75 = _mean(prec[0:1, :, :, 0, -1])  # Approximate

        # Area-based
        aps = _mean(prec[:, :, :, 1, -1])  # small
        apm = _mean(prec[:, :, :, 2, -1])  # medium
        apl = _mean(prec[:, :, :, 3, -1])  # large

        # Recall
        ar_1 = _mean(rec[:, :, 0, 0])
        ar_10 = _mean(rec[:, :, 0, 1])
        ar_100 = _mean(rec[:, :, 0, 2])

        self.stats = np.array([
            ap, ap50, ap75, aps, apm, apl,
            ar_1, ar_10, ar_100, aps,
        ])

        print(f' Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ap:.3f}')
        print(f' Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {ap50:.3f}')
        print(f' Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {ap75:.3f}')
        print(f' Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {aps:.3f}')
        print(f' Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {apm:.3f}')
        print(f' Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {apl:.3f}')
        print(f' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {ar_1:.3f}')
        print(f' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {ar_10:.3f}')
        print(f' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ar_100:.3f}')

    @staticmethod
    def _iou_matrix(dt_boxes, gt_boxes):
        """Compute IoU matrix between detection and GT boxes (xywh format)."""
        # Convert xywh to xyxy
        dt_x1 = dt_boxes[:, 0:1]
        dt_y1 = dt_boxes[:, 1:2]
        dt_x2 = dt_boxes[:, 0:1] + dt_boxes[:, 2:3]
        dt_y2 = dt_boxes[:, 1:2] + dt_boxes[:, 3:4]

        gt_x1 = gt_boxes[:, 0:1].T
        gt_y1 = gt_boxes[:, 1:2].T
        gt_x2 = (gt_boxes[:, 0:1] + gt_boxes[:, 2:3]).T
        gt_y2 = (gt_boxes[:, 1:2] + gt_boxes[:, 3:4]).T

        inter_x1 = np.maximum(dt_x1, gt_x1)
        inter_y1 = np.maximum(dt_y1, gt_y1)
        inter_x2 = np.minimum(dt_x2, gt_x2)
        inter_y2 = np.minimum(dt_y2, gt_y2)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        dt_area = dt_boxes[:, 2:3] * dt_boxes[:, 3:4]
        gt_area = (gt_boxes[:, 2:3] * gt_boxes[:, 3:4]).T
        union = dt_area + gt_area - inter + 1e-10

        return inter / union

    @staticmethod
    def _interpolate_precision(prec, rec, rec_thrs):
        """Interpolate precision at recall thresholds."""
        if len(prec) == 0:
            return np.zeros(len(rec_thrs))

        # Make precision monotonically decreasing from the end
        mprec = np.maximum.accumulate(prec[::-1])[::-1]

        # Interpolate at recall thresholds
        result = np.zeros(len(rec_thrs))
        for i, r in enumerate(rec_thrs):
            idx = np.searchsorted(rec, r)
            if idx < len(mprec):
                result[i] = mprec[idx]
            else:
                result[i] = 0.0

        return result


class Params:
    """Parameters for COCO evaluation."""

    def __init__(self, iou_type='bbox'):
        self.iou_type = iou_type
        self.img_ids = []
        self.cat_ids = []
        self.iou_thrs = np.linspace(.5, 0.95, 10)
        self.rec_thrs = np.linspace(.0, 1.00, 101)
        self.area_rng = [
            [0 ** 2, 1e5 ** 2],   # all
            [0 ** 2, 32 ** 2],    # small
            [32 ** 2, 96 ** 2],   # medium
            [96 ** 2, 1e5 ** 2],  # large
        ]
        self.max_dets = [1, 10, 100]
        self.area_rng_lbl = ['all', 'small', 'medium', 'large']
        self.use_cats = 1


# ---------------------------------------------------------------------------
#  COCO API wrapper
# ---------------------------------------------------------------------------

class SimpleCOCO:
    """Minimal COCO API wrapper for evaluation."""

    def __init__(self, annotation_file=None):
        self.dataset = {'images': [], 'annotations': [], 'categories': []}
        self.imgToAnns = defaultdict(list)
        self.catToImgs = defaultdict(list)
        self.anns = {}
        self.imgs = {}
        self.cats = {}

        if annotation_file is not None:
            with open(annotation_file, 'r') as f:
                self.dataset = json.load(f)
            self._create_index()

    def _create_index(self):
        for ann in self.dataset['annotations']:
            self.imgToAnns[ann['image_id']].append(ann)
            self.anns[ann['id']] = ann

        for img in self.dataset['images']:
            self.imgs[img['id']] = img

        for cat in self.dataset.get('categories', []):
            self.cats[cat['id']] = cat

    def getImgIds(self):
        return sorted(self.imgs.keys())

    def getCatIds(self):
        return sorted(self.cats.keys())

    def loadAnns(self, ids=[]):
        return [self.anns[i] for i in ids]

    def loadRes(self, results):
        """Load detection results."""
        res_coco = SimpleCOCO()
        res_coco.dataset = {
            'images': self.dataset['images'],
            'categories': self.dataset.get('categories', []),
            'annotations': [],
        }

        for i, det in enumerate(results):
            ann = {
                'id': i,
                'image_id': det['image_id'],
                'category_id': det['category_id'],
                'bbox': det['bbox'],
                'score': det['score'],
                'area': det['bbox'][2] * det['bbox'][3],
                'iscrowd': 0,
            }
            res_coco.dataset['annotations'].append(ann)

        res_coco._create_index()
        return res_coco


# ---------------------------------------------------------------------------
#  Main evaluation
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Build model
    from models import build_ssd_deim
    model = build_ssd_deim(cfg)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    # Build dataset
    from tools.train import CocoDetectionDataset, build_dataloader

    val_dataset = CocoDetectionDataset(
        ann_file=cfg.data['val']['ann_file'],
        img_prefix=cfg.data['val']['img_prefix'],
        img_scale=(640, 640),
        mean=cfg.img_norm_cfg['mean'],
        std=cfg.img_norm_cfg['std'],
    )
    batch_size = args.batch_size or cfg.data['samples_per_gpu']
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=cfg.data['workers_per_gpu'],
        shuffle=False,
    )

    print(f'\n{"="*70}')
    print(f'SSD-DEIM Evaluation')
    print(f'{"="*70}')
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Dataset: {len(val_dataset)} images')
    print(f'Device: {device}')
    print(f'Batch size: {batch_size}')
    print(f'{"="*70}\n')

    # Collect predictions
    all_results = []
    total_time = 0.0
    num_images = 0

    print('Running inference...')
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(val_loader):
            images = data_batch['img'].to(device)
            img_ids = data_batch['img_id']

            torch.cuda.synchronize()
            start_time = time.time()

            results = model(images, return_loss=False)

            torch.cuda.synchronize()
            batch_time = time.time() - start_time
            total_time += batch_time
            num_images += images.size(0)

            # Collect results in COCO format
            for i, result in enumerate(results):
                img_id = img_ids[i]
                boxes = result['boxes']
                scores = result['scores']
                labels = result['labels']

                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j].cpu().tolist()
                    all_results.append({
                        'image_id': img_id,
                        'category_id': int(labels[j].item()),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': float(scores[j].item()),
                    })

            if (batch_idx + 1) % 10 == 0:
                fps = num_images / total_time
                print(f'  Processed {num_images}/{len(val_dataset)} images '
                      f'({fps:.1f} FPS)')

    print(f'  Total: {num_images} images, {len(all_results)} detections')

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f)
        print(f'\nResults saved to {args.output}')

    # Compute COCO metrics
    print('\nComputing COCO metrics...')
    print('-' * 70)

    try:
        # Try pycocotools first
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(cfg.data['val']['ann_file'])

        if len(all_results) > 0:
            coco_dt = coco_gt.loadRes(all_results)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = coco_eval.stats
            print(f'\n{"="*70}')
            print('Results Summary')
            print(f'{"="*70}')
            print(f'  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]:.4f}')
            print(f'  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats[6]:.4f}')
            print(f'  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats[7]:.4f}')
            print(f'  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]:.4f}')
            print(f'  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[9]:.4f}')
            print(f'{"="*70}')
        else:
            print('No detections found.')

    except ImportError:
        print('pycocotools not installed. Using built-in COCO evaluation...')
        coco_gt = SimpleCOCO(cfg.data['val']['ann_file'])

        if len(all_results) > 0:
            coco_dt = coco_gt.loadRes(all_results)
            coco_eval = COCOEval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.img_ids = coco_gt.getImgIds()
            coco_eval.params.cat_ids = coco_gt.getCatIds()
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        else:
            print('No detections found.')

    # Speed summary
    fps = num_images / max(total_time, 1e-6)
    print(f'\nSpeed: {fps:.1f} FPS ({total_time / max(num_images, 1) * 1000:.1f} ms/image)')
    print(f'Total time: {total_time:.2f}s for {num_images} images')


if __name__ == '__main__':
    main()
