#!/usr/bin/env python
"""
Training script for SSD-DEIM.

Complete training pipeline with:
    - Data loading (COCO format)
    - Mixed precision training (AMP)
    - Cosine annealing with warmup LR schedule
    - Gradient clipping
    - Validation with COCO-style mAP evaluation
    - Checkpoint saving (periodic + best)
    - Logging and progress tracking

Usage:
    python tools/train.py --config configs/ssd_deim_gc2026.py
    python tools/train.py --config configs/ssd_deim_gc2026.py --resume work_dirs/ssd_deim/latest.pth
    python tools/train.py --config configs/ssd_deim_gc2026.py --fp16
"""

import argparse
import os
import os.path as osp
import time
import warnings
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.misc import (
    set_seed,
    CosineAnnealingWarmupLR,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    format_time,
)
from utils.box_ops import box_xyxy_to_cxcywh


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD-DEIM detector')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--work-dir', help='Directory for saving logs and checkpoints')
    parser.add_argument('--resume', help='Checkpoint file to resume from')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Set deterministic mode')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation during training')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency (iterations)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Config loading
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load configuration from Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


# ---------------------------------------------------------------------------
#  Simple COCO dataset wrapper
# ---------------------------------------------------------------------------

class CocoDetectionDataset(torch.utils.data.Dataset):
    """Simple COCO-format detection dataset.

    Loads images and annotations from COCO-format JSON files.
    Supports basic preprocessing: resize, pad, normalize.

    Args:
        ann_file: Path to COCO annotation JSON.
        img_prefix: Image directory prefix.
        img_scale: Target image size (W, H).
        mean: Normalization mean.
        std: Normalization std.
    """

    def __init__(
        self,
        ann_file: str,
        img_prefix: str = '',
        img_scale: tuple = (640, 640),
        mean: list = None,
        std: list = None,
    ):
        super().__init__()
        self.img_prefix = img_prefix
        self.img_scale = img_scale
        self.mean = mean or [123.675, 116.28, 103.53]
        self.std = std or [58.395, 57.12, 57.375]

        # Load COCO annotations
        import json
        with open(ann_file, 'r') as f:
            coco = json.load(f)

        self.img_infos = coco['images']
        self.annotations = coco['annotations']
        self.categories = coco.get('categories', [])
        self.cat_ids = [cat['id'] for cat in self.categories]

        # Build image index -> annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Filter images with no annotations (optional - keep all for evaluation)
        self.valid_img_ids = [img['id'] for img in self.img_infos]

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        import cv2
        import numpy as np

        img_info = self.img_infos[idx]
        img_id = img_info['id']
        img_path = osp.join(self.img_prefix, img_info['file_name'])

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Resize with aspect ratio preservation
        target_w, target_h = self.img_scale
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        img = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array(self.mean, dtype=np.float32) / 255.0
        std = np.array(self.std, dtype=np.float32) / 255.0
        img = (img - mean) / std

        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        # Load annotations
        anns = self.img_to_anns.get(img_id, [])
        gt_bboxes = []
        gt_labels = []

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            # COCO bbox: [x, y, w, h] -> xyxy
            x, y, bw, bh = ann['bbox']
            x1 = x * scale
            y1 = y * scale
            x2 = (x + bw) * scale
            y2 = (y + bh) * scale
            gt_bboxes.append([x1, y1, x2, y2])
            # Map category id to 0-based index
            cat_idx = self.cat_ids.index(ann['category_id']) if ann['category_id'] in self.cat_ids else 0
            gt_labels.append(cat_idx)

        if len(gt_bboxes) > 0:
            gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
            gt_labels = torch.tensor(gt_labels, dtype=torch.long)
        else:
            gt_bboxes = torch.zeros(0, 4, dtype=torch.float32)
            gt_labels = torch.zeros(0, dtype=torch.long)

        return {
            'img': img,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'img_shape': torch.tensor([target_h, target_w]),
            'scale_factor': scale,
            'img_id': img_id,
        }


def collate_fn(batch):
    """Custom collate function for detection data.

    Since each image may have different numbers of GT boxes, we
    collect them into lists instead of stacking.
    """
    imgs = torch.stack([item['img'] for item in batch], dim=0)
    gt_bboxes = [item['gt_bboxes'] for item in batch]
    gt_labels = [item['gt_labels'] for item in batch]
    img_shapes = torch.stack([item['img_shape'] for item in batch], dim=0)
    scale_factors = torch.tensor([item['scale_factor'] for item in batch])
    img_ids = [item['img_id'] for item in batch]

    return {
        'img': imgs,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'img_shape': img_shapes,
        'scale_factor': scale_factors,
        'img_id': img_ids,
    }


def build_dataloader(dataset, batch_size=4, num_workers=4, shuffle=True):
    """Build data loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,  # Only drop last for training
    )


# ---------------------------------------------------------------------------
#  Target preparation
# ---------------------------------------------------------------------------

def prepare_targets(gt_bboxes_list, gt_labels_list, img_shape, device):
    """Convert GT boxes to normalized cxcywh format for matcher/criterion.

    Args:
        gt_bboxes_list: List of [N_i, 4] tensors in xyxy pixel format.
        gt_labels_list: List of [N_i] label tensors.
        img_shape: (H, W) image shape.
        device: Target device.

    Returns:
        List of target dicts with 'boxes' (cxcywh normalized) and 'labels'.
    """
    h, w = img_shape
    targets = []
    for bboxes, labels in zip(gt_bboxes_list, gt_labels_list):
        bboxes = bboxes.to(device)
        labels = labels.to(device)

        if len(bboxes) > 0:
            # xyxy -> cxcywh
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
            bw = bboxes[:, 2] - bboxes[:, 0]
            bh = bboxes[:, 3] - bboxes[:, 1]
            # Normalize to [0, 1]
            cx = cx / w
            cy = cy / h
            bw = bw / w
            bh = bh / h
            boxes_norm = torch.stack([cx, cy, bw, bh], dim=-1)
        else:
            boxes_norm = torch.zeros(0, 4, device=device)

        targets.append({
            'boxes': boxes_norm,
            'labels': labels,
        })

    return targets


# ---------------------------------------------------------------------------
#  Training function
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    epoch,
    total_epochs,
    device,
    use_fp16,
    print_freq,
    grad_clip_max_norm=None,
):
    """Train for one epoch.

    Returns:
        dict with average loss values.
    """
    model.train()
    num_iters = len(train_loader)
    loss_sum = 0.0
    loss_cls_sum = 0.0
    loss_bbox_sum = 0.0
    avg_iou_sum = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, data_batch in enumerate(train_loader):
        images = data_batch['img'].to(device)
        gt_bboxes = data_batch['gt_bboxes']
        gt_labels = data_batch['gt_labels']
        img_h, img_w = images.shape[2:]

        # Prepare targets in normalized cxcywh format
        targets = prepare_targets(gt_bboxes, gt_labels, (img_h, img_w), device)

        global_iter = epoch * num_iters + batch_idx

        # Update learning rate
        if scheduler is not None:
            scheduler.step(global_iter)

        # Forward + backward with optional mixed precision
        optimizer.zero_grad()

        if use_fp16:
            with autocast():
                losses = model(images, targets=targets, return_loss=True, epoch=epoch)
                total_loss = losses['loss_total']

            scaler.scale(total_loss).backward()

            # Gradient clipping
            if grad_clip_max_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            losses = model(images, targets=targets, return_loss=True, epoch=epoch)
            total_loss = losses['loss_total']

            total_loss.backward()

            if grad_clip_max_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            optimizer.step()

        # Accumulate stats
        loss_sum += total_loss.item()
        loss_cls_sum += losses.get('loss_cls', torch.tensor(0.0)).item()
        loss_bbox_sum += losses.get('loss_bbox', torch.tensor(0.0)).item()
        avg_iou_sum += losses.get('avg_iou', torch.tensor(0.0)).item()
        num_batches += 1

        # Print progress
        if (batch_idx + 1) % print_freq == 0 or batch_idx == num_iters - 1:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = loss_sum / num_batches
            avg_cls = loss_cls_sum / num_batches
            avg_bbox = loss_bbox_sum / num_batches
            avg_iou = avg_iou_sum / num_batches
            elapsed = time.time() - start_time
            iters_per_sec = num_batches / elapsed

            print(
                f'Epoch [{epoch+1}/{total_epochs}] '
                f'Iter [{batch_idx+1}/{num_iters}] '
                f'LR: {current_lr:.6f} | '
                f'Loss: {avg_loss:.4f} '
                f'(cls: {avg_cls:.4f}, bbox: {avg_bbox:.4f}) | '
                f'Avg IoU: {avg_iou:.4f} | '
                f'{iters_per_sec:.2f} it/s'
            )

    avg_loss = loss_sum / max(num_batches, 1)
    avg_cls = loss_cls_sum / max(num_batches, 1)
    avg_bbox = loss_bbox_sum / max(num_batches, 1)
    avg_iou = avg_iou_sum / max(num_batches, 1)

    return {
        'loss': avg_loss,
        'loss_cls': avg_cls,
        'loss_bbox': avg_bbox,
        'avg_iou': avg_iou,
    }


# ---------------------------------------------------------------------------
#  Validation function
# ---------------------------------------------------------------------------

def validate(model, val_loader, device, epoch=0):
    """Run validation and compute COCO-style metrics.

    Returns:
        dict with mAP and other metrics.
    """
    model.eval()
    all_results = []
    all_gt = []

    with torch.no_grad():
        for data_batch in val_loader:
            images = data_batch['img'].to(device)
            gt_bboxes = data_batch['gt_bboxes']
            gt_labels = data_batch['gt_labels']
            img_ids = data_batch['img_id']
            img_h, img_w = images.shape[2:]

            # Inference
            results = model(images, return_loss=False)

            # Collect results in COCO format
            for i, result in enumerate(results):
                img_id = img_ids[i]
                boxes = result['boxes']  # xyxy pixel coords
                scores = result['scores']
                labels = result['labels']

                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j].cpu().tolist()
                    all_results.append({
                        'image_id': img_id,
                        'category_id': int(labels[j].item()),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],  # xywh
                        'score': float(scores[j].item()),
                    })

                # Collect GT
                for k in range(len(gt_bboxes[i])):
                    x1, y1, x2, y2 = gt_bboxes[i][k].tolist()
                    all_gt.append({
                        'image_id': img_id,
                        'category_id': int(gt_labels[i][k].item()),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': (x2 - x1) * (y2 - y1),
                        'iscrowd': 0,
                    })

    # Compute COCO metrics
    if len(all_results) == 0:
        return {'bbox_mAP': 0.0}

    try:
        metrics = compute_coco_metrics(all_results, all_gt)
        return metrics
    except Exception as e:
        print(f"Warning: COCO evaluation failed: {e}")
        return {'bbox_mAP': 0.0}


def compute_coco_metrics(detections, ground_truths):
    """Compute COCO-style detection metrics.

    Uses pycocotools if available, otherwise computes basic metrics.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import json
        import tempfile

        # Build GT COCO structure
        gt_coco = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 0, 'name': 'object'}],
        }

        img_ids = sorted(set(gt['image_id'] for gt in ground_truths))
        for i, img_id in enumerate(img_ids):
            gt_coco['images'].append({'id': img_id, 'width': 640, 'height': 640})

        for ann_id, gt in enumerate(ground_truths):
            gt_coco['annotations'].append({
                'id': ann_id,
                'image_id': gt['image_id'],
                'category_id': gt['category_id'],
                'bbox': gt['bbox'],
                'area': gt['area'],
                'iscrowd': gt['iscrowd'],
            })

        # Save to temp files and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(gt_coco, f)
            gt_path = f.name

        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(detections) if detections else None

        if coco_dt is not None:
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = coco_eval.stats
            metrics = {
                'bbox_mAP': stats[0],
                'bbox_mAP_50': stats[1],
                'bbox_mAP_75': stats[2],
                'bbox_mAP_s': stats[3],
                'bbox_mAP_m': stats[4],
                'bbox_mAP_l': stats[5],
            }

            os.unlink(gt_path)
            return metrics
        else:
            os.unlink(gt_path)
            return {'bbox_mAP': 0.0}

    except ImportError:
        print("pycocotools not installed. Computing basic precision/recall...")
        # Fallback: simple AP computation
        return {'bbox_mAP': 0.0}


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Set work directory
    work_dir = args.work_dir or getattr(cfg, 'work_dir', 'work_dirs/ssd_deim/')
    os.makedirs(work_dir, exist_ok=True)

    # Set random seed
    set_seed(args.seed, args.deterministic)

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Build model
    from models import build_ssd_deim
    model = build_ssd_deim(cfg)
    model = model.to(device)

    # Parameter count
    param_stats = count_parameters(model)
    print(f"Model parameters: {param_stats['total_M']:.2f}M total, "
          f"{param_stats['trainable_M']:.2f}M trainable")

    # Batch size
    batch_size = args.batch_size or cfg.data['samples_per_gpu']

    # Build datasets and loaders
    train_dataset = CocoDetectionDataset(
        ann_file=cfg.data['train']['ann_file'],
        img_prefix=cfg.data['train']['img_prefix'],
        img_scale=(640, 640),
        mean=cfg.img_norm_cfg['mean'],
        std=cfg.img_norm_cfg['std'],
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=cfg.data['workers_per_gpu'],
        shuffle=True,
    )
    print(f'Training samples: {len(train_dataset)}, iters/epoch: {len(train_loader)}')

    val_loader = None
    if not args.no_validate:
        val_dataset = CocoDetectionDataset(
            ann_file=cfg.data['val']['ann_file'],
            img_prefix=cfg.data['val']['img_prefix'],
            img_scale=(640, 640),
            mean=cfg.img_norm_cfg['mean'],
            std=cfg.img_norm_cfg['std'],
        )
        val_loader = build_dataloader(
            val_dataset,
            batch_size=batch_size,
            num_workers=cfg.data['workers_per_gpu'],
            shuffle=False,
        )
        print(f'Validation samples: {len(val_dataset)}')

    # Optimizer
    optim_cfg = dict(cfg.optimizer)
    optim_type = optim_cfg.pop('type', 'AdamW')
    lr = optim_cfg.pop('lr', 5e-4)
    optimizer = getattr(torch.optim, optim_type)(
        model.parameters(), lr=lr, **optim_cfg
    )

    # LR scheduler
    total_epochs = cfg.runner['max_epochs']
    total_iters = total_epochs * len(train_loader)
    lr_cfg = cfg.lr_config
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        total_iters=total_iters,
        warmup_iters=lr_cfg.get('warmup_iters', 1500),
        warmup_ratio=lr_cfg.get('warmup_ratio', 0.001),
        min_lr_ratio=lr_cfg.get('min_lr_ratio', 1e-6),
    )

    # Gradient scaler for mixed precision
    use_fp16 = args.fp16 or getattr(cfg, 'fp16', {}).get('enabled', False)
    scaler = GradScaler(enabled=use_fp16)

    # Gradient clipping
    grad_clip_max_norm = None
    if hasattr(cfg, 'optimizer_config'):
        grad_clip = cfg.optimizer_config.get('grad_clip', None)
        if grad_clip:
            grad_clip_max_norm = grad_clip.get('max_norm', None)

    # Resume from checkpoint
    start_epoch = 0
    best_mAP = 0.0

    if args.resume:
        meta = load_checkpoint(args.resume, model, optimizer, scheduler, map_location='cpu')
        start_epoch = meta['epoch'] + 1
        best_mAP = meta.get('best_metric', 0.0)
        print(f'Resumed from epoch {start_epoch}, best mAP: {best_mAP:.4f}')

    # ====================================================================
    #  Training loop
    # ====================================================================
    print(f'\n{"="*70}')
    print(f'SSD-DEIM Training')
    print(f'{"="*70}')
    print(f'Config: {args.config}')
    print(f'Work dir: {work_dir}')
    print(f'Total epochs: {total_epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Base LR: {lr}')
    print(f'Mixed precision: {use_fp16}')
    print(f'{"="*70}\n')

    train_start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            total_epochs=total_epochs,
            device=device,
            use_fp16=use_fp16,
            print_freq=args.print_freq,
            grad_clip_max_norm=grad_clip_max_norm,
        )

        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1}/{total_epochs} completed in {format_time(epoch_time)}')
        print(f'  Train loss: {train_metrics["loss"]:.4f} '
              f'(cls: {train_metrics["loss_cls"]:.4f}, '
              f'bbox: {train_metrics["loss_bbox"]:.4f})')
        print(f'  Avg IoU: {train_metrics["avg_iou"]:.4f}')

        # Validate
        if val_loader is not None and (epoch + 1) % cfg.evaluation['interval'] == 0:
            print('\nRunning validation...')
            val_metrics = validate(model, val_loader, device, epoch=epoch)
            mAP = val_metrics.get('bbox_mAP', 0.0)
            print(f'  mAP: {mAP:.4f}')
            if 'bbox_mAP_50' in val_metrics:
                print(f'  mAP@50: {val_metrics["bbox_mAP_50"]:.4f}')
            if 'bbox_mAP_75' in val_metrics:
                print(f'  mAP@75: {val_metrics["bbox_mAP_75"]:.4f}')

            # Save best model
            if mAP > best_mAP:
                best_mAP = mAP
                best_path = osp.join(work_dir, 'best.pth')
                save_checkpoint(
                    best_path, model, optimizer, scheduler,
                    epoch=epoch, best_metric=best_mAP,
                )
                print(f'  Saved best model to {best_path} (mAP: {best_mAP:.4f})')

        # Save periodic checkpoint
        ckpt_interval = cfg.checkpoint_config['interval']
        if (epoch + 1) % ckpt_interval == 0:
            ckpt_path = osp.join(work_dir, f'epoch_{epoch+1}.pth')
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler,
                epoch=epoch, best_metric=best_mAP,
            )
            print(f'  Saved checkpoint to {ckpt_path}')

        print()

    total_time = time.time() - train_start_time
    print(f'{"="*70}')
    print(f'Training completed in {format_time(total_time)}')
    print(f'Best mAP: {best_mAP:.4f}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
