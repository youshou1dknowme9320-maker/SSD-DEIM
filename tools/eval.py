#!/usr/bin/env python
"""
Evaluation script for SSD-DEIM.

Usage:
    python tools/eval.py --config configs/ssd_deim_gc2026.py --checkpoint work_dirs/best.pth --eval bbox
"""

import argparse
import os.path as osp
import sys
import time

import torch
import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SSD-DEIM detector')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file')
    parser.add_argument('--eval', type=str, default='bbox', help='Evaluation metric')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--classwise', action='store_true', help='Show per-class results')
    return parser.parse_args()


def load_config(config_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Build model
    from models import build_ssd_deim
    model = build_ssd_deim(cfg)
    
    # Load checkpoint
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Build validation dataset
    from datasets import build_dataset, build_dataloader
    
    val_dataset = build_dataset(cfg.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size or cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False,
    )
    
    print(f'\n{"="*60}')
    print(f'SSD-DEIM Evaluation')
    print(f'{"="*60}')
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Dataset: {len(val_dataset)} images')
    print(f'Device: {device}')
    print(f'{"="*60}\n')
    
    # Collect predictions
    all_results = []
    total_time = 0.0
    num_images = 0
    
    print('Running inference...')
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(val_loader):
            images = data_batch['img'].to(device)
            img_metas = data_batch.get('img_metas', [])
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            results = model(images, img_metas=img_metas, return_loss=False)
            
            torch.cuda.synchronize()
            batch_time = time.time() - start_time
            total_time += batch_time
            num_images += images.size(0)
            
            all_results.extend(results)
            
            if (batch_idx + 1) % 10 == 0:
                fps = num_images / total_time
                print(f'  Processed {num_images}/{len(val_dataset)} images '
                      f'({fps:.1f} FPS)')
    
    # Compute COCO metrics
    print('\nComputing COCO metrics...')
    
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        coco_gt = COCO(cfg.data.val.ann_file)
        coco_dt = coco_gt.loadRes(all_results) if all_results else None
        
        if coco_dt is not None:
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract key metrics
            stats = coco_eval.stats
            print(f'\n{"="*60}')
            print(f'Results Summary')
            print(f'{"="*60}')
            print(f'  Average Precision  (AP) @[ IoU=0.50:0.95 ] = {stats[0]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.50      ] = {stats[1]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.75      ] = {stats[2]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small ] = {stats[3]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium ] = {stats[4]:.4f}')
            print(f'  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large ] = {stats[5]:.4f}')
            print(f'\n  Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets= 1 ] = {stats[6]:.4f}')
            print(f'  Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets=10 ] = {stats[7]:.4f}')
            print(f'  Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets=100] = {stats[8]:.4f}')
            print(f'  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small ] = {stats[9]:.4f}')
            print(f'{"="*60}')
        else:
            print('No detections found.')
            
    except ImportError:
        print('pycocotools not installed. Install with: pip install pycocotools')
        print(f'Inference completed: {num_images} images in {total_time:.2f}s')
        print(f'Average FPS: {num_images / total_time:.1f}')
    
    # Speed summary
    fps = num_images / total_time
    print(f'\nSpeed: {fps:.1f} FPS ({total_time/num_images*1000:.1f} ms/image)')


if __name__ == '__main__':
    main()
