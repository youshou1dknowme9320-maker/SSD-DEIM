#!/usr/bin/env python
"""
Training script for SSD-DEIM.

Usage:
    python tools/train.py --config configs/ssd_deim_gc2026.py --gpus 0,1,2,3
    python tools/train.py --config configs/ssd_deim_gc2026.py --resume work_dirs/ssd_deim/latest.pth
"""

import argparse
import os
import os.path as osp
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD-DEIM detector')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--work-dir', help='Directory for saving logs and checkpoints')
    parser.add_argument('--resume', help='Checkpoint file to resume from')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated GPU IDs')
    parser.add_argument('--batch-size', type=int, default=None, help='Total batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Set deterministic mode')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation during training')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def load_config(config_path):
    """Load configuration from Python file."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def train_single_gpu(args):
    """Training on single GPU."""
    import torch.nn as nn
    from torch.utils.data import DataLoader
    
    # Load config
    cfg = load_config(args.config)
    
    # Set work directory
    work_dir = args.work_dir or cfg.work_dir
    os.makedirs(work_dir, exist_ok=True)
    
    # Set random seed
    set_random_seed(args.seed, args.deterministic)
    
    # Build model
    from models import build_ssd_deim
    model = build_ssd_deim(cfg)
    
    # Move to GPU
    device = torch.device(f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Build optimizer
    optimizer_cfg = cfg.optimizer.copy()
    optim_type = optimizer_cfg.pop('type', 'AdamW')
    optimizer = getattr(torch.optim, optim_type)(
        model.parameters(), **optimizer_cfg
    )
    
    # Build data loaders
    from datasets import build_dataset, build_dataloader
    
    train_dataset = build_dataset(cfg.data.train)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=True,
    )
    
    if not args.no_validate:
        val_dataset = build_dataset(cfg.data.val)
        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            shuffle=False,
        )
    else:
        val_loader = None
    
    # Learning rate scheduler
    total_epochs = cfg.runner.max_epochs
    lr_config = cfg.lr_config
    
    # Resume if specified
    start_epoch = 0
    best_mAP = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_mAP' in checkpoint:
            best_mAP = checkpoint['best_mAP']
        print(f'Resumed from epoch {start_epoch}, best mAP: {best_mAP:.4f}')
    
    # Training loop
    total_iters = len(train_loader)
    print(f'\n{"="*60}')
    print(f'SSD-DEIM Training')
    print(f'{"="*60}')
    print(f'Config: {args.config}')
    print(f'Work dir: {work_dir}')
    print(f'Device: {device}')
    print(f'Total epochs: {total_epochs}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Iterations per epoch: {total_iters}')
    print(f'Batch size: {cfg.data.samples_per_gpu}')
    print(f'Learning rate: {cfg.optimizer.lr}')
    print(f'{"="*60}\n')
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        
        # Update learning rate (cosine annealing with warmup)
        current_iter = epoch * total_iters
        warmup_iters = lr_config.get('warmup_iters', 1500)
        base_lr = cfg.optimizer['lr']
        min_lr = base_lr * lr_config.get('min_lr_ratio', 1e-6)
        
        if current_iter < warmup_iters:
            # Linear warmup
            warmup_ratio = lr_config.get('warmup_ratio', 0.001)
            k = current_iter / warmup_iters
            lr = base_lr * warmup_ratio + base_lr * (1 - warmup_ratio) * k
        else:
            # Cosine annealing
            progress = (current_iter - warmup_iters) / (total_epochs * total_iters - warmup_iters)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Train one epoch
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, data_batch in enumerate(train_loader):
            # Forward pass
            images = data_batch['img'].to(device)
            gt_bboxes = [b.to(device) for b in data_batch['gt_bboxes']]
            gt_labels = [l.to(device) for l in data_batch['gt_labels']]
            
            losses = model(images, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
            
            # Compute total loss
            loss = sum(losses.values())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_clip = cfg.optimizer_config.get('grad_clip', None)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip['max_norm'])
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % cfg.log_config.interval == 0:
                avg_loss = epoch_loss / num_batches
                print(
                    f'Epoch [{epoch+1}/{total_epochs}] '
                    f'Iter [{batch_idx+1}/{total_iters}] '
                    f'LR: {lr:.6f} | '
                    f'Loss: {avg_loss:.4f} | '
                    f'Cls: {losses.get("loss_cls", torch.tensor(0)).item():.4f} | '
                    f'BBox: {losses.get("loss_bbox", torch.tensor(0)).item():.4f}'
                )
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f'\nEpoch {epoch+1}/{total_epochs} completed. Average loss: {avg_epoch_loss:.4f}\n')
        
        # Validation
        if val_loader is not None and (epoch + 1) % cfg.evaluation.interval == 0:
            print('Running validation...')
            model.eval()
            
            # TODO: Implement full COCO evaluation
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for data_batch in val_loader:
                    images = data_batch['img'].to(device)
                    gt_bboxes = [b.to(device) for b in data_batch['gt_bboxes']]
                    gt_labels = [l.to(device) for l in data_batch['gt_labels']]
                    
                    losses = model(images, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
                    val_loss += sum(losses.values()).item()
                    val_batches += 1
            
            avg_val_loss = val_loss / max(val_batches, 1)
            print(f'Validation loss: {avg_val_loss:.4f}')
            
            # Save best model (placeholder - replace with actual mAP)
            if avg_val_loss < best_mAP or best_mAP == 0:
                best_mAP = avg_val_loss
                best_path = osp.join(work_dir, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_mAP': best_mAP,
                }, best_path)
                print(f'Saved best model to {best_path}')
            
            model.train()
        
        # Save checkpoint
        if (epoch + 1) % cfg.checkpoint_config.interval == 0:
            ckpt_path = osp.join(work_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mAP': best_mAP,
            }, ckpt_path)
            print(f'Saved checkpoint to {ckpt_path}')
    
    print('\nTraining completed!')
    print(f'Best mAP: {best_mAP:.4f}')


def main():
    args = parse_args()
    
    if args.deterministic:
        print('Warning: Deterministic mode may slow down training.')
    
    train_single_gpu(args)


if __name__ == '__main__':
    main()
