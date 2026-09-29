#!/usr/bin/env python
"""
Inference and visualization script for SSD-DEIM.

Usage:
    python tools/infer.py --config configs/ssd_deim_gc2026.py --checkpoint best.pth --input demo.jpg --output outputs/
    python tools/infer.py --config configs/ssd_deim_gc2026.py --checkpoint best.pth --input test_images/ --output outputs/
"""

import argparse
import os
import os.path as osp
import sys
import time
import glob

import torch
import numpy as np
import cv2

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='SSD-DEIM Inference and Visualization')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', default='outputs/', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--score-thr', type=float, default=0.01, help='Score threshold for detection')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')
    parser.add_argument('--save-txt', action='store_true', help='Save results as txt files')
    return parser.parse_args()


def load_config(config_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def preprocess_image(img_path, img_size=640):
    """Preprocess image for model input."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Cannot read image: {img_path}')
    
    orig_h, orig_w = img.shape[:2]
    
    # Resize with aspect ratio preservation
    scale = min(img_size / orig_w, img_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to square
    pad_w = img_size - new_w
    pad_h = img_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Convert to tensor
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.unsqueeze(0), img, (orig_w, orig_h), (pad_left, pad_top, scale)


def draw_detections(img, results, conf_thres=0.3, class_names=None):
    """Draw detection results on image."""
    if class_names is None:
        class_names = ['object']
    
    for det in results:
        if len(det) == 0:
            continue
        
        for *xyxy, conf, cls_id in det:
            if conf < conf_thres:
                continue
            
            x1, y1, x2, y2 = map(int, xyxy)
            cls_id = int(cls_id)
            conf = float(conf)
            
            # Color based on class
            color = (0, 255, 0)  # Green for default
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"}: {conf:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(img, (x1, y1 - label_h - 4), (x1 + label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
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
    
    # Get input files
    if osp.isdir(args.input):
        img_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_paths.extend(glob.glob(osp.join(args.input, ext)))
        img_paths.sort()
    else:
        img_paths = [args.input]
    
    if not img_paths:
        print(f'No images found in {args.input}')
        return
    
    print(f'\n{"="*60}')
    print(f'SSD-DEIM Inference')
    print(f'{"="*60}')
    print(f'Input: {args.input} ({len(img_paths)} images)')
    print(f'Output: {args.output}')
    print(f'Confidence threshold: {args.conf_thres}')
    print(f'Device: {device}')
    print(f'{"="*60}\n')
    
    # Class names
    class_names = ['citrus']  # GC2026: single class
    
    total_time = 0.0
    
    for idx, img_path in enumerate(img_paths):
        print(f'[{idx+1}/{len(img_paths)}] Processing: {osp.basename(img_path)}')
        
        # Preprocess
        img_tensor, orig_img, (orig_w, orig_h), (pad_left, pad_top, scale) = preprocess_image(img_path)
        img_tensor = img_tensor.to(device)
        
        # Inference
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(img_tensor, return_loss=False, rescale=True)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Convert outputs to numpy
        if isinstance(outputs, list):
            results = []
            for output in outputs:
                if output is not None and len(output) > 0:
                    # Output format: [x1, y1, x2, y2, score, cls_id]
                    dets = output.cpu().numpy() if torch.is_tensor(output) else np.array(output)
                    
                    # Adjust for padding and scale
                    dets[:, 0] = (dets[:, 0] - pad_left) / scale
                    dets[:, 1] = (dets[:, 1] - pad_top) / scale
                    dets[:, 2] = (dets[:, 2] - pad_left) / scale
                    dets[:, 3] = (dets[:, 3] - pad_top) / scale
                    
                    # Clip to image bounds
                    dets[:, 0] = np.clip(dets[:, 0], 0, orig_w)
                    dets[:, 1] = np.clip(dets[:, 1], 0, orig_h)
                    dets[:, 2] = np.clip(dets[:, 2], 0, orig_w)
                    dets[:, 3] = np.clip(dets[:, 3], 0, orig_h)
                    
                    results.append(dets)
                else:
                    results.append(np.empty((0, 6)))
        else:
            results = [np.empty((0, 6))]
        
        num_detections = sum(len(r[r[:, 4] > args.conf_thres]) for r in results)
        print(f'  Detected {num_detections} objects above threshold ({elapsed*1000:.1f} ms)')
        
        # Draw and save
        result_img = draw_detections(orig_img.copy(), results, args.conf_thres, class_names)
        
        out_path = osp.join(args.output, osp.basename(img_path))
        cv2.imwrite(out_path, result_img)
        print(f'  Saved to: {out_path}')
        
        # Save txt results if requested
        if args.save_txt:
            txt_path = osp.join(args.output, osp.splitext(osp.basename(img_path))[0] + '.txt')
            with open(txt_path, 'w') as f:
                for det in results[0]:
                    x1, y1, x2, y2, conf, cls_id = det
                    if conf >= args.conf_thres:
                        f.write(f'{int(cls_id)} {conf:.6f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n')
    
    # Summary
    avg_fps = len(img_paths) / total_time if total_time > 0 else 0
    print(f'\n{"="*60}')
    print(f'Inference completed')
    print(f'  Total images: {len(img_paths)}')
    print(f'  Total time: {total_time:.2f} s')
    print(f'  Average FPS: {avg_fps:.1f}')
    print(f'  Average latency: {total_time/len(img_paths)*1000:.1f} ms')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
