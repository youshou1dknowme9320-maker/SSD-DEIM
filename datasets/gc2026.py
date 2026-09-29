"""
GC2026 Dataset - Green Citrus Orchard Detection Benchmark.

Dataset characteristics:
    - 3,470 images (after augmentation from 410 original images)
    - Dense green citrus fruits with severe camouflage
    - High spatial clustering (>40 targets per image)
    - Small target scales (<32x32 pixels for distant instances)
    - COCO format annotations
"""

import os
import os.path as osp
import json
import numpy as np
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class GC2026Dataset(Dataset):
    """
    GC2026 Green Citrus Orchard Detection Dataset.
    
    Args:
        ann_file: Path to COCO format annotation file
        img_prefix: Path to image directory
        pipeline: List of data augmentation transforms
        test_mode: Whether in test mode (no annotations loaded)
    """
    
    CLASSES = ('citrus',)  # Single class: citrus fruit
    
    def __init__(
        self,
        ann_file: str,
        img_prefix: str = '',
        pipeline: Optional[List] = None,
        test_mode: bool = False,
    ):
        super().__init__()
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline or []
        self.test_mode = test_mode
        
        # Load COCO annotations
        self.coco = self._load_coco_annotations(ann_file)
        self.img_ids = list(self.coco.getImgIds())
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
        # Pre-load image info
        self.img_infos = []
        for img_id in self.img_ids:
            info = self.coco.loadImgs([img_id])[0]
            self.img_infos.append(info)
        
        # Load annotations for training
        if not test_mode:
            self.annotations = []
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                anns = self.coco.loadAnns(ann_ids)
                self.annotations.append(anns)
        
        print(f'Loaded GC2026 dataset: {len(self.img_infos)} images from {ann_file}')
    
    def _load_coco_annotations(self, ann_file: str):
        """Load COCO format annotations."""
        try:
            from pycocotools.coco import COCO
            return COCO(ann_file)
        except ImportError:
            # Fallback: minimal COCO-like interface
            print('Warning: pycocotools not installed, using fallback JSON loader')
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            class SimpleCOCO:
                def __init__(self, data):
                    self.dataset = data
                    self.imgs = {img['id']: img for img in data['images']}
                    self.anns = {ann['id']: ann for ann in data['annotations']}
                    self.cats = {cat['id']: cat for cat in data['categories']}
                    self.imgToAnns = {}
                    for ann in data['annotations']:
                        img_id = ann['image_id']
                        if img_id not in self.imgToAnns:
                            self.imgToAnns[img_id] = []
                        self.imgToAnns[img_id].append(ann)
                
                def getImgIds(self):
                    return list(self.imgs.keys())
                
                def getCatIds(self):
                    return list(self.cats.keys())
                
                def loadImgs(self, ids):
                    return [self.imgs[i] for i in ids]
                
                def loadAnns(self, ids):
                    return [self.anns[i] for i in ids]
                
                def getAnnIds(self, imgIds=None):
                    if imgIds is not None:
                        anns = []
                        for img_id in imgIds:
                            anns.extend([a['id'] for a in self.imgToAnns.get(img_id, [])])
                        return anns
                    return list(self.anns.keys())
            
            return SimpleCOCO(data)
    
    def __len__(self) -> int:
        return len(self.img_infos)
    
    def __getitem__(self, idx: int) -> Dict:
        img_info = self.img_infos[idx]
        img_path = osp.join(self.img_prefix, img_info['file_name'])
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'Cannot read image: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        results = {
            'img': img,
            'img_info': img_info,
            'img_path': img_path,
        }
        
        if not self.test_mode:
            anns = self.annotations[idx]
            gt_bboxes = []
            gt_labels = []
            
            for ann in anns:
                if ann.get('ignore', False) or ann.get('iscrowd', False):
                    continue
                
                bbox = ann['bbox']  # COCO format: [x, y, w, h]
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                # Skip invalid boxes
                if w <= 1 or h <= 1:
                    continue
                
                gt_bboxes.append([x1, y1, x2, y2])
                gt_labels.append(self.cat2label.get(ann['category_id'], 0))
            
            if len(gt_bboxes) == 0:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.zeros((0,), dtype=np.int64)
            else:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels
        
        # Apply pipeline (simplified)
        results = self._apply_pipeline(results)
        
        return results
    
    def _apply_pipeline(self, results: Dict) -> Dict:
        """Apply data augmentation pipeline (simplified implementation)."""
        # In practice, use a proper augmentation library like albumentations
        # This is a simplified version for code structure
        
        img = results['img']
        h, w = img.shape[:2]
        
        # Resize to 640x640
        target_size = 640
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Scale bounding boxes
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            gt_bboxes = results['gt_bboxes'].copy()
            gt_bboxes[:, 0] = gt_bboxes[:, 0] * scale + pad_left
            gt_bboxes[:, 1] = gt_bboxes[:, 1] * scale + pad_top
            gt_bboxes[:, 2] = gt_bboxes[:, 2] * scale + pad_left
            gt_bboxes[:, 3] = gt_bboxes[:, 3] * scale + pad_top
            results['gt_bboxes'] = gt_bboxes
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_padded).float().permute(2, 0, 1) / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        results['img'] = img_tensor
        
        # Convert bboxes to tensors
        if 'gt_bboxes' in results:
            results['gt_bboxes'] = torch.from_numpy(results['gt_bboxes']).float()
            results['gt_labels'] = torch.from_numpy(results['gt_labels']).long()
        
        return results
    
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids of image by index."""
        if self.test_mode:
            return []
        anns = self.annotations[idx]
        return [self.cat2label[ann['category_id']] for ann in anns
                if not ann.get('ignore', False)]


def build_dataset(cfg) -> Dataset:
    """Build dataset from configuration."""
    dataset_type = cfg.get('type', 'GC2026Dataset')
    
    if dataset_type == 'CocoDataset' or dataset_type == 'GC2026Dataset':
        return GC2026Dataset(
            ann_file=cfg.ann_file,
            img_prefix=cfg.get('img_prefix', ''),
            pipeline=cfg.get('pipeline', None),
            test_mode=cfg.get('test_mode', False),
        )
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')


def build_dataloader(
    dataset: Dataset,
    samples_per_gpu: int = 4,
    workers_per_gpu: int = 4,
    shuffle: bool = False,
    num_gpus: int = 1,
) -> DataLoader:
    """Build dataloader from dataset."""
    
    def collate_fn(batch):
        """Custom collate function for variable-length annotations."""
        imgs = torch.stack([item['img'] for item in batch])
        gt_bboxes = [item['gt_bboxes'] for item in batch] if 'gt_bboxes' in batch[0] else None
        gt_labels = [item['gt_labels'] for item in batch] if 'gt_labels' in batch[0] else None
        img_infos = [item.get('img_info', {}) for item in batch]
        
        result = {
            'img': imgs,
            'img_metas': img_infos,
        }
        if gt_bboxes is not None:
            result['gt_bboxes'] = gt_bboxes
            result['gt_labels'] = gt_labels
        
        return result
    
    dataloader = DataLoader(
        dataset,
        batch_size=samples_per_gpu * num_gpus,
        shuffle=shuffle,
        num_workers=workers_per_gpu * num_gpus,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,  # Drop last only for training
    )
    
    return dataloader
