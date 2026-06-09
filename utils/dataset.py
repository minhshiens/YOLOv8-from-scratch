import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.transforms import DetectionTransform

class DetectionDataset(Dataset):

    CLASSES = ['person', 'car', 'dog', 'cat', 'chair']
    NUM_CLASSES = 5

    def __init__(self, annotation_path, image_dir, img_size=416, train=True):
        
        self.image_dir = image_dir
        self.transform = DetectionTransform(img_size=img_size, train=train)
        self.class_to_idx = {name: i for i, name in enumerate(self.CLASSES)}

        # Load and parse annotations
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.images = data['images']

        # Group annotations by image_id for efficient lookup
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        print(f"[Dataset] Loaded {len(self.images)} images, "
              f"{sum(len(v) for v in self.annotations.values())} annotations "
              f"({'train' if train else 'val'} mode)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']

        # Load image
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')

        # Collect all annotations for this image
        anns = self.annotations.get(img_id, [])

        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [xmin, ymin, xmax, ymax]
            boxes.append(bbox)
            labels.append(self.class_to_idx[ann['class']])

        boxes = np.array(boxes, dtype=np.float32) if boxes else \
            np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply transforms (resize, augment, normalize)
        image_tensor, boxes_tensor, labels_tensor, meta = self.transform(
            image, boxes, labels
        )

        targets = {
            'boxes': boxes_tensor,      # (N, 4) xyxy on resized image
            'labels': labels_tensor,     # (N,) class indices
            'image_id': img_id,          # original filename
            **meta,                      # scale, pad_x, pad_y, orig_w, orig_h
        }

        return image_tensor, targets

def collate_fn(batch):
    
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    images = torch.stack(images, dim=0)
    return images, targets
