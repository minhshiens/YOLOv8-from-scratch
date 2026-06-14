import json
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.transforms import (
    DetectionTransform,
    random_horizontal_flip,
    hsv_augment,
    cutout,
    to_normalized_tensor,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class DetectionDataset(Dataset):

    CLASSES = ['person', 'car', 'dog', 'cat', 'chair']
    NUM_CLASSES = 5

    def __init__(self, annotation_path, image_dir, img_size=416, train=True,
                 mosaic_prob=0.5, mixup_prob=0.3):

        self.image_dir = image_dir
        self.img_size = img_size
        self.train = train
        self.mosaic_prob = mosaic_prob if train else 0.0
        self.mixup_prob = mixup_prob if train else 0.0
        self.use_mosaic = True  # toggled off for last N epochs
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mosaic(self, enabled: bool):
        """Toggle Mosaic augmentation (disable for last N epochs)."""
        self.use_mosaic = enabled

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # ----- Mosaic + MixUp path (training only) -----
        if self.train and self.use_mosaic and random.random() < self.mosaic_prob:
            image, boxes, labels = self._load_mosaic(idx)
            img_id = self.images[idx]['id']

            # Optional MixUp on top of the Mosaic
            if random.random() < self.mixup_prob:
                idx2 = random.randint(0, len(self) - 1)
                image2, boxes2, labels2 = self._load_mosaic(idx2)
                image, boxes, labels = self._mixup(
                    image, boxes, labels, image2, boxes2, labels2
                )

            # Post-mosaic augmentations
            w, h = image.size
            image, boxes = random_horizontal_flip(image, boxes, w)
            image = hsv_augment(image)
            image = cutout(image)

            # Normalise to tensor
            image_tensor = to_normalized_tensor(image)

            if len(boxes) > 0:
                boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.long)

            meta = {
                'scale': 1.0,
                'pad_x': 0,
                'pad_y': 0,
                'orig_w': self.img_size,
                'orig_h': self.img_size,
            }

            targets = {
                'boxes': boxes_tensor,
                'labels': labels_tensor,
                'image_id': img_id,
                **meta,
            }

        # ----- Standard single-image path -----
        else:
            image, boxes, labels, img_id = self._load_raw(idx)

            image_tensor, boxes_tensor, labels_tensor, meta = self.transform(
                image, boxes, labels
            )

            targets = {
                'boxes': boxes_tensor,
                'labels': labels_tensor,
                'image_id': img_id,
                **meta,
            }

        return image_tensor, targets

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_raw(self, idx):
        """Load a single image and its annotations without any transforms."""
        img_info = self.images[idx]
        img_id = img_info['id']

        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')

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

        return image, boxes, labels, img_id

    # ------------------------------------------------------------------
    # Mosaic augmentation
    # ------------------------------------------------------------------

    def _load_mosaic(self, idx):
        """Create a mosaic of 4 randomly-chosen images.

        Places 4 images in a 2×2 grid around a random centre point on an
        img_size × img_size canvas. Each image is first rescaled to fit
        within img_size, then the appropriate quadrant region is pasted.
        """
        s = self.img_size

        # Random centre for the mosaic split
        yc = int(random.uniform(s * 0.25, s * 0.75))
        xc = int(random.uniform(s * 0.25, s * 0.75))

        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]

        mosaic_np = np.full((s, s, 3), 114, dtype=np.uint8)
        all_boxes = []
        all_labels = []

        for i, index in enumerate(indices):
            img, boxes, labels, _ = self._load_raw(index)
            orig_w, orig_h = img.size

            # Resize to fit within img_size (maintain aspect ratio)
            r = min(s / orig_w, s / orig_h)
            new_w = int(orig_w * r)
            new_h = int(orig_h * r)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img_np = np.array(img)

            # Scale boxes to resized coordinates
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes *= r

            # Compute source and destination regions for each quadrant
            if i == 0:  # top-left
                x1a = max(xc - new_w, 0)
                y1a = max(yc - new_h, 0)
                x2a = xc
                y2a = yc
                x1b = new_w - (x2a - x1a)
                y1b = new_h - (y2a - y1a)
                x2b = new_w
                y2b = new_h
            elif i == 1:  # top-right
                x1a = xc
                y1a = max(yc - new_h, 0)
                x2a = min(xc + new_w, s)
                y2a = yc
                x1b = 0
                y1b = new_h - (y2a - y1a)
                x2b = min(new_w, x2a - x1a)
                y2b = new_h
            elif i == 2:  # bottom-left
                x1a = max(xc - new_w, 0)
                y1a = yc
                x2a = xc
                y2a = min(yc + new_h, s)
                x1b = new_w - (x2a - x1a)
                y1b = 0
                x2b = new_w
                y2b = min(new_h, y2a - y1a)
            else:  # bottom-right
                x1a = xc
                y1a = yc
                x2a = min(xc + new_w, s)
                y2a = min(yc + new_h, s)
                x1b = 0
                y1b = 0
                x2b = min(new_w, x2a - x1a)
                y2b = min(new_h, y2a - y1a)

            # Paste region
            dst_h = y2a - y1a
            dst_w = x2a - x1a
            src_h = y2b - y1b
            src_w = x2b - x1b
            # Guard against size mismatch from rounding
            h_min = min(dst_h, src_h)
            w_min = min(dst_w, src_w)
            if h_min > 0 and w_min > 0:
                mosaic_np[y1a:y1a + h_min, x1a:x1a + w_min] = \
                    img_np[y1b:y1b + h_min, x1b:x1b + w_min]

            # Adjust box coordinates (offset = destination - source)
            if len(boxes) > 0:
                padw = x1a - x1b
                padh = y1a - y1b
                adj = boxes.copy()
                adj[:, [0, 2]] += padw
                adj[:, [1, 3]] += padh
                all_boxes.append(adj)
                all_labels.append(labels)

        # Merge and clip to canvas bounds
        if all_boxes:
            all_boxes = np.concatenate(all_boxes)
            all_labels = np.concatenate(all_labels)

            np.clip(all_boxes[:, 0], 0, s, out=all_boxes[:, 0])
            np.clip(all_boxes[:, 1], 0, s, out=all_boxes[:, 1])
            np.clip(all_boxes[:, 2], 0, s, out=all_boxes[:, 2])
            np.clip(all_boxes[:, 3], 0, s, out=all_boxes[:, 3])

            # Remove degenerate boxes (too small after clipping)
            w = all_boxes[:, 2] - all_boxes[:, 0]
            h = all_boxes[:, 3] - all_boxes[:, 1]
            valid = (w > 4) & (h > 4)
            all_boxes = all_boxes[valid]
            all_labels = all_labels[valid]
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_labels = np.array([], dtype=np.int64)

        return Image.fromarray(mosaic_np), all_boxes, all_labels

    # ------------------------------------------------------------------
    # MixUp augmentation
    # ------------------------------------------------------------------

    def _mixup(self, img1, boxes1, labels1, img2, boxes2, labels2):
        """Blend two images and merge their annotations (MixUp)."""
        alpha = random.uniform(0.5, 0.5)  # fixed 0.5 blend for simplicity

        w1, h1 = img1.size
        w2, h2 = img2.size

        # Ensure same spatial size
        if (w1, h1) != (w2, h2):
            img2 = img2.resize((w1, h1), Image.BILINEAR)
            if len(boxes2) > 0:
                sx = w1 / w2
                sy = h1 / h2
                boxes2 = boxes2.copy()
                boxes2[:, [0, 2]] *= sx
                boxes2[:, [1, 3]] *= sy

        # Alpha blend
        np1 = np.array(img1, dtype=np.float32)
        np2 = np.array(img2, dtype=np.float32)
        mixed = (alpha * np1 + (1 - alpha) * np2).astype(np.uint8)

        # Merge annotations
        boxes_list, labels_list = [], []
        if len(boxes1) > 0:
            boxes_list.append(boxes1)
            labels_list.append(labels1)
        if len(boxes2) > 0:
            boxes_list.append(boxes2)
            labels_list.append(labels2)

        if boxes_list:
            boxes = np.concatenate(boxes_list)
            labels = np.concatenate(labels_list)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)

        return Image.fromarray(mixed), boxes, labels


def collate_fn(batch):

    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    images = torch.stack(images, dim=0)
    return images, targets
