"""
Data augmentation and preprocessing transforms for object detection.

All transforms handle both images AND bounding boxes consistently.
Key design decisions:
- Letterbox resize (preserves aspect ratio with gray padding)
- ImageNet normalization (for pretrained backbone compatibility)
- Augmentations only applied during training
"""

import random
import numpy as np
import torch
from PIL import Image, ImageEnhance


class DetectionTransform:
    """
    Transform pipeline for object detection.

    Training augmentations:
    - Random horizontal flip (50% probability)
    - Random color jitter (brightness, contrast, saturation)

    Always applied:
    - Letterbox resize to target size (aspect-ratio preserving)
    - Normalize with ImageNet mean/std
    """

    # ImageNet statistics (used because backbone is pretrained on ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, img_size=416, train=True):
        """
        Args:
            img_size: target image size (square)
            train: if True, apply data augmentation
        """
        self.img_size = img_size
        self.train = train

    def __call__(self, image, boxes, labels):
        """
        Apply transforms to image and bounding boxes.

        Args:
            image: PIL Image (RGB)
            boxes: numpy array (N, 4) in [xmin, ymin, xmax, ymax] on original image
            labels: numpy array (N,) of class indices

        Returns:
            image_tensor: (3, img_size, img_size) normalized float tensor
            boxes_tensor: (N, 4) tensor in xyxy format on the resized image
            labels_tensor: (N,) long tensor of class indices
            meta: dict with {'scale', 'pad_x', 'pad_y', 'orig_w', 'orig_h'}
        """
        orig_w, orig_h = image.size

        # ---- Training augmentations ----
        if self.train:
            image, boxes = self._random_horizontal_flip(image, boxes, orig_w)
            image = self._random_color_jitter(image)

        # ---- Letterbox resize (always) ----
        image, boxes, scale, pad_x, pad_y = self._letterbox_resize(
            image, boxes, self.img_size
        )

        # ---- Convert to tensor and normalize ----
        image_tensor = self._to_normalized_tensor(image)

        # ---- Convert targets to tensors ----
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)

        meta = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'orig_w': orig_w,
            'orig_h': orig_h,
        }

        return image_tensor, boxes_tensor, labels_tensor, meta

    # ------------------------------------------------------------------
    # Augmentation methods
    # ------------------------------------------------------------------

    def _random_horizontal_flip(self, image, boxes, orig_w, prob=0.5):
        """
        Randomly flip image and boxes horizontally.

        When flipping, x-coordinates transform as:
            new_xmin = orig_w - old_xmax
            new_xmax = orig_w - old_xmin
        """
        if random.random() < prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes) > 0:
                boxes = boxes.copy()
                x_min_old = boxes[:, 0].copy()
                x_max_old = boxes[:, 2].copy()
                boxes[:, 0] = orig_w - x_max_old
                boxes[:, 2] = orig_w - x_min_old
        return image, boxes

    def _random_color_jitter(self, image, prob=0.5):
        """Apply random brightness, contrast, and saturation changes."""
        if random.random() < prob:
            # Brightness: scale pixel values
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Brightness(image).enhance(factor)

            # Contrast: adjust difference from mean
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Contrast(image).enhance(factor)

            # Saturation: adjust color intensity
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Color(image).enhance(factor)

        return image

    # ------------------------------------------------------------------
    # Resize and normalization
    # ------------------------------------------------------------------

    def _letterbox_resize(self, image, boxes, target_size):
        """
        Resize image with padding to preserve aspect ratio (letterbox).

        Steps:
        1. Compute scale factor to fit image in target_size x target_size
        2. Resize image proportionally
        3. Pad shorter dimension with gray (114, 114, 114)
        4. Adjust bounding box coordinates accordingly

        Args:
            image: PIL Image
            boxes: (N, 4) numpy array in xyxy format
            target_size: int, target square size

        Returns:
            padded_image: PIL Image of size (target_size, target_size)
            adjusted_boxes: (N, 4) numpy array in xyxy on the padded image
            scale: float scale factor applied
            pad_x: int horizontal padding offset
            pad_y: int vertical padding offset
        """
        orig_w, orig_h = image.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Compute padding (center the image)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2

        # Create padded image with gray background
        padded = Image.new('RGB', (target_size, target_size), (114, 114, 114))
        padded.paste(image, (pad_x, pad_y))

        # Adjust box coordinates: scale then shift
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y

        return padded, boxes, scale, pad_x, pad_y

    def _to_normalized_tensor(self, image):
        """
        Convert PIL Image to normalized tensor.

        Steps:
        1. Convert to float32 numpy array in [0, 1]
        2. Transpose from (H, W, C) to (C, H, W)
        3. Normalize with ImageNet mean and std
        """
        arr = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

        # Normalize channel-wise
        mean = torch.tensor(self.MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor
