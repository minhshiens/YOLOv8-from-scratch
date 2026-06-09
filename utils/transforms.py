import random
import numpy as np
import torch
from PIL import Image, ImageEnhance

class DetectionTransform:

    # ImageNet statistics (used because backbone is pretrained on ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, img_size=416, train=True):
        
        self.img_size = img_size
        self.train = train

    def __call__(self, image, boxes, labels):
        
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
        
        arr = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

        # Normalize channel-wise
        mean = torch.tensor(self.MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor
