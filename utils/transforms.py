import random
import numpy as np
import torch
from PIL import Image, ImageEnhance


# ---------------------------------------------------------------------------
# ImageNet statistics (used because backbone is pretrained on ImageNet)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Standalone augmentation functions
# (extracted so they can be called from both DetectionTransform and Dataset)
# ---------------------------------------------------------------------------

def random_horizontal_flip(image, boxes, orig_w, prob=0.5):
    """Randomly flip image horizontally and mirror box coordinates."""
    if random.random() < prob:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if len(boxes) > 0:
            boxes = boxes.copy()
            x_min_old = boxes[:, 0].copy()
            x_max_old = boxes[:, 2].copy()
            boxes[:, 0] = orig_w - x_max_old
            boxes[:, 2] = orig_w - x_min_old
    return image, boxes


def hsv_augment(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """HSV color-space augmentation (YOLOv5/v8 style, using PIL)."""
    img_hsv = image.convert('HSV')
    h, s, v = img_hsv.split()

    h_np = np.array(h, dtype=np.float32)
    s_np = np.array(s, dtype=np.float32)
    v_np = np.array(v, dtype=np.float32)

    # Random multiplicative gains
    r = np.random.uniform(-1, 1, 3) * [h_gain * 256, s_gain, v_gain] + 1

    h_np = np.mod(h_np * r[0], 256).astype(np.uint8)
    s_np = np.clip(s_np * r[1], 0, 255).astype(np.uint8)
    v_np = np.clip(v_np * r[2], 0, 255).astype(np.uint8)

    img_hsv = Image.merge('HSV', [
        Image.fromarray(h_np),
        Image.fromarray(s_np),
        Image.fromarray(v_np),
    ])
    return img_hsv.convert('RGB')


def random_color_jitter(image, prob=0.5):
    """Random brightness, contrast, and saturation jitter."""
    if random.random() < prob:
        factor = random.uniform(0.6, 1.4)
        image = ImageEnhance.Brightness(image).enhance(factor)

        factor = random.uniform(0.6, 1.4)
        image = ImageEnhance.Contrast(image).enhance(factor)

        factor = random.uniform(0.6, 1.4)
        image = ImageEnhance.Color(image).enhance(factor)

    return image


def cutout(image, num_patches=3, max_ratio=0.12):
    """Random erasing / CutOut augmentation.

    Randomly erases rectangular patches, forcing the network to rely on
    global context rather than memorising local texture.
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    for _ in range(num_patches):
        if random.random() > 0.5:
            continue
        ph = int(h * random.uniform(0.02, max_ratio))
        pw = int(w * random.uniform(0.02, max_ratio))
        y = random.randint(0, max(h - ph, 0))
        x = random.randint(0, max(w - pw, 0))
        img_np[y:y + ph, x:x + pw] = 114  # gray fill (matches letterbox)
    return Image.fromarray(img_np)


def letterbox_resize(image, boxes, target_size):
    """Resize image with preserved aspect ratio, centered on a gray canvas."""
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


def to_normalized_tensor(image, mean=None, std=None):
    """Convert PIL Image to normalised float tensor (3, H, W)."""
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    arr = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    # Normalize channel-wise
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - m) / s

    return tensor


# ---------------------------------------------------------------------------
# Transform class (wraps the standalone functions)
# ---------------------------------------------------------------------------

class DetectionTransform:

    # Kept as class-level constants for backward-compat
    MEAN = IMAGENET_MEAN
    STD = IMAGENET_STD

    def __init__(self, img_size=416, train=True):
        self.img_size = img_size
        self.train = train

    def __call__(self, image, boxes, labels):

        orig_w, orig_h = image.size

        # ---- Training augmentations ----
        if self.train:
            image, boxes = random_horizontal_flip(image, boxes, orig_w)
            image = hsv_augment(image)
            image = random_color_jitter(image)

        # ---- Letterbox resize (always) ----
        image, boxes, scale, pad_x, pad_y = letterbox_resize(
            image, boxes, self.img_size
        )

        # ---- CutOut (training only, after resize) ----
        if self.train:
            image = cutout(image)

        # ---- Convert to tensor and normalize ----
        image_tensor = to_normalized_tensor(image)

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
