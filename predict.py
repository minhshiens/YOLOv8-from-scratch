#!/usr/bin/env python3
"""
Prediction (inference) script for the FCOS object detector.

Usage:
    python predict.py \
        --image_dir /path/to/images \
        --output predictions.json

Loads the trained model, runs inference on all images in the directory,
applies confidence thresholding and per-class NMS, then saves results
in the required JSON format.

Output format:
    [
        {
            "image_id": "img_xxx.jpg",
            "boxes": [
                {
                    "class": "person",
                    "confidence": 0.91,
                    "bbox": [48, 72, 210, 356]
                }
            ]
        }
    ]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from model import FCOSDetector
from utils.box_utils import generate_grid_points, ltrb_to_xyxy
from utils.nms import per_class_nms
from utils.transforms import DetectionTransform


CLASSES = ['person', 'car', 'dog', 'cat', 'chair']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run FCOS object detection inference'
    )
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Output JSON file path')
    parser.add_argument('--checkpoint', type=str, default='./models/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, default=416,
                        help='Input image size (must match training)')
    parser.add_argument('--conf_thresh', type=float, default=0.3,
                        help='Confidence threshold for filtering')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detected if omitted)')
    return parser.parse_args()


@torch.no_grad()
def predict_single_image(model, image_path, transform, device,
                         conf_thresh, nms_thresh):
    """
    Run detection on a single image.

    Pipeline:
    1. Load image and apply letterbox resize + normalization
    2. Forward pass through the model (3 FPN levels)
    3. Decode predictions: sigmoid(cls) × sigmoid(centerness) → final score
    4. Convert ltrb distances to xyxy boxes
    5. Apply confidence thresholding
    6. Apply per-class NMS
    7. Convert boxes back to original image coordinates

    Args:
        model: FCOSDetector in eval mode
        image_path: path to the image file
        transform: DetectionTransform instance (train=False)
        device: torch device
        conf_thresh: minimum confidence to keep
        nms_thresh: NMS IoU threshold

    Returns:
        list of detection dicts with 'class', 'confidence', 'bbox'
    """
    strides = [8, 16, 32]

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size

    # Apply transforms (no augmentation, just resize + normalize)
    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    empty_labels = np.array([], dtype=np.int64)
    img_tensor, _, _, meta = transform(image, empty_boxes, empty_labels)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

    # Forward pass
    predictions = model(img_tensor)

    # Collect predictions from all FPN levels
    all_boxes = []
    all_scores = []
    all_labels = []

    for level_idx, (cls_logits, reg_pred, ctr_logits) in enumerate(predictions):
        _, C, H, W = cls_logits.shape
        stride = strides[level_idx]

        # Generate grid points for this FPN level
        points = generate_grid_points(H, W, stride, device=device)  # (HW, 2)

        # Decode class scores: sigmoid to get probabilities
        cls_scores = cls_logits[0].sigmoid().permute(1, 2, 0).reshape(-1, C)

        # Decode centerness: sigmoid to get [0, 1] score
        ctr_scores = ctr_logits[0].sigmoid().permute(1, 2, 0).reshape(-1, 1)

        # Regression predictions (already positive via ReLU in head)
        reg = reg_pred[0].permute(1, 2, 0).reshape(-1, 4)  # (HW, 4) ltrb

        # Final score = class_score × centerness
        # This downweights predictions far from object centers
        scores = cls_scores * ctr_scores  # (HW, C)

        # Get the best class for each location
        max_scores, max_labels = scores.max(dim=1)

        # Filter by confidence
        keep = max_scores > conf_thresh
        if not keep.any():
            continue

        # Decode boxes: convert ltrb distances to xyxy coordinates
        boxes = ltrb_to_xyxy(reg[keep], points[keep])

        all_boxes.append(boxes)
        all_scores.append(max_scores[keep])
        all_labels.append(max_labels[keep])

    # If no predictions pass the threshold, return empty
    if not all_boxes:
        return []

    # Merge predictions from all FPN levels
    all_boxes = torch.cat(all_boxes)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    # Per-class NMS
    keep_boxes, keep_scores, keep_labels = per_class_nms(
        all_boxes, all_scores, all_labels,
        iou_threshold=nms_thresh,
        score_threshold=conf_thresh,
    )

    if keep_boxes.numel() == 0:
        return []

    # ---- Convert to original image coordinates ----
    keep_boxes = keep_boxes.cpu().float()
    scale = meta['scale']
    pad_x = meta['pad_x']
    pad_y = meta['pad_y']

    # Undo letterbox: subtract padding, then divide by scale
    keep_boxes[:, [0, 2]] = (keep_boxes[:, [0, 2]] - pad_x) / scale
    keep_boxes[:, [1, 3]] = (keep_boxes[:, [1, 3]] - pad_y) / scale

    # Clip to original image bounds
    keep_boxes[:, 0] = keep_boxes[:, 0].clamp(min=0, max=orig_w)
    keep_boxes[:, 1] = keep_boxes[:, 1].clamp(min=0, max=orig_h)
    keep_boxes[:, 2] = keep_boxes[:, 2].clamp(min=0, max=orig_w)
    keep_boxes[:, 3] = keep_boxes[:, 3].clamp(min=0, max=orig_h)

    # Filter degenerate boxes (width or height <= 0 after clipping)
    valid = (keep_boxes[:, 2] > keep_boxes[:, 0]) & \
            (keep_boxes[:, 3] > keep_boxes[:, 1])

    keep_boxes = keep_boxes[valid]
    keep_scores = keep_scores[valid].cpu()
    keep_labels = keep_labels[valid].cpu()

    # Format output
    results = []
    for i in range(len(keep_boxes)):
        results.append({
            'class': CLASSES[keep_labels[i].item()],
            'confidence': round(keep_scores[i].item(), 4),
            'bbox': [round(v, 1) for v in keep_boxes[i].tolist()],
        })

    return results


def main():
    args = parse_args()

    # Device
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Device: {device}')

    # ---- Load model ----
    model = FCOSDetector(
        num_classes=5,
        pretrained_backbone=False,  # weights loaded from checkpoint
    ).to(device)

    if not os.path.exists(args.checkpoint):
        print(f'ERROR: Checkpoint not found: {args.checkpoint}')
        print('Please train the model first with train.py')
        return

    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    # Use img_size from checkpoint if available
    if 'img_size' in checkpoint and args.img_size == 416:
        args.img_size = checkpoint['img_size']

    model.eval()
    print(f'Loaded model from {args.checkpoint}')
    print(f'Image size: {args.img_size}')

    # ---- Transform (no augmentation for inference) ----
    transform = DetectionTransform(img_size=args.img_size, train=False)

    # ---- Discover images ----
    image_dir = args.image_dir
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    if not image_files:
        print(f'No images found in {image_dir}')
        return

    print(f'Processing {len(image_files)} images...')

    # ---- Run inference ----
    predictions = []
    total_detections = 0
    t0 = time.time()

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)

        det_boxes = predict_single_image(
            model, img_path, transform, device,
            args.conf_thresh, args.nms_thresh,
        )

        predictions.append({
            'image_id': img_file,
            'boxes': det_boxes,
        })

        total_detections += len(det_boxes)

        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f'  [{i + 1}/{len(image_files)}] '
                  f'{fps:.1f} img/s, {total_detections} detections so far')

    elapsed = time.time() - t0

    # ---- Save predictions ----
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # ---- Summary ----
    imgs_with_det = sum(1 for p in predictions if p['boxes'])
    print(f'\nDone! {elapsed:.1f}s total')
    print(f'Total detections: {total_detections}')
    print(f'Images with detections: {imgs_with_det}/{len(predictions)}')
    print(f'Saved to: {args.output}')


if __name__ == '__main__':
    main()
