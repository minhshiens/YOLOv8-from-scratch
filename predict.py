
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from models import FCOSDetector
from utils.box_utils import generate_grid_points, ltrb_to_xyxy
from utils.nms import per_class_nms, weighted_boxes_fusion
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
    parser.add_argument('--conf_thresh', type=float, default=0.01,
                        help='Confidence threshold for filtering')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detected if omitted)')
    return parser.parse_args()

@torch.no_grad()
def predict_single_image(model, image_path, base_img_size, device,
                         conf_thresh, nms_thresh):
    
    strides = [8, 16, 32]

    # Load image
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size

    # Collect predictions per TTA view for Weighted Boxes Fusion
    views_boxes = []
    views_scores = []
    views_labels = []

    # Test-Time Augmentation (TTA): Multi-Scale + Flips
    scales = [0.8, 1.0, 1.2]
    
    for scale_factor in scales:
        # Determine optimal size (multiple of 32)
        target_size = int(base_img_size * scale_factor)
        target_size = (target_size + 31) // 32 * 32
        
        transform = DetectionTransform(img_size=target_size, train=False)

        for flip in [False, True]:
            view_boxes = []
            view_scores = []
            view_labels = []

            if flip:
                img_in = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img_in = image

            # Apply transforms (resize + normalize)
            empty_boxes = np.zeros((0, 4), dtype=np.float32)
            empty_labels = np.array([], dtype=np.int64)
            img_tensor, _, _, meta = transform(img_in, empty_boxes, empty_labels)
            img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

            # Forward pass
            predictions = model(img_tensor)

            for level_idx, (cls_logits, reg_pred, ctr_logits) in enumerate(predictions):
                _, C, H, W = cls_logits.shape
                stride = strides[level_idx]

                # Generate grid points
                points = generate_grid_points(H, W, stride, device=device)  # (HW, 2)

                # Decode outputs
                cls_scores = cls_logits[0].sigmoid().permute(1, 2, 0).reshape(-1, C)
                ctr_scores = ctr_logits[0].sigmoid().permute(1, 2, 0).reshape(-1, 1)
                reg = reg_pred[0].permute(1, 2, 0).reshape(-1, 4)  # (HW, 4) ltrb

                # Centerness penalty
                scores = cls_scores * (ctr_scores ** 2)

                max_scores, max_labels = scores.max(dim=1)

                # Filter by confidence
                keep = max_scores > conf_thresh
                if not keep.any():
                    continue

                # Top-k filter: cap candidates per FPN level to prevent slowdown
                max_per_level = 1000
                if keep.sum() > max_per_level:
                    topk_vals, topk_idx = max_scores[keep].topk(max_per_level)
                    keep_where = torch.where(keep)[0][topk_idx]
                    keep = torch.zeros_like(keep)
                    keep[keep_where] = True

                # Decode boxes to xyxy
                boxes = ltrb_to_xyxy(reg[keep], points[keep])

                # Convert immediately to original image coordinates
                boxes = boxes.cpu().float()
                scale = meta['scale']
                pad_x = meta['pad_x']
                pad_y = meta['pad_y']

                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

                # If flipped, invert x coordinates
                if flip:
                    x_min_old = boxes[:, 0].clone()
                    x_max_old = boxes[:, 2].clone()
                    boxes[:, 0] = orig_w - x_max_old
                    boxes[:, 2] = orig_w - x_min_old

                view_boxes.append(boxes.to(device))
                view_scores.append(max_scores[keep])
                view_labels.append(max_labels[keep])

            # Per-view Soft-NMS to clean up duplicates within each view
            if view_boxes:
                vb = torch.cat(view_boxes)
                vs = torch.cat(view_scores)
                vl = torch.cat(view_labels)
                vb, vs, vl = per_class_nms(
                    vb, vs, vl, iou_threshold=nms_thresh,
                    score_threshold=conf_thresh, use_soft=True,
                )
                views_boxes.append(vb)
                views_scores.append(vs)
                views_labels.append(vl)

    # Weighted Boxes Fusion across all TTA views
    keep_boxes, keep_scores, keep_labels = weighted_boxes_fusion(
        views_boxes, views_scores, views_labels,
        iou_threshold=0.55, score_threshold=conf_thresh,
    )

    if keep_boxes.shape[0] == 0:
        return []

    keep_boxes = keep_boxes.cpu().float()

    # Clip to original image bounds
    keep_boxes[:, 0] = keep_boxes[:, 0].clamp(min=0, max=orig_w)
    keep_boxes[:, 1] = keep_boxes[:, 1].clamp(min=0, max=orig_h)
    keep_boxes[:, 2] = keep_boxes[:, 2].clamp(min=0, max=orig_w)
    keep_boxes[:, 3] = keep_boxes[:, 3].clamp(min=0, max=orig_h)

    # Filter degenerate boxes
    valid = (keep_boxes[:, 2] > keep_boxes[:, 0]) & \
            (keep_boxes[:, 3] > keep_boxes[:, 1])

    keep_boxes = keep_boxes[valid]
    keep_scores = keep_scores[valid].cpu()
    keep_labels = keep_labels[valid].cpu()

    # Format output
    results = []
    for i in range(len(keep_boxes)):
        bbox = [round(v, 3) for v in keep_boxes[i].tolist()]
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            results.append({
                'class': CLASSES[keep_labels[i].item()],
                'confidence': round(keep_scores[i].item(), 4),
                'bbox': bbox,
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
            model, img_path, args.img_size, device,
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
