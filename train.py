#!/usr/bin/env python3
"""
Training script for the FCOS object detector.

Usage:
    python train.py \
        --train_data ./public/annotations/train.json \
        --val_data ./public/annotations/val.json \
        --image_dir ./public/train/images \
        --val_image_dir ./public/val/images \
        --checkpoint_dir ./models/

Saves the best model (by val mAP@0.5) to <checkpoint_dir>/best.pth.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FCOSDetector
from model.loss import FCOSLoss
from utils.box_utils import generate_grid_points, ltrb_to_xyxy
from utils.dataset import DetectionDataset, collate_fn
from utils.nms import per_class_nms


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train FCOS object detector'
    )
    # Required paths (as specified by the assignment)
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to train.json annotation file')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to val.json annotation file')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--val_image_dir', type=str, required=True,
                        help='Path to validation images directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./models/',
                        help='Directory to save model checkpoints')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--img_size', type=int, default=416,
                        help='Input image size (square)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker threads')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Number of linear warmup epochs')

    # Model options
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet-pretrained backbone')
    parser.add_argument('--no_pretrained', action='store_true', default=False,
                        help='Train backbone from scratch')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch):
    """Train for one epoch. Returns averaged loss metrics."""
    model.train()

    running_loss = 0.0
    running_cls = 0.0
    running_reg = 0.0
    running_ctr = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # Forward pass
        predictions = model(images)
        losses = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping (prevents exploding gradients)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Accumulate metrics
        running_loss += losses['total'].item()
        running_cls += losses['cls'].item()
        running_reg += losses['reg'].item()
        running_ctr += losses['ctr'].item()
        num_batches += 1

        # Log progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            avg = running_loss / num_batches
            print(f'  Batch [{batch_idx + 1}/{len(dataloader)}] '
                  f'Loss: {avg:.4f} '
                  f'(cls={running_cls / num_batches:.4f}, '
                  f'reg={running_reg / num_batches:.4f}, '
                  f'ctr={running_ctr / num_batches:.4f}) '
                  f'pos={losses["num_pos"]}')

    n = max(num_batches, 1)
    return {
        'loss': running_loss / n,
        'cls': running_cls / n,
        'reg': running_reg / n,
        'ctr': running_ctr / n,
    }


# ---------------------------------------------------------------------------
# Validation (inference + mAP evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, dataloader, device, conf_thresh=0.05, nms_thresh=0.5):
    """
    Run inference on validation set and format predictions.

    Returns list of predictions in the required JSON format.
    """
    model.eval()
    strides = [8, 16, 32]
    classes = DetectionDataset.CLASSES
    all_predictions = []

    for images, targets in dataloader:
        images = images.to(device)
        predictions = model(images)
        batch_size = images.shape[0]

        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []

            # Collect predictions from all FPN levels
            for level_idx, (cls_logits, reg_pred, ctr_logits) in \
                    enumerate(predictions):
                _, C, H, W = cls_logits.shape
                stride = strides[level_idx]

                # Generate grid points for this level
                points = generate_grid_points(H, W, stride, device=device)

                # Decode predictions
                cls_scores = cls_logits[b].sigmoid()  # (C, H, W)
                cls_scores = cls_scores.permute(1, 2, 0).reshape(-1, C)  # (HW, C)

                ctr_scores = ctr_logits[b].sigmoid()  # (1, H, W)
                ctr_scores = ctr_scores.permute(1, 2, 0).reshape(-1, 1)  # (HW, 1)

                reg = reg_pred[b].permute(1, 2, 0).reshape(-1, 4)  # (HW, 4)

                # Final score = class_score × centerness
                scores = cls_scores * ctr_scores  # (HW, C)

                # Get best class per location
                max_scores, max_labels = scores.max(dim=1)  # (HW,)

                # Filter by confidence threshold
                keep = max_scores > conf_thresh
                if not keep.any():
                    continue

                # Decode boxes from ltrb distances
                boxes = ltrb_to_xyxy(reg[keep], points[keep])

                all_boxes.append(boxes)
                all_scores.append(max_scores[keep])
                all_labels.append(max_labels[keep])

            # Merge predictions from all levels
            if all_boxes:
                merged_boxes = torch.cat(all_boxes)
                merged_scores = torch.cat(all_scores)
                merged_labels = torch.cat(all_labels)

                # Apply per-class NMS
                keep_boxes, keep_scores, keep_labels = per_class_nms(
                    merged_boxes, merged_scores, merged_labels,
                    iou_threshold=nms_thresh,
                    score_threshold=conf_thresh,
                )
            else:
                keep_boxes = torch.zeros((0, 4), device=device)
                keep_scores = torch.zeros(0, device=device)
                keep_labels = torch.zeros(0, dtype=torch.long, device=device)

            # Convert back to original image coordinates
            scale = targets[b]['scale']
            pad_x = targets[b]['pad_x']
            pad_y = targets[b]['pad_y']
            orig_w = targets[b]['orig_w']
            orig_h = targets[b]['orig_h']

            det_boxes = []
            if keep_boxes.numel() > 0:
                kb = keep_boxes.cpu().float()
                # Undo letterbox transform
                kb[:, [0, 2]] = (kb[:, [0, 2]] - pad_x) / scale
                kb[:, [1, 3]] = (kb[:, [1, 3]] - pad_y) / scale

                # Clip to original image bounds
                kb[:, 0] = kb[:, 0].clamp(min=0, max=orig_w)
                kb[:, 1] = kb[:, 1].clamp(min=0, max=orig_h)
                kb[:, 2] = kb[:, 2].clamp(min=0, max=orig_w)
                kb[:, 3] = kb[:, 3].clamp(min=0, max=orig_h)

                ks = keep_scores.cpu()
                kl = keep_labels.cpu()

                # Filter degenerate boxes (width or height <= 0)
                valid = (kb[:, 2] > kb[:, 0]) & (kb[:, 3] > kb[:, 1])

                for i in range(len(kb)):
                    if valid[i]:
                        det_boxes.append({
                            'class': classes[kl[i].item()],
                            'confidence': round(ks[i].item(), 4),
                            'bbox': [round(v, 1) for v in kb[i].tolist()],
                        })

            all_predictions.append({
                'image_id': targets[b]['image_id'],
                'boxes': det_boxes,
            })

    return all_predictions


def evaluate_predictions(val_data_path, pred_path, output_path):
    """
    Run the provided evaluation script to compute mAP@0.5.

    Returns the mAP@0.5 score, or None if evaluation fails.
    """
    try:
        result = subprocess.run(
            [sys.executable, 'public/tools/evaluate_predictions.py',
             '--ground_truth', val_data_path,
             '--predictions', pred_path,
             '--output', output_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            with open(output_path, 'r') as f:
                score = json.load(f)
            return score
        else:
            print(f'  [Eval] Error: {result.stderr[:300]}')
            return None
    except Exception as e:
        print(f'  [Eval] Exception: {e}')
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- Datasets ----
    train_dataset = DetectionDataset(
        args.train_data, args.image_dir,
        img_size=args.img_size, train=True,
    )
    val_dataset = DetectionDataset(
        args.val_data, args.val_image_dir,
        img_size=args.img_size, train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )

    print(f'Train: {len(train_dataset)} images, '
          f'Val: {len(val_dataset)} images')
    print(f'Batches per epoch: {len(train_loader)}')

    # ---- Model ----
    use_pretrained = args.pretrained and not args.no_pretrained
    model = FCOSDetector(
        num_classes=5,
        pretrained_backbone=use_pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {total_params:,} total, {trainable_params:,} trainable')

    # ---- Loss ----
    criterion = FCOSLoss(num_classes=5)

    # ---- Optimizer: SGD with momentum ----
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    # ---- LR Scheduler: cosine annealing (applied after warmup) ----
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-5,
    )

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_map = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', -1) + 1
        best_map = ckpt.get('best_map', 0.0)
        print(f'Resumed from epoch {start_epoch}, best mAP: {best_map:.4f}')

    # ---- Training loop ----
    print(f'\n{"=" * 60}')
    print(f'Starting training for {args.epochs} epochs...')
    print(f'{"=" * 60}\n')

    for epoch in range(start_epoch, args.epochs):
        # ---- Learning rate: linear warmup then cosine decay ----
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{args.epochs}  lr={current_lr:.6f}')

        # ---- Train ----
        t0 = time.time()
        metrics = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch + 1,
        )
        train_time = time.time() - t0

        print(f'  Train: loss={metrics["loss"]:.4f} '
              f'(cls={metrics["cls"]:.4f} reg={metrics["reg"]:.4f} '
              f'ctr={metrics["ctr"]:.4f})  [{train_time:.1f}s]')

        # ---- LR scheduler step (only after warmup completes) ----
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # ---- Validate every 5 epochs or in the last 10 epochs ----
        do_val = ((epoch + 1) % 5 == 0) or (epoch >= args.epochs - 10)
        if do_val:
            print('  Validating...')
            t0 = time.time()
            val_preds = validate(model, val_loader, device)
            val_time = time.time() - t0

            # Save predictions JSON
            pred_path = os.path.join(args.checkpoint_dir, 'val_predictions.json')
            with open(pred_path, 'w', encoding='utf-8') as f:
                json.dump(val_preds, f, ensure_ascii=False)

            # Compute mAP using the provided evaluation script
            score_path = os.path.join(args.checkpoint_dir, 'val_score.json')
            score = evaluate_predictions(args.val_data, pred_path, score_path)

            if score is not None:
                map50 = score['mAP@0.5']
                perf = score.get('performance_points', '?')
                print(f'  Val: mAP@0.5={map50:.4f}  '
                      f'points={perf}  [{val_time:.1f}s]')

                # Save best model
                if map50 > best_map:
                    best_map = map50
                    best_path = os.path.join(args.checkpoint_dir, 'best.pth')
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_map': best_map,
                        'img_size': args.img_size,
                    }, best_path)
                    print(f'  *** New best! Saved to {best_path} ***')
            else:
                print(f'  Val: evaluation failed  [{val_time:.1f}s]')

        # ---- Save latest checkpoint ----
        latest_path = os.path.join(args.checkpoint_dir, 'latest.pth')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_map': best_map,
            'img_size': args.img_size,
        }, latest_path)

    # ---- Done ----
    print(f'\n{"=" * 60}')
    print(f'Training complete!')
    print(f'Best mAP@0.5: {best_map:.4f}')
    print(f'Best model: {os.path.join(args.checkpoint_dir, "best.pth")}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
