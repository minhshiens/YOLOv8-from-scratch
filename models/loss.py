import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_utils import (
    generate_grid_points,
    ltrb_to_xyxy,
    compute_ciou_loss,
)

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        
        p = torch.sigmoid(logits)

        # Binary cross-entropy (per element)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # p_t = p for positive, (1-p) for negative
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal weight: down-weight easy examples
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: balance positive/negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce
        return loss.sum()

class FCOSLoss(nn.Module):

    # Scale ranges determine which FPN level is responsible for which object sizes
    SCALE_RANGES = [
        (0, 64),       # P3 (stride 8):  small objects
        (64, 128),     # P4 (stride 16): medium objects
        (128, 1e8),    # P5 (stride 32): large objects
    ]

    STRIDES = [8, 16, 32]

    def __init__(self, num_classes=5, reg_weight=1.5):
        super().__init__()
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, predictions, targets):
        
        device = predictions[0][0].device
        batch_size = predictions[0][0].shape[0]

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_ctr_loss = torch.tensor(0.0, device=device)
        total_pos = 0

        for level_idx, (cls_logits, reg_pred, ctr_logits) in enumerate(predictions):
            B, C, H, W = cls_logits.shape
            stride = self.STRIDES[level_idx]
            scale_min, scale_max = self.SCALE_RANGES[level_idx]

            # Generate grid center points for this FPN level
            points = generate_grid_points(H, W, stride, device=device)  # (HW, 2)

            # Flatten spatial dims: (B, C, H, W) → (B, HW, C)
            cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(B, -1, C)
            reg_flat = reg_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
            ctr_flat = ctr_logits.permute(0, 2, 3, 1).reshape(B, -1, 1)

            # Process each image independently (different # of GT boxes)
            for b in range(batch_size):
                gt_boxes = targets[b]['boxes'].to(device)    # (N, 4) xyxy
                gt_labels = targets[b]['labels'].to(device)  # (N,)

                num_gt = gt_boxes.shape[0]

                if num_gt == 0:
                    # No ground truth: all locations are negative
                    neg_targets = torch.zeros(
                        points.shape[0], self.num_classes, device=device
                    )
                    total_cls_loss += self.focal_loss(cls_flat[b], neg_targets)
                    continue

                # ---- Assign targets ----
                cls_targets, reg_targets, ctr_targets, pos_mask = \
                    self._assign_targets_for_level(
                        points, gt_boxes, gt_labels,
                        scale_min, scale_max, device, stride
                    )

                num_pos = pos_mask.sum().item()
                total_pos += num_pos

                # ---- Classification loss (all locations) ----
                total_cls_loss += self.focal_loss(cls_flat[b], cls_targets)

                # ---- Regression + centerness loss (positive locations only) ----
                if num_pos > 0:
                    # Regression: CIoU loss
                    pos_reg_pred = reg_flat[b][pos_mask]      # (P, 4) ltrb
                    pos_reg_target = reg_targets[pos_mask]     # (P, 4) ltrb
                    pos_points = points[pos_mask]              # (P, 2)

                    # Convert ltrb to xyxy for CIoU computation
                    pred_boxes_xyxy = ltrb_to_xyxy(pos_reg_pred, pos_points)
                    target_boxes_xyxy = ltrb_to_xyxy(pos_reg_target, pos_points)

                    reg_loss = compute_ciou_loss(pred_boxes_xyxy, target_boxes_xyxy)
                    total_reg_loss += reg_loss * num_pos  # CIoU returns mean

                    # Centerness: BCE loss
                    pos_ctr_logits = ctr_flat[b][pos_mask].squeeze(-1)  # (P,)
                    pos_ctr_targets = ctr_targets[pos_mask]              # (P,)

                    total_ctr_loss += F.binary_cross_entropy_with_logits(
                        pos_ctr_logits, pos_ctr_targets, reduction='sum'
                    )

        # Normalize all losses by total number of positive samples
        num_pos_safe = max(total_pos, 1)
        cls_loss = total_cls_loss / num_pos_safe
        reg_loss = self.reg_weight * total_reg_loss / num_pos_safe
        ctr_loss = total_ctr_loss / num_pos_safe

        total_loss = cls_loss + reg_loss + ctr_loss

        return {
            'total': total_loss,
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'ctr': ctr_loss.detach(),
            'num_pos': total_pos,
        }

    @torch.no_grad()
    def _assign_targets_for_level(self, points, gt_boxes, gt_labels,
                                  scale_min, scale_max, device, stride):
        
        num_points = points.shape[0]
        num_gt = gt_boxes.shape[0]

        # ---- Compute distances from every point to every GT box ----
        # points: (HW, 1, 2), boxes: (1, N, 4) → distances: (HW, N, 4)
        points_x = points[:, None, 0]  # (HW, 1)
        points_y = points[:, None, 1]  # (HW, 1)

        l = points_x - gt_boxes[None, :, 0]  # (HW, N) left distance
        t = points_y - gt_boxes[None, :, 1]  # (HW, N) top distance
        r = gt_boxes[None, :, 2] - points_x  # (HW, N) right distance
        b = gt_boxes[None, :, 3] - points_y  # (HW, N) bottom distance

        ltrb = torch.stack([l, t, r, b], dim=-1)  # (HW, N, 4)

        # ---- Condition 1: point must be inside the GT box ----
        inside_mask = ltrb.min(dim=-1).values > 0  # (HW, N)

        # ---- Condition 2: Center Sampling (point must be near GT center) ----
        radius = 1.5
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0  # (N,)
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0  # (N,)
        
        # Calculate ltrb distances from points to the center bounding box
        center_l = points_x - (gt_cx[None, :] - radius * stride)
        center_t = points_y - (gt_cy[None, :] - radius * stride)
        center_r = (gt_cx[None, :] + radius * stride) - points_x
        center_b = (gt_cy[None, :] + radius * stride) - points_y
        
        center_ltrb = torch.stack([center_l, center_t, center_r, center_b], dim=-1)
        center_mask = center_ltrb.min(dim=-1).values > 0  # (HW, N)

        # ---- Condition 3: max distance must be within scale range ----
        max_dist = ltrb.max(dim=-1).values  # (HW, N)
        scale_mask = (max_dist >= scale_min) & (max_dist <= scale_max)

        # ---- Combined validity ----
        valid_mask = inside_mask & center_mask & scale_mask  # (HW, N)

        # ---- Resolve multi-assignment: pick smallest GT box area ----
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
                   (gt_boxes[:, 3] - gt_boxes[:, 1])  # (N,)

        # For invalid pairs, set area to infinity so they're never selected
        areas_expanded = gt_areas[None, :].expand(num_points, num_gt).clone()
        areas_expanded[~valid_mask] = float('inf')

        # For each point, find the GT with smallest area
        min_areas, best_gt_idx = areas_expanded.min(dim=1)  # (HW,), (HW,)

        # Positive locations: at least one valid GT assignment
        pos_mask = min_areas < float('inf')  # (HW,)

        # ---- Build target tensors ----
        cls_targets = torch.zeros(num_points, self.num_classes, device=device)
        reg_targets = torch.zeros(num_points, 4, device=device)
        ctr_targets = torch.zeros(num_points, device=device)

        if pos_mask.any():
            pos_idx = torch.where(pos_mask)[0]      # indices of positive points
            pos_gt_idx = best_gt_idx[pos_idx]        # which GT each positive maps to

            # Classification: one-hot encoding
            pos_labels = gt_labels[pos_gt_idx]
            cls_targets[pos_idx, pos_labels] = 1.0

            # Regression: (l, t, r, b) distances
            pos_ltrb = ltrb[pos_idx, pos_gt_idx]     # (P, 4)
            reg_targets[pos_idx] = pos_ltrb

            # Centerness: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
            # Values near 1 when the point is near the box center,
            # near 0 when it's near the box edge
            pos_l, pos_t, pos_r, pos_b = pos_ltrb.unbind(dim=-1)

            lr_ratio = torch.min(pos_l, pos_r) / torch.max(pos_l, pos_r).clamp(min=1e-4)
            tb_ratio = torch.min(pos_t, pos_b) / torch.max(pos_t, pos_b).clamp(min=1e-4)
            centerness = torch.sqrt((lr_ratio * tb_ratio).clamp(min=0.0))

            ctr_targets[pos_idx] = centerness

        return cls_targets, reg_targets, ctr_targets, pos_mask
