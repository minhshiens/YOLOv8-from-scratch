import torch
from utils.box_utils import compute_iou

def nms(boxes, scores, iou_threshold=0.5):
    
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Sort by score in descending order
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        # Pick the detection with highest confidence
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # Compute IoU between the kept box and all remaining boxes
        remaining = order[1:]
        ious = compute_iou(
            boxes[i : i + 1],   # (1, 4)
            boxes[remaining]    # (K, 4)
        ).squeeze(0)  # (K,)

        # Keep only boxes with IoU below the threshold (not suppressed)
        mask = ious <= iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def per_class_nms(boxes, scores, labels, iou_threshold=0.5, score_threshold=0.05):
    
    # Filter by confidence threshold first
    mask = scores > score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    if boxes.numel() == 0:
        return (
            torch.zeros((0, 4), device=boxes.device),
            torch.zeros(0, device=scores.device),
            torch.zeros(0, dtype=torch.long, device=labels.device),
        )

    all_keep_boxes = []
    all_keep_scores = []
    all_keep_labels = []

    # Apply NMS independently for each class
    unique_labels = labels.unique()
    for cls in unique_labels:
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        keep_idx = nms(cls_boxes, cls_scores, iou_threshold)

        all_keep_boxes.append(cls_boxes[keep_idx])
        all_keep_scores.append(cls_scores[keep_idx])
        all_keep_labels.append(torch.full(
            (len(keep_idx),), cls.item(),
            dtype=torch.long, device=labels.device
        ))

    return (
        torch.cat(all_keep_boxes, dim=0),
        torch.cat(all_keep_scores, dim=0),
        torch.cat(all_keep_labels, dim=0),
    )
