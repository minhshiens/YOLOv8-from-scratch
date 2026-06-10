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

def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001):
    """
    Gaussian Soft-NMS: instead of hard-removing overlapping boxes,
    decay their scores with a Gaussian function.
    Preserves detections of overlapping objects (e.g. people standing close).
    """
    N = boxes.shape[0]
    if N == 0:
        return (torch.zeros(0, dtype=torch.long, device=boxes.device),
                torch.zeros(0, device=scores.device))

    scores = scores.clone()
    indices = torch.arange(N, device=boxes.device)
    keep = []
    keep_scores = []

    while indices.numel() > 0:
        # Pick the box with highest current score
        max_pos = scores[indices].argmax()
        max_idx = indices[max_pos]

        keep.append(max_idx.item())
        keep_scores.append(scores[max_idx].item())

        if indices.numel() == 1:
            break

        # Remove current box from candidates
        remaining_mask = torch.ones(indices.numel(), dtype=torch.bool, device=boxes.device)
        remaining_mask[max_pos] = False
        remaining = indices[remaining_mask]

        # Compute IoU between kept box and remaining
        ious = compute_iou(boxes[max_idx:max_idx+1], boxes[remaining]).squeeze(0)

        # Gaussian decay: high-overlap boxes get their scores reduced
        scores[remaining] *= torch.exp(-(ious ** 2) / sigma)

        # Remove boxes that fell below threshold
        valid = scores[remaining] > score_threshold
        indices = remaining[valid]

    keep_idx = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    new_scores = torch.tensor(keep_scores, dtype=torch.float32, device=boxes.device)
    return keep_idx, new_scores

def per_class_nms(boxes, scores, labels, iou_threshold=0.5, score_threshold=0.05,
                  use_soft=False, sigma=0.5):
    
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

        if use_soft:
            keep_idx, updated_scores = soft_nms(
                cls_boxes, cls_scores, sigma=sigma,
                score_threshold=score_threshold,
            )
            all_keep_boxes.append(cls_boxes[keep_idx])
            all_keep_scores.append(updated_scores)
        else:
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

def weighted_boxes_fusion(views_boxes, views_scores, views_labels,
                          iou_threshold=0.55, score_threshold=0.001):
    """
    Weighted Boxes Fusion (WBF): merge predictions from multiple TTA views.
    Instead of picking one box (NMS), WBF averages coordinates weighted by
    confidence scores, producing more precise localization.
    """
    num_views = len(views_boxes)
    if num_views == 0:
        return (torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long))

    device = views_boxes[0].device if views_boxes[0].numel() > 0 else 'cpu'

    # Merge all view predictions
    all_boxes = torch.cat(views_boxes)
    all_scores = torch.cat(views_scores)
    all_labels = torch.cat(views_labels)

    if all_boxes.numel() == 0:
        return (torch.zeros((0, 4), device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    final_boxes = []
    final_scores = []
    final_labels = []

    # Process each class independently
    for cls in all_labels.unique():
        cls_mask = all_labels == cls
        cls_boxes = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]

        # Sort by score descending
        order = cls_scores.argsort(descending=True)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        # Greedy clustering
        cluster_box_lists = []   # list of lists of boxes
        cluster_score_lists = [] # list of lists of scores
        cluster_avg_box = []     # weighted average box per cluster

        for i in range(cls_boxes.shape[0]):
            box = cls_boxes[i]
            score = cls_scores[i].item()

            if score < score_threshold:
                continue

            # Find best matching cluster
            matched_cluster = -1
            best_iou = iou_threshold

            for c_idx in range(len(cluster_avg_box)):
                iou_val = compute_iou(
                    box.unsqueeze(0), cluster_avg_box[c_idx].unsqueeze(0)
                ).item()
                if iou_val > best_iou:
                    best_iou = iou_val
                    matched_cluster = c_idx

            if matched_cluster >= 0:
                # Add to existing cluster and recompute weighted average
                cluster_box_lists[matched_cluster].append(box)
                cluster_score_lists[matched_cluster].append(score)

                all_b = torch.stack(cluster_box_lists[matched_cluster])
                all_s = torch.tensor(cluster_score_lists[matched_cluster],
                                     device=device)
                weights = all_s / all_s.sum()
                cluster_avg_box[matched_cluster] = (all_b * weights.unsqueeze(1)).sum(0)
            else:
                # Create new cluster
                cluster_box_lists.append([box])
                cluster_score_lists.append([score])
                cluster_avg_box.append(box.clone())

        # Extract final predictions from clusters
        for c_idx in range(len(cluster_avg_box)):
            scores_sum = sum(cluster_score_lists[c_idx])
            # Normalize by number of views: boxes seen in more views score higher
            fused_score = scores_sum / num_views

            final_boxes.append(cluster_avg_box[c_idx])
            final_scores.append(fused_score)
            final_labels.append(cls.item())

    if not final_boxes:
        return (torch.zeros((0, 4), device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    return (torch.stack(final_boxes).to(device),
            torch.tensor(final_scores, device=device),
            torch.tensor(final_labels, dtype=torch.long, device=device))
