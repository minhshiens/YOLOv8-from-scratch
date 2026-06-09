import torch

def compute_iou(boxes1, boxes2):
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # Intersection coordinates
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])  # (N, M)
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    # Intersection area (clamp to 0 for non-overlapping boxes)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)  # (N, M)

    # Union area
    union = area1[:, None] + area2[None, :] - inter_area

    return inter_area / union.clamp(min=1e-6)

def compute_giou_loss(pred_boxes, target_boxes):
    
    # Areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])

    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)

    # Union
    union = pred_area + target_area - inter_area
    iou = inter_area / union.clamp(min=1e-6)

    # Enclosing box (smallest box containing both boxes)
    enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * \
                   (enclose_y2 - enclose_y1).clamp(min=0)

    # GIoU = IoU - (area_enclosing - area_union) / area_enclosing
    giou = iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)

    # Loss = 1 - GIoU (mean over all pairs)
    return (1 - giou).mean()

def ltrb_to_xyxy(ltrb, points):
    
    x1 = points[..., 0] - ltrb[..., 0]
    y1 = points[..., 1] - ltrb[..., 1]
    x2 = points[..., 0] + ltrb[..., 2]
    y2 = points[..., 1] + ltrb[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_ltrb(boxes, points):
    
    l = points[..., 0] - boxes[..., 0]
    t = points[..., 1] - boxes[..., 1]
    r = boxes[..., 2] - points[..., 0]
    b = boxes[..., 3] - points[..., 1]
    return torch.stack([l, t, r, b], dim=-1)

def clip_boxes(boxes, width, height):
    
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=width)
    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=height)
    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=width)
    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=height)
    return boxes

def generate_grid_points(height, width, stride, device='cpu'):
    
    shifts_x = torch.arange(0, width, device=device) * stride + stride // 2
    shifts_y = torch.arange(0, height, device=device) * stride + stride // 2

    # Create meshgrid: shift_y is (H, W), shift_x is (H, W)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

    # Flatten to (H*W, 2)
    points = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1).float()
    return points
