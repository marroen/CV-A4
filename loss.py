import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=1, C=2, λ_coord=5.0, λ_noobj=0.5):
        super().__init__()
        self.S = S  # Grid size (7x7)
        self.B = B  # Boxes per cell (1)
        self.C = C  # Classes (cat/dog)
        self.λ_coord = λ_coord
        self.λ_noobj = λ_noobj

    # TODO: Revise if gives proper IoU
    def _compute_iou(self, box1, box2):
        """Calculate IoU between two sets of boxes (x_center, y_center, w, h)"""
        # Convert to (x1, y1, x2, y2)
        box1 = torch.stack([
            box1[..., 0] - box1[..., 2]/2,
            box1[..., 1] - box1[..., 3]/2,
            box1[..., 0] + box1[..., 2]/2,
            box1[..., 1] + box1[..., 3]/2,
        ], dim=-1)
        
        box2 = torch.stack([
            box2[..., 0] - box2[..., 2]/2,
            box2[..., 1] - box2[..., 3]/2,
            box2[..., 0] + box2[..., 2]/2,
            box2[..., 1] + box2[..., 3]/2,
        ], dim=-1)

        # Calculate intersection area
        inter_left = torch.max(box1[..., 0], box2[..., 0])
        inter_right = torch.min(box1[..., 2], box2[..., 2])
        inter_top = torch.max(box1[..., 1], box2[..., 1])
        inter_bottom = torch.min(box1[..., 3], box2[..., 3])
        
        inter_area = torch.clamp(inter_right - inter_left, min=0) * \
                     torch.clamp(inter_bottom - inter_top, min=0)
        
        # Calculate union area
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)

    # TODO: Revise if this is source of bad mAP@0.5
    def forward(self, preds, targets):
        # Reshape tensors
        preds = preds.reshape(-1, self.S, self.S, self.B*5 + self.C)
        
        # Mask object/no-object cells
        obj_mask = targets[..., 4] > 0 # Target has object
        noobj_mask = targets[..., 4] == 0 # Target has no object

        # --- Object Loss ---
        obj_preds = preds[obj_mask]
        obj_targets = targets[obj_mask]

        # Calculate IoU for confidence targets
        with torch.no_grad():
            pred_boxes = obj_preds[:, :4]
            true_boxes = obj_targets[:, :4]
            ious = self._compute_iou(pred_boxes, true_boxes)
        
        # Box coordinates (x,y,w,h)
        box_loss = F.mse_loss(obj_preds[:, :4], obj_targets[:, :4], reduction='sum') * self.λ_coord
        
        # Object confidence
        obj_conf_loss = F.mse_loss(obj_preds[:, 4], obj_targets[:, 4], reduction='sum')
        
        # Class prediction
        class_loss = F.mse_loss(obj_preds[:, 5:], obj_targets[:, 5:], reduction='sum')

        # --- No-Object Loss ---
        noobj_preds = preds[noobj_mask]
        noobj_targets = targets[noobj_mask]
        noobj_conf_loss = F.mse_loss(noobj_preds[:, 4], noobj_targets[:, 4], reduction='sum') * self.λ_noobj

        # Total loss
        total_loss = (box_loss + obj_conf_loss + class_loss + noobj_conf_loss) / targets.size(0)  # Batch norm
        return total_loss, box_loss, obj_conf_loss, class_loss, noobj_conf_loss