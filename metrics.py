from collections import defaultdict
import torch
import numpy as np

def compute_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes
    Args:
        box1: [x_center, y_center, width, height] (normalized 0-1)
        box2: [x_center, y_center, width, height] (normalized 0-1)
    Returns:
        iou: Intersection over Union score
    """
    # Convert from center coordinates to corner coordinates
    box1 = [
        box1[0] - box1[2]/2,  # x1
        box1[1] - box1[3]/2,  # y1
        box1[0] + box1[2]/2,  # x2
        box1[1] + box1[3]/2   # y2
    ]
    
    box2 = [
        box2[0] - box2[2]/2,
        box2[1] - box2[3]/2,
        box2[0] + box2[2]/2,
        box2[1] + box2[3]/2
    ]

    # Calculate intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union

def process_predictions(preds, confidence_threshold=0.3, S=7, B=1, C=2):
    """
    Convert raw model output to detection format
    Args:
        preds: Tensor of shape [batch, S*S*(5*B + C)] 
               or [batch, S, S, 5*B + C]
    """
    processed = []
    batch_size = preds.size(0)
    
    # Reshape if needed
    if preds.dim() == 2:  # Handle flattened output
        preds = preds.view(batch_size, S, S, 5*B + C)
    
    for b in range(batch_size):
        for i in range(S):
            for j in range(S):
                # Get confidence score
                conf = preds[b, i, j, 4].item()
                if conf < confidence_threshold:
                    continue
                
                # Get class probabilities
                class_probs = preds[b, i, j, 5:].cpu().numpy()
                class_id = np.argmax(class_probs)
                
                # Get box coordinates
                x_center, y_center, w, h = preds[b, i, j, :4].cpu().numpy()
                
                processed.append((
                    int(class_id),
                    float(conf),
                    [float(x_center), float(y_center), float(w), float(h)],
                    int(b)  # Using batch index as image ID
                ))
    
    return processed

def process_targets(targets):
    """
    Convert YOLO-formatted ground truth to mAP evaluation format
    
    Args:
        targets: Tensor of shape [batch_size, S, S, 5+C] 
                 (YOLO format from your dataset)
    
    Returns:
        list: Ground truth entries as (class_id, box, image_id)
    """
    gt_entries = []
    batch_size = targets.size(0)
    
    for batch_idx in range(batch_size):
        for i in range(7):  # S=7 grid cells
            for j in range(7):
                # Check if cell contains object
                if targets[batch_idx, i, j, 4] > 0.5:  # Confidence threshold
                    # Get class ID (cat=0, dog=1)
                    class_probs = targets[batch_idx, i, j, 5:]
                    class_id = torch.argmax(class_probs).item()
                    
                    # Get box coordinates (normalized 0-1)
                    x_center, y_center, w, h = targets[batch_idx, i, j, :4].cpu().numpy()
                    box = [x_center, y_center, w, h]
                    
                    gt_entries.append((class_id, box, batch_idx))
    
    return gt_entries

def calculate_map(predictions, targets, iou_threshold=0.5):
    """Calculate mean Average Precision for object detection"""
    aps = []
    
    # Group by class
    class_stats = defaultdict(lambda: {'scores': [], 'tp': [], 'gt': 0})
    
    # Process all predictions
    for pred in predictions:
        pred_class, pred_conf, pred_box, img_id = pred
        matched = False
        
        # Find matching ground truth
        for gt in targets:
            gt_class, gt_box, gt_img_id = gt
            
            if gt_img_id != img_id:
                continue
                
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold and pred_class == gt_class:
                class_stats[pred_class]['tp'].append(1)
                matched = True
                break
                
        if not matched:
            class_stats[pred_class]['tp'].append(0)
        
        class_stats[pred_class]['scores'].append(pred_conf)
    
    # Count ground truths per class
    for gt in targets:
        class_stats[gt[0]]['gt'] += 1
    
    # Calculate AP per class
    for cls_id, stats in class_stats.items():
        if stats['gt'] == 0:
            continue
            
        # Convert to numpy arrays
        scores = np.array(stats['scores'])
        tp = np.array(stats['tp'])
        
        # Sort by confidence score descending
        sort_idx = np.argsort(-scores)
        tp = tp[sort_idx]
        
        # Compute cumulative TP/FP
        fp = 1 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        
        # Compute precision/recall
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall = tp_cum / (stats['gt'] + 1e-6)
        
        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recall >= t
            if np.any(mask):
                ap += np.max(precision[mask]) / 11
                
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0