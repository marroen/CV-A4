from collections import defaultdict
import torch
import numpy as np

# Compute IoU between two bounding boxes
def compute_iou(box1, box2):
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
    
    # No intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union

# Process model output and prepare for mAP calculation
def process_predictions(preds, confidence_threshold=0.3, S=7, B=1, C=2):
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

# Process ground truth targets and prepare for mAP calculation
def process_targets(targets):
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

# Get the mAP per class
def calculate_map(predictions, targets, iou_threshold=0.5):
    aps = []
    class_stats = defaultdict(lambda: {'scores': [], 'tp': [], 'gt': 0})

    # Process predictions and calculate TP/FP for AP
    for pred in predictions:
        pred_class, pred_conf, pred_box, img_id = pred
        matched = False
        
        # Find best matching ground truth
        for gt in targets:
            gt_class, gt_box, gt_img_id = gt
            if gt_img_id != img_id:
                continue
                
            # Check if IoU is above threshold and classes match
            if compute_iou(pred_box, gt_box) >= iou_threshold and pred_class == gt_class:
                class_stats[pred_class]['tp'].append(1)
                matched = True
                break
        
        # If no match, add 0 TP
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
            
        # Sort predictions by confidence
        scores = np.array(stats['scores'])
        tp = np.array(stats['tp'])
        sort_idx = np.argsort(-scores)
        tp_sorted = tp[sort_idx]
        
        # Calculate precision and recall
        fp_sorted = 1 - tp_sorted
        tp_cum = np.cumsum(tp_sorted)
        fp_cum = np.cumsum(fp_sorted)
        
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall = tp_cum / (stats['gt'] + 1e-6)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recall >= t
            if np.any(mask):
                ap += np.max(precision[mask]) / 11
        aps.append(ap)

    return {
        'map': np.mean(aps) if aps else 0.0,
        'class_aps': {cls_id: ap for cls_id, ap in zip(class_stats.keys(), aps)}
    }

# Calculate confusion matrix
def calculate_confusion_matrix(predictions, targets, iou_threshold=0.5):
    stats = {
        'total': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'classes': {
            'cat': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'dog': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        },
        'cross_errors': {'cat_as_dog': 0, 'dog_as_cat': 0}
    }
    
    # Per-image tracking: {img_id: {'gt': {'cat': set(), 'dog': set()}, 'pred': {'cat': set(), 'dog': set()}}}
    image_data = defaultdict(lambda: {
        'gt': defaultdict(set),
        'pred': defaultdict(set)
    })

    matched_gt_keys = set()  # Tracks (img_id, gt_idx) to prevent double-counting

    # Process predictions and track class-cell relationships
    for pred in predictions:
        pred_class_idx, _, pred_box, img_id = pred
        pred_class = 'dog' if pred_class_idx == 1 else 'cat'
        cell_x = int(pred_box[0] * 7)
        cell_y = int(pred_box[1] * 7)
        image_data[img_id]['pred'][pred_class].add((cell_x, cell_y))

        # Find best matching ground truth
        best_iou = 0
        best_gt_key = None
        for gt_idx, gt in enumerate(targets):
            gt_class_idx, gt_box, gt_img_id = gt
            if gt_img_id != img_id:
                continue

            # Calculate IoU
            iou = compute_iou(pred_box, gt_box)
            gt_class = 'dog' if gt_class_idx == 1 else 'cat'
            gt_cell_x = int(gt_box[0] * 7)
            gt_cell_y = int(gt_box[1] * 7)
            image_data[img_id]['gt'][gt_class].add((gt_cell_x, gt_cell_y))

            # Update best match
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_key = (img_id, gt_idx)

        # Update TP/FP/cross-errors
        if best_gt_key:
            gt_img_id, gt_idx = best_gt_key
            gt_class_idx, _, _ = targets[gt_idx]
            gt_class = 'dog' if gt_class_idx == 1 else 'cat'

            # Check if GT already matched
            if best_gt_key not in matched_gt_keys:
                if pred_class == gt_class:
                    stats['classes'][pred_class]['tp'] += 1
                    stats['total']['tp'] += 1
                else:
                    stats['cross_errors'][f'{gt_class}_as_{pred_class}'] += 1
                    stats['classes'][pred_class]['fp'] += 1
                    stats['classes'][gt_class]['fn'] += 1
                matched_gt_keys.add(best_gt_key)
        else:
            stats['classes'][pred_class]['fp'] += 1
            stats['total']['fp'] += 1

    # Process unmatched ground truths (FN)
    for gt_idx, gt in enumerate(targets):
        gt_class_idx, gt_box, gt_img_id = gt
        gt_class = 'dog' if gt_class_idx == 1 else 'cat'
        gt_key = (gt_img_id, gt_idx)
        
        if gt_key not in matched_gt_keys:
            stats['classes'][gt_class]['fn'] += 1
            stats['total']['fn'] += 1

    # Calculate TN per class
    for img_id in image_data:
        all_cells = {(x, y) for x in range(7) for y in range(7)}
        
        for class_name in ['cat', 'dog']:
            # Cells with this class in ground truth
            gt_cells = image_data[img_id]['gt'][class_name]
            # Cells predicted as this class
            pred_cells = image_data[img_id]['pred'][class_name]
            
            # True negatives = cells without GT of this class AND without predictions of this class
            tn_cells = len(all_cells - gt_cells - pred_cells)
            stats['classes'][class_name]['tn'] += tn_cells
            stats['total']['tn'] += tn_cells  # Global TN if needed

    return stats