import torch
import csv

from metrics import process_predictions, process_targets, calculate_map, calculate_confusion_matrix

# Train model
def train_model(model, train_loader, val_loader, device, criterion, optimizer, max_epochs=3, patience=5, save=False):
    model = model.to(device)
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    if save:
        # Update CSV header for detection metrics
        csv_filename = f"{model.__class__.__name__.lower()}_metrics.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_train_loss', 'train_total_loss', 'train_box_loss',
                            'train_conf_loss', 'train_class_loss', 'train_class_loss', 'train_noobj_loss',
                            'val_total_loss', 'val_box_loss', 'val_conf_loss', 'val_class_loss',
                            'val_noobj_loss', 'val_map'])

    # Training loop with early stopping
    for epoch in range(max_epochs):
        model.train()
        train_total_loss = 0.0
        train_box_loss = 0.0
        train_conf_loss = 0.0
        train_class_loss = 0.0
        train_noobj_loss = 0.0

        # Training phase
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            total_loss, box_loss, conf_loss, cls_loss, noobj_loss = criterion(preds, targets)
            total_loss.backward()
            optimizer.step()

            # Accumulate training losses
            train_total_loss += total_loss.item()
            train_box_loss += box_loss.item()
            train_conf_loss += conf_loss.item()
            train_class_loss += cls_loss.item()
            train_noobj_loss += noobj_loss.item()

        # Calculate training metrics
        avg_train_loss = train_total_loss / len(train_loader)
        avg_train_box = train_box_loss / len(train_loader)
        avg_train_conf = train_conf_loss / len(train_loader)
        avg_train_class = train_class_loss / len(train_loader)
        avg_train_noobj = train_noobj_loss / len(train_loader)

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        val_total_loss = val_metrics['total_loss']
        val_box_loss = val_metrics['box_loss']
        val_conf_loss = val_metrics['conf_loss']
        val_class_loss = val_metrics['class_loss']
        val_noobj_loss = val_metrics['noobj_loss']
        val_map = val_metrics['map']

        # Early stopping logic
        if val_total_loss < best_loss - 0.001:  # Threshold for meaningful improvement
            best_loss = val_total_loss
            epochs_no_improve = 0
            best_weights = model.state_dict().copy()
            print(f"Validation loss improved to {val_total_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")

        # Save metrics
        if save:
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1,
                    f"{avg_train_loss:.4f}",
                    f"{train_total_loss:.4f}",
                    f"{train_box_loss:.4f}",
                    f"{train_conf_loss:.4f}",
                    f"{train_class_loss:.4f}",
                    f"{train_noobj_loss:.4f}",
                    f"{val_total_loss:.4f}",
                    f"{val_metrics['box_loss']:.4f}",
                    f"{val_metrics['conf_loss']:.4f}",
                    f"{val_metrics['class_loss']:.4f}",
                    f"{val_metrics['noobj_loss']:.4f}",
                    f"{val_map:.4f}"
                ])

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_total_loss:.4f}")
        print(f"Train Box Loss: {avg_train_box:.4f} | Val Box Loss: {val_box_loss:.4f}")
        print(f"Train Conf Loss: {avg_train_conf:.4f} | Val Conf Loss: {val_conf_loss:.4f}")
        print(f"Train Class Loss: {avg_train_class:.4f} | Val Class Loss: {val_class_loss:.4f}")
        print(f"Train No-Obj Loss: {avg_train_noobj:.4f} | Val No-Obj Loss: {val_noobj_loss:.4f}")
        print(f"Val mAP@0.5: {val_map:.4f}")
        print("-" * 50)

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break

    # Load best model weights
    if best_weights:
        model.load_state_dict(best_weights)
        print("Loaded best model weights from early stopping")

    print('Training finished')
    return model

# Evaluate model
def evaluate_model(model, loader, device, criterion, iou_threshold=0.5):
    model.eval()
    total_loss = 0.0
    box_loss = 0.0
    conf_loss = 0.0
    class_loss = 0.0
    noobj_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss components
            total_loss, box_loss, conf_loss, class_loss, noobj_loss = criterion(outputs, targets)
            
            # Accumulate losses
            total_loss += total_loss.item()
            box_loss += box_loss.item()
            conf_loss += conf_loss.item()
            class_loss += class_loss.item()
            noobj_loss += noobj_loss.item()
            
            # Process predictions and targets for mAP calculation
            batch_preds = process_predictions(outputs)
            batch_targets = process_targets(targets)
            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)
    
    # Calculate averages
    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    avg_box = box_loss / num_batches
    avg_conf = conf_loss / num_batches
    avg_class = class_loss / num_batches
    avg_noobj = noobj_loss / num_batches
    
    # Calculate metrics
    map = calculate_map(all_preds, all_targets, iou_threshold)
    metrics_all = calculate_confusion_matrix(all_preds, all_targets)
    metrics = metrics_all['total']
    confusion_matrix_per_class = metrics_all['classes']
    cross_errors = metrics_all['cross_errors']

    # Precision calculation
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0.0

    # Recall calculation 
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0.0

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    
    return {
        'total_loss': avg_loss,
        'box_loss': avg_box,
        'conf_loss': avg_conf,
        'class_loss': avg_class,
        'noobj_loss': avg_noobj,
        'map': map['map'],
        'cat_confusion_matrix': confusion_matrix_per_class['cat'],
        'dog_confusion_matrix': confusion_matrix_per_class['dog'],
        'cross_errors': cross_errors
    }