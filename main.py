import os
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from torchinfo import summary
import torchvision.transforms.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

from example import visualize_batch

from sklearn.model_selection import train_test_split
from matplotlib import patches

from dataset import check_dataset, parse_xml_annotations, DogCatDataset
from video_data import video_tensor_batch
from models import LittleYOLO, LittleYOLO_ResNet18
from loss import YOLOLoss
from train_eval_fns import train_model, evaluate_model
from plot import plot_iou_metrics

# CHOICE TASK 6
# Takes a dataset, and adds a color jittered, autoconstrasted, and grayscaled version of each image to the dataset
def augment_dataset(dataset):
    augmented_images = []
    augmented_targets = []
    
    # Apply transformations on each image-target pair
    for image, yolo_target in dataset:
        # Original image
        augmented_images.append(image)
        augmented_targets.append(yolo_target)
        
        # Random HSV adjustments
        h_change = random.uniform(-0.1, 0.1)
        s_change = random.uniform(0.8, 1.2)
        v_change = random.uniform(0.8, 1.2)
        color_img = F.adjust_hue(image, h_change)
        color_img = F.adjust_saturation(color_img, s_change)
        color_img = F.adjust_brightness(color_img, v_change)
        augmented_images.append(color_img)
        augmented_targets.append(yolo_target)
        
        # Random gaussian blur
        sigma = random.uniform(0.1, 2.0)
        blurred_img = F.gaussian_blur(image, kernel_size=3, sigma=sigma)
        augmented_images.append(blurred_img)
        augmented_targets.append(yolo_target)
        
        # Grayscale
        grayscale_img = F.rgb_to_grayscale(image, num_output_channels=3)
        augmented_images.append(grayscale_img)
        augmented_targets.append(yolo_target)

    # Convert to tensors and create dataset
    augmented_images = torch.stack(augmented_images)
    augmented_targets = torch.stack(augmented_targets)
    
    return torch.utils.data.TensorDataset(augmented_images, augmented_targets)

# Visualize processed images with ground truth and predicted bboxes overlayed
def display_processed_images(dataset, model, device):

    model.eval()

    for i in range(len(dataset)):

        # Get next image
        with torch.no_grad():
            image, target = dataset[i]
            input_img = image.unsqueeze(0).to(device)

            # Get the predicted bbox
            pred = model(input_img)
            pred = pred[0].cpu()

        # Reshape predicted bbox (S, S, 5 + num_classes)
        S = 7
        pred = pred.view(S, S, 5 + 2)

        # Denormalize image
        std = torch.tensor([0.229, 0.224, 0.225])
        mean = torch.tensor([0.485, 0.456, 0.406])
        img_disp = image.clone().detach() * std[:, None, None] + mean[:, None, None]
        img_disp = img_disp.permute(1, 2, 0).numpy().clip(0, 1)

        img_size = 112
        cell_size = img_size / S

        # Extract ground truth bbox from target
        obj_cells = torch.where(target[..., 4] == 1)
        if len(obj_cells[0]) > 0:
            cell_y_gt, cell_x_gt = obj_cells[0][0].item(), obj_cells[1][0].item()
            gt_box = target[cell_y_gt, cell_x_gt, :4].numpy()

            # Convert cell-relative bbox to image coordinates
            gt_x_center = cell_x_gt * cell_size + gt_box[0] * cell_size
            gt_y_center = cell_y_gt * cell_size + gt_box[1] * cell_size
            gt_width = gt_box[2] * img_size
            gt_height = gt_box[3] * img_size

            gt_xmin = int(gt_x_center - gt_width / 2)
            gt_ymin = int(gt_y_center - gt_height / 2)
            gt_xmax = int(gt_x_center + gt_width / 2)
            gt_ymax = int(gt_y_center + gt_height / 2)
        
        else:
            print("No ground truth bbox!")
            return

        # Extract predicted bbox
        pred_confidences = pred[..., 4]
        cell_conf, idx = torch.max(pred_confidences.view(-1), dim=0)
        cell_index = idx.item()
        cell_y_pred = cell_index // S
        cell_x_pred = cell_index % S
        pred_box = pred[cell_y_pred, cell_x_pred, :4].numpy()
        pred_class_idx = torch.argmax(pred[cell_y_pred, cell_x_pred, 5:]).item()
        pred_label = 'Cat' if pred_class_idx == 0 else 'Dog'

        # Convert predicted bbox from cell coordinates to image coordinates
        pred_x_center = cell_x_pred * cell_size + pred_box[0] * cell_size
        pred_y_center = cell_y_pred * cell_size + pred_box[1] * cell_size
        pred_width = pred_box[2] * img_size
        pred_height = pred_box[3] * img_size

        pred_xmin = int(pred_x_center - pred_width / 2)
        pred_ymin = int(pred_y_center - pred_height / 2)
        pred_xmax = int(pred_x_center + pred_width / 2)
        pred_ymax = int(pred_y_center + pred_height / 2)

        # Draw bboxes
        fig, ax = plt.subplots(1)
        ax.imshow(img_disp)
        
        # Blue ground truth bbox
        rect_gt = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin,
                                    linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect_gt)

        # Red predicted bbox
        rect_pred = patches.Rectangle((pred_xmin, pred_ymin), pred_xmax - pred_xmin, pred_ymax - pred_ymin,
                                    linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect_pred)

        # State predicted class
        ax.set_title(f"Prediction: {pred_label}")
        plt.axis('off')
        plt.show()

def main():
    dataset_path = check_dataset()

    # Load annotations
    annotations_dir = os.path.join(dataset_path, "annotations")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = parse_xml_annotations(annotations_dir)

    # Handle dataset splits
    if 'set' in df.columns:
        trainval_df = df[df['set'] == 'trainval']
        test_df = df[df['set'] == 'test']
        
        # If no test samples in XMLs, split manually
        if test_df.empty:
            trainval_df, test_df = train_test_split(
                df,
                test_size=0.2,
                stratify=df['label'],
                random_state=42
            )
    else:
        # Create fresh splits if no set info
        trainval_df, test_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df['label'],
            random_state=42
        )

    # Split trainval into train/val
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.2,
        stratify=trainval_df['label'],
        random_state=42
    )

    # Check ratio of cats to dogs in each split
    train_cat_count = (train_df['label'] == 'cat').sum()
    train_dog_count = (train_df['label'] == 'dog').sum()
    print(f"Cats: {train_cat_count}, Dogs: {train_dog_count}")
    train_ratio = train_cat_count / train_dog_count
    print("train ratio:", train_ratio)

    val_cat_count = (val_df['label'] == 'cat').sum()
    val_dog_count = (val_df['label'] == 'dog').sum()
    print(f"Cats: {val_cat_count}, Dogs: {val_dog_count}")
    val_ratio = val_cat_count / val_dog_count
    print("val ratio:", val_ratio)

    test_cat_count = (test_df['label'] == 'cat').sum()
    test_dog_count = (test_df['label'] == 'dog').sum()
    print(f"Cats: {test_cat_count}, Dogs: {test_dog_count}")
    test_ratio = test_cat_count / test_dog_count
    print("test ratio:", test_ratio)

    # Create datasets
    img_dir = os.path.join(dataset_path, "images")  # Folder containing images
    train_dataset = DogCatDataset(train_df, img_dir, transform=transform)
    val_dataset = DogCatDataset(val_df, img_dir, transform=transform)
    test_dataset = DogCatDataset(test_df, img_dir, transform=transform)

    # After creating train_dataset, val_dataset, test_dataset:
    print("\nDataset Sizes:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    '''
    print("Augmenting training data: please wait...")
    train_dataset = augment_dataset(train_dataset)
    print("Augmentation complete!")'''

    print(f"{len(train_dataset)} samples")

    # DataLoaders
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #visualize_batch(train_loader)

    # Define training attributes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = YOLOLoss()

    print("\nWhich model would you like to run?")
    print("   1 -> LittleYOLO")
    print("   2 -> LittleYOLO_ResNet18")
    user_input = input("Choose wisely: ")

    if user_input == "1":
        model = LittleYOLO()
        print("\nLittleYOLO selected")
    else:
        model = LittleYOLO_ResNet18()
        print("\nLittleYOLO_ResNet18 selected")
    
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Using Adam optimizer with learning rate:", learning_rate)
    print("Batch size:", batch_size)
    
    print(f'\nTraining Model')
    model = train_model(model, train_loader, val_loader, device, criterion, optimizer, max_epochs=30, save=True)
    summary(model, input_size=(1, 3, 112, 112))

    # Save the model weights after training
    model_weights = model.to(device)
    torch.save(model_weights.state_dict(), "littleyolo_weights.pth")
    
    val_metrics = evaluate_model(model, val_loader, device, criterion)
    print(f"""
    Val Results:
    Total Loss: {val_metrics['total_loss']:.4f}
    Box Loss: {val_metrics['box_loss']:.4f}
    Confidence Loss: {val_metrics['conf_loss']:.4f}
    Class Loss: {val_metrics['class_loss']:.4f}
    No-Object Loss: {val_metrics['noobj_loss']:.4f}
    mAP@0.5: {val_metrics['map']:.4f}
    Cat Confusion Matrix: {val_metrics['cat_confusion_matrix']}
    Dog Confusion Matrix: {val_metrics['dog_confusion_matrix']}
    Cross Errors: {val_metrics['cross_errors']}
    """)
    
    # Then evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device, criterion)
    print(f"""
    Test Results:
    Total Loss: {test_metrics['total_loss']:.4f}
    Box Loss: {test_metrics['box_loss']:.4f}
    Confidence Loss: {test_metrics['conf_loss']:.4f}
    Class Loss: {test_metrics['class_loss']:.4f}
    No-Object Loss: {test_metrics['noobj_loss']:.4f}
    mAP@0.5: {test_metrics['map']:.4f}
    Cat Confusion Matrix: {test_metrics['cat_confusion_matrix']}
    Dog Confusion Matrix: {test_metrics['dog_confusion_matrix']}
    Cross Errors: {test_metrics['cross_errors']}
    """)

    plot_iou_metrics(model, test_loader, device, criterion)

    # Display test images with bboxes
    display_processed_images(test_dataset, model, device)
    

if __name__ == "__main__":
    main()