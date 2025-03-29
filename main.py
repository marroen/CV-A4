from dataset import check_dataset, parse_xml_annotations, DogCatDataset, visualize_sample
import os
from torchvision import transforms
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

from model import LittleYOLO
from loss import YOLOLoss
from train_eval_fns import train_model, evaluate_model
import xml.etree.ElementTree as ET

from torchinfo import summary


def yolo_collate(batch):
    """Custom collate function for YOLO-formatted batches"""
    images = [item[0] for item in batch]  # List of image tensors
    targets = [item[1] for item in batch] # List of YOLO target tensors
    return torch.stack(images), torch.stack(targets)

def main():
    print("Hello World!")
    
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

    # Assuming `df` is your DataFrame loaded via `parse_xml_annotations`
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
    
    print("\nOriginal Train Sample:")
    visualize_sample(train_dataset, index=0)

    # CHOICE TASK 6
    # Takes a dataset, and adds a color jittered, autoconstrasted, and grayscaled version of each image to the dataset
    def augment_dataset(dataset):
        augmented_images = []
        augmented_targets = []

        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        
        # Apply transformations on each image-target pair
        for image, yolo_target in dataset:
            # Original image
            augmented_images.append(image)
            augmented_targets.append(yolo_target)
            
            # Color jittered
            jittered_img = color_jitter(image)
            augmented_images.append(jittered_img)
            augmented_targets.append(yolo_target)
            
            # Autocontrast
            contrasted_img = transforms.functional.autocontrast(image)
            augmented_images.append(contrasted_img)
            augmented_targets.append(yolo_target)
            
            # Grayscaled (keep 3 channels)
            grayscale_img = transforms.functional.rgb_to_grayscale(image, num_output_channels=3)
            augmented_images.append(grayscale_img)
            augmented_targets.append(yolo_target)

        # Convert to tensors and create dataset
        augmented_images = torch.stack(augmented_images)
        augmented_targets = torch.stack(augmented_targets)
        
        return torch.utils.data.TensorDataset(augmented_images, augmented_targets)
    
    print("Augmenting training data: please wait...")
    train_dataset = augment_dataset(train_dataset)
    print("Augmentation complete!")

    # After augmentation:
    print(f"Augmented Train: {len(train_dataset)} samples (should be 4x original)")

    # DataLoaders
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define training attributes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = YOLOLoss()
    model = LittleYOLO()
    
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Using Adam optimizer with learning rate:", learning_rate)
    print("Batch size:", batch_size)
    
    print(f'\nTraining Model')
    model = train_model(model, train_loader, val_loader, device, criterion, optimizer, max_epochs=30, save=False)
    summary(model, input_size=(1, 3, 112, 112))
    
    # First evaluate on validation set
    val_metrics = evaluate_model(model, val_loader, device, criterion)
    print(f"""
    Val Results:
    Total Loss: {val_metrics['total_loss']:.4f}
    Box Loss: {val_metrics['box_loss']:.4f}
    Confidence Loss: {val_metrics['conf_loss']:.4f}
    Class Loss: {val_metrics['class_loss']:.4f}
    No-Object Loss: {val_metrics['noobj_loss']:.4f}
    mAP@0.5: {val_metrics['mAP']:.4f}
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
    mAP@0.5: {test_metrics['mAP']:.4f}
    """)
    

if __name__ == "__main__":
    main()