from dataset import check_dataset, parse_xml_annotations, DogCatDataset
import os
from torchvision import transforms
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from model import LittleYOLO
from train_eval_fns import train_model, evaluate_model

def main():
    print("Hello World!")
    dataset_path = check_dataset()

    # Load annotations
    annotations_dir = os.path.join(dataset_path, "annotations")
    df = parse_xml_annotations(annotations_dir)

    # Split into trainval/test
    trainval_df = df[df['set'] == 'trainval']
    test_df = df[df['set'] == 'test']

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.2,          # 20% of `trainval_df` becomes validation data
        stratify=trainval_df['label'],  # Preserve class balance in splits
        random_state=42         # Reproducibility
    )

    # Create datasets
    img_dir = os.path.join(dataset_path, "images")  # Folder containing images
    train_dataset = DogCatDataset(train_df, img_dir, transform=transform)
    val_dataset = DogCatDataset(val_df, img_dir, transform=transform)
    test_dataset = DogCatDataset(test_df, img_dir, transform=transform)

    # CHOICE TASK 6
    # Takes a dataset, and adds a color jittered, autoconstrasted, and grayscaled version of each image to the dataset
    def augment_dataset(dataset):

        augmented_images = []
        unchanged_labels = []
        unchanged_bboxes = []

        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

        # Apply transformations on each image in dataset
        for image, label, bbox in dataset:

            augmented_images.append(image) # Keep original image

            augmented_images.append(color_jitter(image)) # Color jitter

            contrasted = transforms.functional.autocontrast(image) # Autocontrast
            augmented_images.append(contrasted)

            grayscaled = transforms.functional.rgb_to_grayscale(image, num_output_channels=3) # Grayscale
            augmented_images.append(grayscaled)

            # Labels and bboxes unchanged
            unchanged_labels.extend([label] * 4)
            unchanged_bboxes.extend([bbox] * 4)

        # Recombine into dataset
        augmented_dataset = torch.utils.data.TensorDataset(
            torch.stack(augmented_images),
            torch.stack(unchanged_labels),
            torch.stack(unchanged_bboxes)
        )

        return augmented_dataset
    
    print("Augmenting training data: please wait...")
    train_dataset = augment_dataset(train_dataset)
    print("Augmentation complete!")

    # DataLoaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define training attributes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    model = LittleYOLO()

    # TODO must update train_eval_fns.py for the following to work
    
    '''
    print(f'\nTraining Model')
    model = train_model(model, train_loader, val_loader, device, criterion, save=True)
    
    # First evaluate on validation set
    val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
    print(f'Validation Results:')
    print(f'Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
    
    # Then evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)
    print(f'Test Results:')
    print(f'Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    '''

if __name__ == "__main__":
    main()