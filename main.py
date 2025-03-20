from dataset import check_dataset, parse_xml_annotations, DogCatDataset
import os
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split

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

    # DataLoaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    main()