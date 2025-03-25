import os
import subprocess
import zipfile
import pandas as pd
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import torch

class DogCatDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, target_size=(112, 112)):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        self.label_map = {'cat': 0, 'dog': 1}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row['image_id']
        label = self.label_map[row['label']]
        xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']].values
        
        # Load image
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Resize and adjust bounding box
        image = image.resize(self.target_size)
        scale_x = self.target_size[0] / orig_width
        scale_y = self.target_size[1] / orig_height
        bbox = [
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y
        ]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label), torch.tensor(bbox, dtype=torch.float32)

def check_dataset():
    # Download dataset
    dataset_name = "datamunge/dog-cat-detection"
    dataset_path = "."
    os.makedirs(dataset_path, exist_ok=True)

    # Download and unzip if not already present
    if not os.path.exists(os.path.join(dataset_path, "images")):
        subprocess.run(f"kaggle datasets download -d {dataset_name} -p {dataset_path}".split())
        with zipfile.ZipFile(os.path.join(dataset_path, "dataset.zip"), 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")
    return dataset_path


def parse_xml_annotations(annotations_dir):
    annotations = []
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
        
        # Parse XML
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        
        # Extract data
        filename = root.find('filename').text
        label = root.find('object/name').text  # Assume one object per image
        xmin = int(root.find('object/bndbox/xmin').text)
        ymin = int(root.find('object/bndbox/ymin').text)
        xmax = int(root.find('object/bndbox/xmax').text)
        ymax = int(root.find('object/bndbox/ymax').text)
        
        # Get split (trainval/test) if available
        split = root.find('set').text if root.find('set') is not None else 'trainval'
        
        annotations.append({
            'image_id': filename,
            'label': label,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'set': split
        })
    
    return pd.DataFrame(annotations)
