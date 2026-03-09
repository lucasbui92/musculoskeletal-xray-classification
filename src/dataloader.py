import os
from PIL import Image
from sklearn.model_selection import train_test_split

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset

class SimpleTestDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.image_paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform

    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: 
            image = self.transform(image)
        return image, os.path.basename(self.image_paths[idx])

class DatasetTransformWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: 
            x = self.transform(x)
        return x, y
        
    def __len__(self): 
        return len(self.subset)

def get_dataloaders(train_dir, test_dir, transform_train, transform_test, batch_size=32):
    master_dataset = ImageFolder(root=train_dir)
    print(f"Class mapping: {master_dataset.class_to_idx}") 

    # Train/Val Split (80/20)
    train_idx, val_idx = train_test_split(list(range(len(master_dataset))), test_size=0.2, random_state=42)

    train_subset = Subset(master_dataset, train_idx)
    val_subset = Subset(master_dataset, val_idx)

    # Apply Transforms
    train_dataset = DatasetTransformWrapper(train_subset, transform_train)
    val_dataset = DatasetTransformWrapper(val_subset, transform_test)
    test_dataset = SimpleTestDataset(folder=test_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
