import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataLoaderCreator:
    def __init__(self, train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.transforms = self.create_transforms()

    def create_transforms(self):
        """
        Define transformations for train, validation, and test datasets.
        """
        train_transform = transforms.Compose([
            transforms.Resize(self.img_size),  # Resize all images to the same dimensions
            transforms.RandomHorizontalFlip(p=0.5),  # Augmentation: Flip
            transforms.RandomRotation(degrees=15),  # Augmentation: Rotation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        val_test_transform = transforms.Compose([
            transforms.Resize(self.img_size),  # Resize all images to the same dimensions
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        return {"train": train_transform, "val_test": val_test_transform}

    

    def create_data_loaders(self):
        """
        Create DataLoaders for train, validation, and test datasets.
        """
        data_loaders = {}

        # train_dataset_directory = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
        # valid_dataset_directory = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
        # test_dataset_directory = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")
        
        # Train DataLoader
        train_dataset = datasets.ImageFolder(self.train_dir, transform=self.transforms["train"])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        data_loaders["train"] = train_loader

        # Validation DataLoader
        val_dataset = datasets.ImageFolder(self.val_dir, transform=self.transforms["val_test"])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        data_loaders["val"] = val_loader

        # Test DataLoader
        test_dataset = datasets.ImageFolder(self.test_dir, transform=self.transforms["val_test"])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        data_loaders["test"] = test_loader

        return data_loaders

