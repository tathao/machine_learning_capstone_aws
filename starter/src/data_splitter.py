import os
import shutil
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, base_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.validate_ratios()
        self.create_output_dirs()

    def validate_ratios(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    def create_output_dirs(self):
        for split in ["train", "val", "test"]:
            for folder_name in os.listdir(self.base_dir):
                output_path = os.path.join(self.output_dir, split, folder_name)
                os.makedirs(output_path, exist_ok=True)

    def split_folder(self, folder_name):
        folder_path = os.path.join(self.base_dir, folder_name)
        if os.path.isdir(folder_path):
            images = os.listdir(folder_path)
            
            # Split into training, validation, and test sets
            train_imgs, temp_imgs = train_test_split(images, test_size=(1 - self.train_ratio), random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(self.test_ratio / (self.val_ratio + self.test_ratio)), random_state=42)
            
            # Move images to corresponding folders
            self.copy_images(train_imgs, folder_path, "train", folder_name)
            self.copy_images(val_imgs, folder_path, "val", folder_name)
            self.copy_images(test_imgs, folder_path, "test", folder_name)

    def copy_images(self, images, src_folder, split, folder_name):
        for img in images:
            src_path = os.path.join(src_folder, img)
            dest_path = os.path.join(self.output_dir, split, folder_name, img)
            shutil.copy(src_path, dest_path)

    def execute_split(self):
        for folder_name in os.listdir(self.base_dir):
            self.split_folder(folder_name)
        print("Data successfully split into train, val, and test sets.")

