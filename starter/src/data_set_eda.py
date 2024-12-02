import os
import matplotlib.pyplot as plt
from collections import Counter
from torchvision.datasets import ImageFolder

class DatasetEDA:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset = ImageFolder(self.dataset_dir)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.image_count = self._count_images()

    def _count_images(self):
        """
        Count the number of images per class in the dataset.
        """
        counts = Counter()
        for _, label in self.dataset.imgs:
            counts[label] += 1
        return counts

    def analyze_distribution(self):
        """
        Analyze and display the distribution of objects in bins.
        """
        print("Class Distribution:")
        for class_name, count in zip(self.classes, self.image_count.values()):
            print(f"{class_name}: {count} images")

        # Plot distribution
        plt.figure(figsize=(10, 6))
        plt.bar(self.classes, self.image_count.values(), color='skyblue')
        plt.xlabel("Class")
        plt.ylabel("Number of Images")
        plt.title("Class Distribution")
        plt.xticks(rotation=45)
        plt.show()

    def visualize_samples(self, num_samples=5):
        """
        Visualize sample images from each class to understand variability and complexity.
        """
        plt.figure(figsize=(15, len(self.classes) * 5))
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.dataset_dir, class_name)
            sample_images = os.listdir(class_dir)[:num_samples]

            for i, img_name in enumerate(sample_images):
                img_path = os.path.join(class_dir, img_name)
                img = plt.imread(img_path)

                plt.subplot(len(self.classes), num_samples, class_idx * num_samples + i + 1)
                plt.imshow(img)
                plt.axis('off')
                if i == 0:
                    plt.ylabel(class_name, fontsize=14)
        plt.tight_layout()
        plt.show()

    def detect_anomalies(self):
        """
        Identify and handle anomalies in the dataset, such as images with unexpected dimensions or corrupted files.
        """
        anomalies = []
        for img_path, _ in self.dataset.imgs:
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Check if the image is valid
                    if img.size[0] == 0 or img.size[1] == 0:
                        anomalies.append(img_path)
            except Exception as e:
                anomalies.append(img_path)
        
        # if anomalies:
        #     print("Anomalies detected:")
        #     for anomaly in anomalies:
        #         print(anomaly)
        # else:
        #     print("No anomalies detected.")
        return anomalies

