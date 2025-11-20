import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ISIC2017Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_csv, transform=None, task="segmentation"):
        """
        Args:
            image_dir (str): Path to the directory with images.
            mask_dir (str): Path to the directory with segmentation masks.
            label_csv (str): Path to the CSV file with binary classification labels.
            transform (callable, optional): Optional transform to be applied on an image.
            task (str): Task type, either "segmentation" or "classification".
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.labels = pd.read_csv(label_csv, index_col=0)  # Load labels using image_id as index
        self.image_ids = list(self.labels.index)
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.task == "segmentation":
            mask_path = os.path.join(self.mask_dir, f"{img_id}_segmentation.png")
            mask = Image.open(mask_path).convert("L")  # Grayscale for segmentation masks

            if self.transform:
                image = self.transform["image"](image)
                mask = self.transform["mask"](mask)
            return image, mask

        elif self.task == "classification":
            label = self.labels.loc[img_id, "melanoma"]  # Access the 'melanoma' column
            
            # Reshape label to be [16, 1] instead of [16]
            label = torch.tensor(label).view(-1, 1).squeeze(-1)  # This ensures the shape is [16, 1]
            
            if self.transform:
                image = self.transform(image)
            return image, label
        

def get_dataloaders(root_dir, transform_image, transform_mask, batch_size=16):
    train_image_dir = os.path.join(root_dir, "train", "ISIC-2017_Training_Data")
    train_mask_dir = os.path.join(root_dir, "train", "ISIC-2017_Training_Part1_GroundTruth")
    train_label_csv = os.path.join(root_dir, "train", "ISIC-2017_Training_Part3_GroundTruth.csv")

    val_image_dir = os.path.join(root_dir, "val", "ISIC-2017_Validation_Data")
    val_mask_dir = os.path.join(root_dir, "val", "ISIC-2017_Validation_Part1_GroundTruth")
    val_label_csv = os.path.join(root_dir, "val", "ISIC-2017_Validation_Part3_GroundTruth.csv")

    test_image_dir = os.path.join(root_dir, "test", "ISIC-2017_Test_v2_Data")
    test_mask_dir = os.path.join(root_dir, "test", "ISIC-2017_Test_v2_Part1_GroundTruth")
    test_label_csv = os.path.join(root_dir, "test", "ISIC-2017_Test_v2_Part3_GroundTruth.csv")

    # Create datasets for segmentation
    train_dataset_seg = ISIC2017Dataset(
        train_image_dir, train_mask_dir, train_label_csv,
        transform={"image": transform_image, "mask": transform_mask},
        task="segmentation"
    )
    val_dataset_seg = ISIC2017Dataset(
        val_image_dir, val_mask_dir, val_label_csv,
        transform={"image": transform_image, "mask": transform_mask},
        task="segmentation"
    )
    test_dataset_seg = ISIC2017Dataset(
        test_image_dir, test_mask_dir, test_label_csv,
        transform={"image": transform_image, "mask": transform_mask},
        task="segmentation"
    )

    # Create datasets for classification (only image transform is needed)
    train_dataset_cls = ISIC2017Dataset(
        train_image_dir, train_mask_dir, train_label_csv,
        transform=transform_image, task="classification"
    )
    val_dataset_cls = ISIC2017Dataset(
        val_image_dir, val_mask_dir, val_label_csv,
        transform=transform_image, task="classification"
    )
    test_dataset_cls = ISIC2017Dataset(
        test_image_dir, test_mask_dir, test_label_csv,
        transform=transform_image, task="classification"
    )

    # Create dataloaders
    train_loader_seg = DataLoader(train_dataset_seg, batch_size=batch_size, shuffle=True)
    val_loader_seg = DataLoader(val_dataset_seg, batch_size=batch_size, shuffle=False)
    test_loader_seg = DataLoader(test_dataset_seg, batch_size=batch_size, shuffle=False)

    train_loader_cls = DataLoader(train_dataset_cls, batch_size=batch_size, shuffle=True)
    val_loader_cls = DataLoader(val_dataset_cls, batch_size=batch_size, shuffle=False)
    test_loader_cls = DataLoader(test_dataset_cls, batch_size=batch_size, shuffle=False)

    return train_loader_seg, val_loader_seg, test_loader_seg, train_loader_cls, val_loader_cls, test_loader_cls

def main():
    # Define transforms
    transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # No normalization for masks
    ])

    # Create dataloaders
    root_dir = "/home/matthewcockayne/Documents/PhD/data/ISIC_2017/"
    train_loader_seg, val_loader_seg, test_loader_seg, train_loader_cls, val_loader_cls, test_loader_cls = get_dataloaders(
        root_dir, transform_image=transform_image, transform_mask=transform_mask, batch_size=16
    )

    # Check the first batch of classification data from the train set
    for images, labels in train_loader_cls:
        print("Train Classification Data:")
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Labels: {labels}")  # Ensure only 0s and 1s for melanoma
        break  # Stop after the first batch
 
    # Check the first batch of segmentation data from the train set
    for images, masks in train_loader_seg:
        print("Train Segmentation Data:")
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        break  # Stop after the first batch
    
    # Check the sizes of the train, val and test sets 
    print("Dataset sizes:")
    print(f"Train Segmentation Dataset: {len(train_loader_seg.dataset)} samples")
    print(f"Validation Segmentation Dataset: {len(val_loader_seg.dataset)} samples")
    print(f"Test Segmentation Dataset: {len(test_loader_seg.dataset)} samples")

    print(f"Train Classification Dataset: {len(train_loader_cls.dataset)} samples")
    print(f"Validation Classification Dataset: {len(val_loader_cls.dataset)} samples")
    print(f"Test Classification Dataset: {len(test_loader_cls.dataset)} samples")

if __name__ == "__main__":
    main()