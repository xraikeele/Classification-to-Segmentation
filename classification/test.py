import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import torch
import gc
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.load_models import load_model  # Import model-loading 
from data.dataloader import get_dataloaders  # Import dataloaders

def test_model(model_name, dataloader, checkpoint_path, device, use_checkpoint=True):
    """
    Test a model on the test dataset.

    Args:
        model_name (str): Name of the model architecture.
        dataloader (DataLoader): DataLoader for the test dataset.
        checkpoint_path (str): Path to the trained model checkpoint.
        device (str): Device to run the model on ('cpu' or 'cuda').
        use_checkpoint (bool): Whether to load weights from the checkpoint.

    Returns:
        test_loss (float): Average loss on the test set.
        test_accuracy (float): Accuracy on the test set.
    """
    # Load the model
    model = load_model(model_name, num_classes=1)

    # Optionally load the trained weights from checkpoint
    if use_checkpoint and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint for {model_name} from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        if use_checkpoint:
            print(f"Checkpoint not found for {model_name}. Using pretrained weights instead.")

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Testing {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate metrics
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()  # Thresholding sigmoid outputs for binary classification
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / len(dataloader)
    test_accuracy = correct / total
    return test_loss, test_accuracy

def main():
    # Data options
    transform_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # No normalization for masks
    ])
    # Lab PC
    #root_dir = "/home/matthewcockayne/Documents/PhD/data/ISIC_2017/"
    # Cluster
    root_dir = "/home/csb66/MyData/ISIC2017/"
    batch_size = 32

    # Assuming get_dataloaders function is defined elsewhere
    train_loader_seg, val_loader_seg, test_loader_seg, train_loader_cls, val_loader_cls, test_loader_cls = get_dataloaders(
        root_dir, transform_image=transform_image, transform_mask=transform_mask, batch_size=batch_size
    )

    # Test dataloader for classification
    dataloaders = {
        "test": test_loader_cls
    }

    # Model configuration
    model_names = ['resnet50', 'mobilenet_v2', 'efficientnet_b0', 'ViT', 'swin']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in model_names:
        print(f"Testing {model_name}...")

        # Path to the saved checkpoint
        # Lab PC 
        #checkpoint_path = f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/classification/{model_name}/{model_name}_best_model.pth"
        # Cluster
        checkpoint_path = f"/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/classification/{model_name}/{model_name}_best_model.pth"
        # Test with zero-shot (pretrained weights only)
        print(f"\n--- Testing {model_name} (Zero-Shot: Pretrained Weights Only) ---")
        zero_shot_loss, zero_shot_accuracy = test_model(
            model_name, dataloaders['test'], checkpoint_path, device, use_checkpoint=False
        )
        print(f"{model_name} (Zero-Shot) - Test Loss: {zero_shot_loss:.4f}, Test Accuracy: {zero_shot_accuracy:.4f}")

        # Save zero-shot test results
        zero_shot_results_file = os.path.join(
            #f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/classification/{model_name}/",
            f"/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/classification/{model_name}/",
            f"{model_name}_zero_shot_test_results.txt"
        )
        os.makedirs(os.path.dirname(zero_shot_results_file), exist_ok=True)
        with open(zero_shot_results_file, "w") as f:
            f.write(f"Zero-Shot Test Loss: {zero_shot_loss:.4f}\n")
            f.write(f"Zero-Shot Test Accuracy: {zero_shot_accuracy:.4f}\n")
        print(f"Zero-shot test results saved to {zero_shot_results_file}")

        # Test with trained model (checkpoint weights)
        if os.path.exists(checkpoint_path):
            print(f"\n--- Testing {model_name} (Trained Model: From Checkpoint) ---")
            checkpoint_loss, checkpoint_accuracy = test_model(
                model_name, dataloaders['test'], checkpoint_path, device, use_checkpoint=True
            )
            print(f"{model_name} (Checkpoint) - Test Loss: {checkpoint_loss:.4f}, Test Accuracy: {checkpoint_accuracy:.4f}")

            # Save checkpoint test results
            checkpoint_results_file = os.path.join(
                #f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/classification/{model_name}/",
                f"/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/classification/{model_name}/",
                f"{model_name}_checkpoint_test_results.txt"
            )
            with open(checkpoint_results_file, "w") as f:
                f.write(f"Checkpoint Test Loss: {checkpoint_loss:.4f}\n")
                f.write(f"Checkpoint Test Accuracy: {checkpoint_accuracy:.4f}\n")
            print(f"Checkpoint test results saved to {checkpoint_results_file}")
        else:
            print(f"No checkpoint found for {model_name} at {checkpoint_path}. Skipping trained model testing.")

if __name__ == "__main__":
    main()