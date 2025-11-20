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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()  # For binary classification
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return epoch_loss, accuracy


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate metrics
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()  # For binary classification
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return epoch_loss, accuracy

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_val_accuracy = -float("inf")  # Start with negative infinity to ensure improvement on the first epoch.

    def __call__(self, val_accuracy):
        if val_accuracy > self.best_val_accuracy + self.min_delta:
            self.best_val_accuracy = val_accuracy
            self.counter = 0  # Reset counter on improvement
            if self.verbose:
                print(f"Improved validation accuracy: {val_accuracy:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation accuracy for {self.counter} epochs.")
        
        # Check if patience exceeded
        if self.counter >= self.patience:
            if self.verbose:
                print(f"Early stopping triggered after {self.counter} epochs without improvement.")
            return True
        return False
    
# Function to read log.txt and extract loss and accuracy values
def read_log_file(log_file):
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    with open(log_file, "r") as log:
        # Skip header line (first line)
        next(log)

        for line in log:
            # Skip any empty lines
            if not line.strip():
                continue
            
            # Parse each line
            try:
                epoch, train_loss, train_accuracy, val_loss, val_accuracy = line.strip().split(',')
                epochs.append(int(epoch))
                train_losses.append(float(train_loss))
                train_accuracies.append(float(train_accuracy))
                val_losses.append(float(val_loss))
                val_accuracies.append(float(val_accuracy))
            except ValueError as e:
                print(f"Skipping malformed line: {line} ({e})")
                continue  # Skip malformed lines

    return epochs, train_losses, train_accuracies, val_losses, val_accuracies

# Function to plot accuracy and loss curves
def plot_accuracy_loss(epochs, train_losses, train_accuracies, val_losses, val_accuracies, model_name, save_path):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()

    # Save the plot
    plot_filepath = os.path.join(save_path, f"{model_name}_accuracy_loss_plots.png")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    print(f"Plots saved to {plot_filepath}")
    plt.close()

# Train model function with logging (includes early stopping and saving logs)
def train_model(model_name, dataloaders, num_epochs=10, learning_rate=1e-4, device="cuda", save_path="./", patience=10):
    # Load the model
    model = load_model(model_name, num_classes=1)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Track best model
    best_model = None

    # Ensure the save path directory exists
    os.makedirs(save_path, exist_ok=True)

    # Set up the log file (open it in write mode initially)
    log_file = os.path.join(save_path, "log.txt")
    
    # Write header once before training starts
    if not os.path.exists(log_file):  # Check if the log file exists to avoid overwriting
        with open(log_file, "w") as log:
            log.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")  # Write header

    # Now start training
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validate for one epoch
        val_loss, val_accuracy = validate_one_epoch(model, dataloaders['val'], criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Log the metrics to the log file
        with open(log_file, "a") as log:
            log.write(f"{epoch+1},{train_loss:.4f},{train_accuracy:.4f},{val_loss:.4f},{val_accuracy:.4f}\n")

        # Check for early stopping
        if early_stopping(val_accuracy):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Save the best model
        if val_accuracy >= early_stopping.best_val_accuracy:
            early_stopping.best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            model_filepath = os.path.join(save_path, f"{model_name}_best_model.pth")
            torch.save(best_model, model_filepath)
            print(f"Best model saved to {model_filepath} with validation accuracy: {early_stopping.best_val_accuracy:.4f}")

    # Read the log file and plot accuracy and loss
    epochs, train_losses, train_accuracies, val_losses, val_accuracies = read_log_file(log_file)
    plot_accuracy_loss(epochs, train_losses, train_accuracies, val_losses, val_accuracies, model_name, save_path)

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
    # Assuming get_dataloaders function is defined elsewhere in your code
    train_loader_seg, val_loader_seg, test_loader_seg, train_loader_cls, val_loader_cls, test_loader_cls = get_dataloaders(
        root_dir, transform_image=transform_image, transform_mask=transform_mask, batch_size=batch_size
    )
    
    # Create dataloaders dictionary for classification
    dataloaders = {
        "train": train_loader_cls,
        "val": val_loader_cls,
        "test": test_loader_cls
    }

    # Model configuration
    model_names = ['resnet50', 'mobilenet_v2', 'efficientnet_b0', 'ViT', 'swin']
    num_epochs = 200
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patience = 50

    for model_name in model_names:
        print(f"Training {model_name}...")
        # Train the model
        # Lab pc
        #save_path = f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/classification/{model_name}"
        # Cluster
        save_path = f"/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/classification/{model_name}"
        train_model(model_name, dataloaders, num_epochs, learning_rate, device, save_path=save_path, patience=patience)

         # Memory cleanup after each model experiment
        if device == "cuda":
            torch.cuda.empty_cache()  # Free unused GPU memory
        gc.collect()  # Call garbage collection to clear any unused memory in CPU

if __name__ == "__main__":
    main()