import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from glob import glob
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataloader import get_dataloaders  # Import dataloaders
from torchvision import transforms
import timm
from models.nest import nest_base_jx

class NesTGradCAT(nn.Module):
    def __init__(self, num_classes=2):
        super(NesTGradCAT, self).__init__()
        self.model = nest_base_jx(pretrained=True)
        
        # Modify classification head
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        print(self.model)
        # Store activations for GradCAT
        self.activations = {}

        # Hook function to capture intermediate feature maps
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output
                output.retain_grad()  # Ensure gradients are retained
            return hook
        
        # Register hooks on layers that we want to extract
        self.model.levels[0].transformer_encoder[0].register_forward_hook(get_activation('level1'))
        self.model.levels[1].transformer_encoder[0].register_forward_hook(get_activation('level2'))
        self.model.levels[2].transformer_encoder[0].register_forward_hook(get_activation('level3'))

    def forward(self, x):
        return self.model(x)

    def get_feature_maps(self):
        """ Returns stored feature maps from hooks. """
        return self.activations

    def get_feature_map_shapes(self):
        """ Returns the correct shapes of the feature maps stored in hooks. """
        feature_map_shapes = {}
        for name, activation in self.activations.items():
            if len(activation.shape) == 4:  # Ensure (batch, channels, height, width)
                batch, channels, height, width = activation.shape
            else:  # If (channels, height, width) only
                channels, height, width = activation.shape
            
            feature_map_shapes[name] = (channels, height, width)

        return feature_map_shapes

def gradcat(model, input_image, target_class=1):
    """
    Implements the GradCAT traversal algorithm.
    Args:
        model (nn.Module): The trained model.
        input_image (Tensor): Input image tensor.
        target_class (int): Target class for gradient computation.
    Returns:
        List of indices representing the traversal path.
    """
    model.eval()
    input_image = input_image.requires_grad_(True)  # Ensure input image tracks gradients

    # Forward pass to get model output
    output = model(input_image)
    
    # Ensure output is a scalar (select the logit corresponding to target_class)
    target_logit = output[:, target_class] if output.shape[1] > 1 else output.squeeze()
    
    # Compute gradient w.r.t. target logit
    model.zero_grad()
    target_logit.backward(retain_graph=True)

    path = []
    alpha_l = input_image  # Initialize with input

    for l in range(3, 0, -1):  # Assuming 3 levels
        if f'level{l}' in model.activations:
            alpha_l = model.activations[f'level{l}']
            
            # Check if alpha_l requires gradients and retain gradients if needed
            if not alpha_l.requires_grad:
                alpha_l.requires_grad = True  # Ensure alpha_l tracks gradients
            alpha_l.retain_grad()  # Ensure gradients are kept
            
            grad = alpha_l.grad  # Retrieve gradient
            
            if grad is None:
                print(f"Warning: Gradient for level{l} is None!")
                continue  # Skip this level

            # Calculate the correct height and width from the flattened shape
            if len(alpha_l.shape) == 4:  # Ensure (batch, channels, height, width)
                batch_size, channels, h, w = alpha_l.shape
                print(f"alpha_l original shape: {alpha_l.shape}")
                alpha_l = alpha_l.view(batch_size * channels, h, w)
                print(f"alpha_l.view shape: {alpha_l.shape}")
                print(f"channels: {channels}, height: {h}, width: {w}")

            grad = grad.view(channels, h, w)

            h_l = alpha_l * (-grad)  # Element-wise multiplication
            h_l_pooled = torch.nn.functional.adaptive_avg_pool2d(h_l, (2, 2))
            n_star_l = torch.argmax(h_l_pooled).item()
            path.append(n_star_l)

    return path

def visualize_traversal_path(image_path, traversal_path, save_path, feature_map_sizes):
    """
    Visualizes the traversal path on the input image and saves it.
    Args:
        image_path (str): Path to the input image.
        traversal_path (List[int]): The traversal path from GradCAT.
        save_path (str): Path to save the output image.
        feature_map_sizes (List[tuple]): List of feature map sizes (channels, height, width) for each level.
    """
    # Load and prepare the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = image.shape

    # Overlay the traversal path on the image
    for idx, (n_star, feature_map_size) in enumerate(zip(traversal_path, feature_map_sizes)):
        channels, f_height, f_width = feature_map_size  # Correct unpacking

        # Compute correct scaling factors
        scale_x = img_width / f_width  # Scale width
        scale_y = img_height / f_height  # Scale height

        # Compute row and col correctly in feature map space
        row = n_star // f_width
        col = n_star % f_width

        # Map feature map coordinates to image space
        start_x = int(col * scale_x)
        start_y = int(row * scale_y)
        end_x = int((col + 1) * scale_x)
        end_y = int((row + 1) * scale_y)

        # Draw rectangle
        color = (255, 0, 0)  # Red
        thickness = max(2, int(scale_x // 10))  # Adaptive thickness
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, thickness)

        # Add text
        text = f"Level {len(traversal_path) - idx}"
        cv2.putText(image, text, (start_x, max(20, start_y - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save and display
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    # Load model
    nest_model = NesTGradCAT(num_classes=2)

    # Load an image and preprocess
    image_path = "/home/matthewcockayne/Documents/PhD/data/ISIC_2017/test/ISIC-2017_Test_v2_Data/ISIC_0012199.jpg"
    save_path = "/home/matthewcockayne/Documents/PhD/Zero-Shot_SkinLesion_Segmentation/NesT-SAM/ISIC_0012199_traversal_path.png"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_image = transform(Image.open(image_path)).unsqueeze(0)

    # Perform a forward pass to store activations
    nest_model(input_image)

    # Compute GradCAT traversal path for class 1
    path = gradcat(nest_model, input_image, target_class=1)
    print("GradCAT Traversal Path:", path)
    
    # Retrieve feature map sizes from the model
    feature_map_shapes = nest_model.get_feature_map_shapes()
    feature_map_sizes = [feature_map_shapes[f'level{l}'] for l in range(3, 0, -1)]
    print("Feature Map Sizes:", feature_map_sizes)

    # Visualize the traversal path on the input image and save it
    visualize_traversal_path(image_path, path, save_path, feature_map_sizes)

if __name__ == "__main__":
    main()

