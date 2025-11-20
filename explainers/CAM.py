import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import normalize, resize, to_pil_image
from models.load_models import load_model
from data.dataloader import get_dataloaders
from tqdm import tqdm
from PIL import Image
import cv2

def layer_selection(model: nn.Module, layer: str):

    if layer == "resnet50":
        return [model.layer4[-1]]        
    elif layer == "mobilenet_v2":
        return [model.features[-1]]
    elif layer == "efficientnet_b0":
        return [model.features[-1]]
    elif layer == "ViT":
        return [model.encoder.layers[-1].self_attention]
    elif layer == "swin":
        return [model.features[-1][1].attn]
    else:
        raise NotImplementedError()
    
def register_hooks(model, model_name):
    """
    Register hooks to capture feature maps and gradients using the appropriate layer
    selected dynamically based on the model_name.
    """
    feature_maps = []
    gradients = []

    # Dynamically select the target layer(s) using layer_selection
    target_layers = layer_selection(model, model_name)
    
    for target_layer in target_layers:
        def forward_hook(module, input, output):
            #print(f"Forward Hook Output Shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
            feature_maps.append(output)

        def backward_hook(module, grad_input, grad_output):
            #print(f"Backward Hook Output Shape: {grad_output[0].shape if isinstance(grad_output[0], torch.Tensor) else type(grad_output[0])}")
            gradients.append(grad_output[0])

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    return feature_maps, gradients

def generate_cam(feature_map, gradient, model_type):
    """
    Generate a CAM heatmap from the feature map and gradients.
    """
    # Handle tuple inputs (which can occur with some hook outputs)
    if isinstance(feature_map, tuple):
        feature_map = feature_map[0]
    if isinstance(gradient, tuple):
        gradient = gradient[0]

    # Special handling for Vision Transformers (ViT) and Swin models
    #if model_type in ["ViT", "swin"]:
    if model_type in ["ViT"]:    
        # Exclude the class token (the first token)
        feature_map = feature_map[:, 1:, :]  # Shape: [batch_size, num_patches, embedding_dim]
        gradient = gradient[:, 1:, :]  # Shape: [batch_size, num_patches, embedding_dim]

        # Compute grid dimensions
        batch_size, num_patches, embedding_dim = gradient.shape
        grid_size = int(num_patches**0.5)  # Assuming a square grid (e.g., 14x14)

        # Reshape feature map and gradient to 2D grid for visualization
        feature_map = feature_map.view(batch_size, grid_size, grid_size, embedding_dim).permute(0, 3, 1, 2)
        gradient = gradient.view(batch_size, grid_size, grid_size, embedding_dim).permute(0, 3, 1, 2)
    elif feature_map.dim() == 4 and gradient.dim() == 4:
        # For CNNs, we expect 4-dimensional tensors
        pass  # No special handling required for CNNs
    else:
        raise ValueError(f"Feature map and gradient must be 4-dimensional for CNNs, "
                         f"but got feature_map {feature_map.shape}, gradient {gradient.shape}.")

    # Perform global average pooling on the gradients
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)  # Shape: [batch_size, channels, 1, 1]

    # Compute the weighted sum of the feature maps
    cam = torch.sum(weights * feature_map, dim=1).squeeze(0)  # Weighted combination of channels

    # Apply ReLU to the CAM to remove negative values
    cam = torch.relu(cam)

    # Normalize the CAM
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:  # Avoid division by zero
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)  # Set CAM to zeros if min and max are the same

    # Convert to numpy for visualization
    cam = cam.detach().cpu().numpy()

    # Replace invalid values with zeros
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)

    return cam

def overlay_cam_on_image(cam, image, alpha=0.5):
    """
    Overlay CAM heatmap on the original image.
    """
    #cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (image.size[0], image.size[1]))  # Resize to match image dimensions
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # Apply color map
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB
    overlay = cv2.addWeighted(np.array(image), alpha, heatmap, 1 - alpha, 0)  # Blend images
    return Image.fromarray(overlay)

def generate_and_save_cam(model_name, model_type, dataloader, checkpoint_path, save_dir, device):
    """
    Generate and save CAM heatmaps for test data.
    """
    # Load model and optionally load checkpoint
    model = load_model(model_name, num_classes=1)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.train()  # Enable gradient tracking for CAM generation

    # Register hooks to capture feature maps and gradients
    feature_maps, gradients = register_hooks(model, model_name)

    os.makedirs(save_dir, exist_ok=True)

    for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Generating CAMs")):
        inputs, labels = inputs.to(device), labels.to(device)

        # Enable gradient tracking for inputs
        inputs.requires_grad_()

        # Forward pass
        outputs = model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)

        # Backward pass to compute gradients
        model.zero_grad()
        loss.backward()

        # Extract the feature map and gradients for CAM computation
        feature_map = feature_maps[-1]
        gradient = gradients[-1]

        # Generate CAM
        cam = generate_cam(feature_map, gradient, model_type)

        # Convert input image tensor to PIL image
        input_image = to_pil_image(inputs[0].cpu())

        # Overlay CAM on the input image
        overlayed_image = overlay_cam_on_image(cam, input_image)

        # Save the heatmap
        overlay_path = os.path.join(save_dir, f"cam_{i}.png")
        overlayed_image.save(overlay_path)

        # Use tqdm.write instead of print
        #tqdm.write(f"Saved CAM heatmap to {overlay_path}")

    # Restore the model to evaluation mode
    model.eval()

def main():
    # Data options
    transform_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    root_dir = "/home/matthewcockayne/Documents/PhD/data/ISIC_2017/"
    batch_size = 1  # Process one image at a time for CAM generation

    # Assuming get_dataloaders function is defined elsewhere
    _, _, test_loader_seg, _, _, test_loader_cls = get_dataloaders(
        root_dir, transform_image=transform_image, transform_mask=None, batch_size=batch_size
    )

    # Test dataloader for classification
    dataloaders = {
        "test": test_loader_cls
    }

    # Model configuration
    model_configs = [
        {'name': 'swin', 'type': 'swin'},
        {'name': 'ViT', 'type': 'ViT'},
        {'name': 'resnet50', 'type': 'CNN'},
        {'name': 'mobilenet_v2', 'type': 'CNN'},
        {'name': 'efficientnet_b0', 'type': 'CNN'},
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_config in model_configs:
        model_name = model_config['name']
        model_type = model_config['type']
        
        print(f"Generating CAMs for {model_name} of type {model_type}...")  # Debugging print

        # Path to the saved checkpoint
        checkpoint_path = None  # Replace with actual checkpoint path if needed
        save_dir = f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/CAM_zero-shot/{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Generate and save CAMs
        generate_and_save_cam(model_name, model_type, dataloaders['test'], checkpoint_path, save_dir, device)

if __name__ == "__main__":
    main()