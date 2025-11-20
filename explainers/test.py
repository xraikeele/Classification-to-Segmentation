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
import requests
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.metrics.road import ROADCombined, ROADMostRelevantFirst
from sklearn.metrics import mean_squared_error

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
        raise NotImplementedError(f"Layer selection not implemented for: {layer}")
    
# Showing the metrics on top of the CAM : 
def visualize_score(visualization, score, name, percentiles):
    visualization = cv2.putText(visualization, name, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "(Least first - Most first)/2", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Percentiles: {percentiles}", (10, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    visualization = cv2.putText(visualization, "Remove and Debias", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA) 
    visualization = cv2.putText(visualization, f"{score:.5f}", (10, 85), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    return visualization


def benchmark(input_tensor, input_image, model, target_layers, eigen_smooth=False, aug_smooth=False):
    # Define the CAM methods
    methods = [
        ("GradCAM", GradCAM(model=model, target_layers=target_layers)),
        ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers)),
        ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers)),
        ("AblationCAM", AblationCAM(model=model, target_layers=target_layers)),
        ("RandomCAM", RandomCAM(model=model, target_layers=target_layers)),
    ]
    
    # Get model's predicted category
    model_output = model(input_tensor)
    predicted_category = torch.argmax(model_output, dim=1).item()  # Get predicted class (0 or 1)
    
    # Initialize metrics
    cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    # Create a target for the predicted class
    targets = [ClassifierOutputTarget(predicted_category)]
    metric_targets = [ClassifierOutputSoftmaxTarget(predicted_category)]

    visualizations = []
    percentiles = [10, 50, 90]

    for name, cam_method in methods:
        try:
            with cam_method:
                attributions = cam_method(
                    input_tensor=input_tensor,
                    targets=targets,
                    eigen_smooth=eigen_smooth,
                    aug_smooth=aug_smooth,
                )
            print(f"{name} attributions shape: {attributions.shape}")  # Debugging

            # Validate and extract attribution
            if attributions is None or attributions.ndim not in [3, 4]:
                raise ValueError(f"Invalid attributions returned by {name}: {attributions}")

            attribution = attributions[0]  # Correct indexing for 3D output
            print(f"{name} single attribution shape: {attribution.shape}")  # Debugging
            print("Attribution stats:", attribution.min(), attribution.max(), attribution.mean())
            print("Input tensor stats:", input_tensor.min(), input_tensor.max())
            print("Confidence before masking:", model(input_tensor)[0, predicted_category].item())

            # Continue with visualization and scoring
            scores = cam_metric(input_tensor, attributions, metric_targets, model)
            print(scores)
            score = scores[0]
            print(score)
            visualization = show_cam_on_image(input_image, attribution, use_rgb=True)
            visualization = visualize_score(visualization, score, name, percentiles)
            visualizations.append(visualization)
        except Exception as e:
            print(f"An exception occurred in CAM with {name}: {type(e)}. Message: {str(e)}")
            continue

    return Image.fromarray(np.hstack(visualizations)) if visualizations else None

def main():
    # Data options
    transform_image = transforms.Compose([  # Adjust image transformations
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    root_dir = "/home/matthewcockayne/Documents/PhD/data/ISIC_2017/"
    batch_size = 1  # Process one image at a time for CAM generation

    _, _, _, _, _, test_loader_cls = get_dataloaders(
        root_dir, transform_image=transform_image, transform_mask=None, batch_size=batch_size
    )

    # Model configuration
    model_names = ['resnet50', 'ViT']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        # Load model
        model = load_model(model_name)
        model.to(device)
        model.eval()

        # Select target layers
        target_layers = layer_selection(model, model_name)

        # Create directory for saving results
        save_dir = f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/CAM_zero-shot/{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        for i, (images, labels) in enumerate(test_loader_cls):
            # Debug: Print shape of input tensor before transformations
            print(f"Input image shape before any transformations: {images.shape}")

            # Prepare input tensor and image
            input_tensor = images.to(device)  # Tensor is already in the correct shape: (batch_size, channels, height, width)
            
            # Debug: Print shape of input tensor after it's moved to device
            print(f"Input tensor shape after moving to device: {input_tensor.shape}")

            # **Do not permute `input_tensor`**
            input_image = input_tensor[0].cpu().permute(1, 2, 0).numpy()  # Permute for visualization (height, width, channels)
            
            # Debug: Print shape of input_image after permuting
            print(f"Shape of input_image after permute (for visualization): {input_image.shape}")
            
            # Normalize for visualization
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # Normalize image

            # Debug: Print shape of input_image after normalization
            print(f"Shape of input_image after normalization: {input_image.shape}")

            # Benchmark CAM methods
            output_image = benchmark(input_tensor, input_image, model, target_layers, eigen_smooth=False, aug_smooth=False)

            if output_image is not None:
                # Save with unique filename
                output_path = os.path.join(save_dir, f"{model_name}_CAM_{i}.png")
                output_image.save(output_path)
                print(f"Saved CAM to {output_path}")

if __name__ == "__main__":
    main()