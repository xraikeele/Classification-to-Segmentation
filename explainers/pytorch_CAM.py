import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import gc
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
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from sklearn.metrics import mean_squared_error
import csv

# Selecting the right target layers for each given model
def layer_selection(model: nn.Module, layer: str):
    if layer == "resnet50":
        return [model.layer4[-1]]
    elif layer == "mobilenet_v2":
        return [model.features[-1]]
    elif layer == "efficientnet_b0":
        return [model.features[-1]]
    elif layer == "ViT":
        #return [model.encoder.layers[-1].self_attention]
        return [model.encoder.layers[-1].ln_1]
    elif layer == "swin":
        return [model.features[-1][1].norm1]
        #return [model.Sequential[-1].SwinTransformerBlock[-1].norm1]
    else:
        raise NotImplementedError(f"Layer selection not implemented for: {layer}")

# Showing the metric on top of the CAM
def visualize_score(visualization, iou_score, name):
    # Display the IoU score
    visualization = cv2.putText(visualization, name, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "IoU Score", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"{iou_score:.5f}", (10, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return visualization
"""
def compute_iou(cam_image, ground_truth_mask, threshold=0.5):
    # Threshold CAM output and ground truth mask to binary
    cam_binary = (cam_image >= threshold).astype(np.uint8)
    gt_binary = (ground_truth_mask >= 0.5).astype(np.uint8)

    intersection = np.sum(np.logical_and(cam_binary, gt_binary))
    union = np.sum(np.logical_or(cam_binary, gt_binary))
    
    return intersection / (union + 1e-6)  # Avoid division by zero
"""
def compute_iou(cam_image, ground_truth_mask, threshold=0.5):
    # Normalize CAM and mask
    cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min())
    ground_truth_mask = (ground_truth_mask - ground_truth_mask.min()) / (ground_truth_mask.max() - ground_truth_mask.min())

    cam_binary = (cam_image >= threshold).astype(np.uint8)
    gt_binary = (ground_truth_mask >= threshold).astype(np.uint8)

    intersection = np.sum(np.logical_and(cam_binary, gt_binary))
    union = np.sum(np.logical_or(cam_binary, gt_binary))

    return intersection / (union + 1e-6)  # Avoid division by zero

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_swin(tensor):
    """
    Reshape tensor for compatibility with Swin Transformer.
    Handles both square and non-square grid sizes.
    """
    #print("Tensor shape before reshape:", tensor.shape)
    
    # Verify the input tensor dimensions
    if len(tensor.shape) == 4:
        batch_size, height, width, num_features = tensor.size()
        # Rearrange to the expected format: (batch_size, num_features, height, width)
        return tensor.permute(0, 3, 1, 2)
    elif len(tensor.shape) == 3:
        batch_size, num_patches, num_features = tensor.size()
        height = width = int(num_patches**0.5)
        if height * width != num_patches:
            raise ValueError("Non-square patch grid is not supported.")
        return tensor.view(batch_size, height, width, num_features).permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

def benchmark(input_tensor, input_image, model, model_type, target_layers, image_idx, ground_truth_mask=None, eigen_smooth=False, aug_smooth=False, log_file=None):
    assert log_file is not None, "log_file must be specified"
    
    # CAM methods
    if model_type in ['ViT']:
        methods = [
            ("GradCAM", GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit)),
            ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit)),
            ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit)),
            ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit, ablation_layer=AblationLayerVit())),
            ("RandomCAM", RandomCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit)),
        ]
    elif model_type in ['swin']:
        methods = [
            ("GradCAM", GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_swin)),
            ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform_swin)),
            ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_swin)),
            ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_swin, ablation_layer=AblationLayerVit())),
            ("RandomCAM", RandomCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_swin)),
        ]
    else:
        methods = [
            ("GradCAM", GradCAM(model=model, target_layers=target_layers)),
            ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers)),
            ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers)),
            ("AblationCAM", AblationCAM(model=model, target_layers=target_layers)),
            ("RandomCAM", RandomCAM(model=model, target_layers=target_layers)),
        ]
    # Predicted class
    model_output = model(input_tensor)
    predicted_category = torch.argmax(model_output, dim=1).item()

    visualizations = []
    
    # Open the CSV file 
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file is empty, write the header row
        if file.tell() == 0:
            writer.writerow(["Image Index", "Method", "IoU Score"])
        
        for name, cam_method in methods:
            try:
                with cam_method:
                    attributions = cam_method(
                        input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(predicted_category)],
                        eigen_smooth=eigen_smooth,
                        aug_smooth=aug_smooth,
                    )

                attribution = attributions[0]

                # Compute IoU if ground_truth_mask is provided
                if ground_truth_mask is not None:
                    score = compute_iou(attribution, ground_truth_mask)
                    print(f"{name} IoU score: {score}")
                else:
                    score = None

                # Log the result to CSV with the correct image index
                if score is not None:
                    writer.writerow([image_idx, name, score])

                
                visualization = show_cam_on_image(input_image, attribution, use_rgb=True)
                visualization = visualize_score(visualization, score, name)
                visualizations.append(visualization)

            except Exception as e:
                print(f"An exception occurred with {name}: {e}")
                continue

    return Image.fromarray(np.hstack(visualizations)) if visualizations else None

def main():
    # Data transform options 
    transform_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_mask = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale masks
    ])
    # Lab PC
    root_dir = "/home/matthewcockayne/Documents/PhD/data/ISIC_2017/"
    # Cluster
    #root_dir = '/home/csb66/MyData/ISIC2017/'
    batch_size = 1

    _, _, test_loader_seg, _, _, test_loader_cls = get_dataloaders(
        root_dir, transform_image=transform_image, transform_mask=transform_mask, batch_size=batch_size
    )

    # Model configuration
    model_configs = [
        {'name': 'resnet50', 'type': 'CNN'},
        {'name': 'ViT', 'type': 'ViT'},
        {'name': 'swin', 'type': 'swin'},
        {'name': 'mobilenet_v2', 'type': 'CNN'},
        {'name': 'efficientnet_b0', 'type': 'CNN'},
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_config in model_configs:
        model_name = model_config['name']
        model_type = model_config['type']
        print(f"Processing model: {model_name}")
        # Path to the saved checkpoint
        # If zero-shot
        #checkpoint_path = None
        # Else if using ISIC-trained use path
        checkpoint_path = f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/classification/{model_name}/{model_name}_best_model.pth"
        # Load model
        model = load_model(model_name)
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint for {model_name} from {checkpoint_path}...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Checkpoint not found for {model_name}. Using pretrained weights instead.")
        model.to(device)
        model.eval()

        # Select target layers
        target_layers = layer_selection(model, model_name)
        
        # Create directory for saving results
        # Lab PC
        save_dir = f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/CAM/{model_name}"
        # Cluster
        #save_dir = f"/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/CAM-zero-shot/{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Create log_file path
        log_file = os.path.join(save_dir, f"iou_scores_{model_name}.csv")

        image_count = 0  # Initialise image count
        
        # Process batches from both test loaders simultaneously
        for batch_idx, ((image_cls, labels_cls), (image_seg, masks_seg)) in enumerate(zip(test_loader_cls, test_loader_seg)):
            # image_cls is from classification loader (no mask), image_seg is from segmentation loader
            # labels_cls are class labels, masks_seg are ground truth segmentation masks

            # Prepare input tensor and image (from classification loader)
            input_tensor = image_cls.to(device)
            input_image = input_tensor[0].cpu().permute(1, 2, 0).numpy() 
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min()) 

            # Ground truth mask (from segmentation loader)
            ground_truth_mask = masks_seg[0].cpu().numpy() 

            # Benchmark CAM methods with IoU evaluation
            output_image = benchmark(
                input_tensor, 
                input_image, 
                model,
                model_type, 
                target_layers, 
                image_idx=batch_idx, 
                ground_truth_mask=ground_truth_mask, 
                log_file=log_file  
            )

            # Save output images for the first 50 
            if output_image is not None and image_count < 50:
                output_path = os.path.join(save_dir, f"{model_name}_CAM_{batch_idx}.png")
                output_image.save(output_path)
                print(f"Saved CAM to {output_path}")
                image_count += 1

            # Explicitly delete items and run garbage collection
            del input_tensor, input_image, ground_truth_mask, output_image
            gc.collect()
                
if __name__ == "__main__":
    main()