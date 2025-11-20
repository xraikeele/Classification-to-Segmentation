import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from glob import glob
import numpy as np
import random
import cv2
from PIL import Image
import json
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, normalize, resize
from models.load_models import load_model 
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.metrics.road import ROADCombined, ROADMostRelevantFirst
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from models.load_models import load_model  
from data.dataloader import get_dataloaders 
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def set_seed(seed=905):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def calculate_metrics(predicted, ground_truth):
    #print("Predicted mask shape:", predicted.shape)
    #print("Ground truth shape:", ground_truth.shape)
    ground_truth = ground_truth.squeeze(0)  # Shape becomes (1936, 2592)
    # Resize ground truth to match predicted mask
    ground_truth = cv2.resize(ground_truth, (224, 224))
    #print("Ground truth shape:", ground_truth.shape)

    predicted = (predicted > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)
    precision = precision_score(ground_truth.flatten(), predicted.flatten(), zero_division=0)
    recall = recall_score(ground_truth.flatten(), predicted.flatten(), zero_division=0)
    f1 = f1_score(ground_truth.flatten(), predicted.flatten(), zero_division=0)
    iou = jaccard_score(ground_truth.flatten(), predicted.flatten(), zero_division=0)
    return {"Precision": precision, "Recall": recall, "F1": f1, "IoU": iou}

def post_process_mask(mask):
    mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure correct input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def transform_mask(mask_path):
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    mask = Image.open(mask_path).convert("L")
    return np.array(transform(mask))

# Sample points based on heatmap
def sample_points(heatmap, num_points):
    height, width = heatmap.shape
    flat_heatmap = heatmap.flatten()

    # Avoid division by zero by adding a small epsilon
    flat_heatmap = np.power(flat_heatmap, 30)  # Amplify high values
    if flat_heatmap.sum() == 0 or np.isnan(flat_heatmap).any():  
        print("Warning: Heatmap contains only zeros or NaNs, returning random points.")
        return np.random.randint(0, width, size=(num_points, 2))  # Fallback: return random points

    probabilities = flat_heatmap / flat_heatmap.sum()  # Normalize safely

    indices = np.arange(len(flat_heatmap))
    sampled_indices = np.random.choice(indices, size=num_points, replace=False, p=probabilities)

    sampled_coords = [(index % width, index // width) for index in sampled_indices]
    return np.array(sampled_coords)

def process_image(img_path, model, cam_extractor):
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    # Ensure the input tensor is on the same device as the model
    device = next(model.parameters()).device  
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_tensor)

    predicted_category = out.argmax().item()
    activation_map = cam_extractor(input_tensor, targets=[ClassifierOutputTarget(predicted_category)])
    # Debug
    """
    print("Activation map type:", type(activation_map))
    print("Activation map length:", len(activation_map))
    if len(activation_map) > 0:
        print("First activation map shape:", activation_map[0].shape)
    plt.imshow(activation_map[0], cmap='viridis')  # Choose a colormap
    plt.colorbar()
    plt.axis("off")  # Hide axes
    plt.savefig("activation_map.png", bbox_inches='tight', pad_inches=0)  # Save as PNG
    plt.close()  # Close the figure to free memory
    """
    heatmap = activation_map[0]
    heatmap = np.clip(heatmap, 0, None)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))

    return heatmap_resized, img

# Post-process mask with morphological operations
def post_process_mask(mask):
    mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Evaluate CAM-SAM
def evaluate_model_with_cam_sam(test_image_folder, gt_folder, output_folder, num_points=3, weight_exponent=0.4, sam_checkpoint="sam_vit_h_4b8939.pth", model=None, model_type=None, target_layers=None, device="cuda"):
    os.makedirs(output_folder, exist_ok=True)
    
    # Define CAM methods based on model type
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

    sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    total_metrics = {"Precision": 0, "Recall": 0, "F1": 0, "IoU": 0}
    num_images = 0

    results = {method_name: [] for method_name, _ in methods}  # Initialize results ONCE

    for image_name in os.listdir(test_image_folder):
        if not image_name.endswith('.jpg'):
            continue

        set_seed()
        img_path = os.path.join(test_image_folder, image_name)
        gt_image_name = image_name.replace('.jpg', '_segmentation.png')
        gt_path = os.path.join(gt_folder, gt_image_name)
        
        if not os.path.exists(gt_path):
            print(f"Ground truth not found for {image_name}, skipping...")
            continue
        
        ground_truth = transform_mask(gt_path)
        transformed_image = transform_image(img_path)
        transformed_image = transformed_image.permute(1, 2, 0)
        
        img = cv2.imread(img_path)  # Load image once

        for method_name, cam_method in methods:  # Iterate over methods
            # Generate heatmap for this specific CAM method
            heatmap_resized, original_img = process_image(img_path, model, cam_method)
            # Handle NaN or zero-value heatmaps safely
            if heatmap_resized is None or np.isnan(heatmap_resized).any() or heatmap_resized.max() == heatmap_resized.min():
                print(f"Warning: Skipping {image_name} due to invalid heatmap from {method_name}")
                continue  # Skip processing this image if heatmap is invalid
            # Sample points based on method-specific heatmap
            sampled_points = sample_points(heatmap_resized, num_points)
            input_points = np.array(sampled_points)
            input_labels = np.ones(input_points.shape[0]) 

            # Predict segmentation mask
            predictor.set_image(transformed_image.numpy())
            predicted_masks, scores, _ = predictor.predict(
                point_coords=input_points, 
                point_labels=input_labels, 
                multimask_output=True
            )
            best_mask_index = np.argmax(scores)  # Get the mask with highest confidence score
            predicted_mask = predicted_masks[best_mask_index]
            #print('predicted_mask shape', predicted_mask.shape)
            predicted_mask = post_process_mask(predicted_mask)

            # Compute metrics
            metrics = calculate_metrics(predicted_mask, ground_truth)
            results[method_name].append({"image": image_name, "metrics": metrics})  

            # Save output image
            method_folder = os.path.join(output_folder, method_name)
            os.makedirs(method_folder, exist_ok=True)
            output_image_path = os.path.join(method_folder, f"{image_name}_result.png")
            cv2.imwrite(output_image_path, predicted_mask)

            # Accumulate metrics for averaging
            for key in total_metrics:
                total_metrics[key] += metrics[key]

        num_images += 1

    # Save results as JSON
    for method_name, _ in methods:
        method_results_path = os.path.join(output_folder, method_name, "results.json")
        with open(method_results_path, "w") as f:
            json.dump(results[method_name], f, indent=4)

    # Compute and save average metrics
    average_metrics = {}
    for method_name in results:
        method_total_metrics = {"Precision": 0, "Recall": 0, "F1": 0, "IoU": 0}
        method_count = len(results[method_name])  

        if method_count > 0:
            for entry in results[method_name]:
                for key in method_total_metrics:
                    method_total_metrics[key] += entry["metrics"][key]

            average_metrics[method_name] = {key: method_total_metrics[key] / method_count for key in method_total_metrics}
        else:
            average_metrics[method_name] = method_total_metrics  

    average_results_path = os.path.join(output_folder, "average_results.json")
    with open(average_results_path, "w") as f:
        json.dump(average_metrics, f, indent=4)

    return results, average_metrics

def main():
    set_seed()
    # Lab-pc
    root_dir = "/home/matthewcockayne/Documents/PhD/data/ISIC_2017/"
    test_image_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Data/")
    gt_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Part1_GroundTruth")
    output_folder = "/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/CAM-MedSAM/"
    # cluster
    #root_dir = "/home/xrai/datasets/ISIC2017/"
    #test_image_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Data/")
    #gt_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Part1_GroundTruth")
    #output_folder = "/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/CAM-MedSAM/"

    os.makedirs(output_folder, exist_ok=True)
    
    sam_model = "vit_b"  
    sam_checkpoint = "./results/model_checkpoints/medsam_vit_b.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint)  
    #sam.to(device)
    #sam_predictor = SamPredictor(sam)
    # Model configuration
    model_configs = [
        {'name': 'ViT', 'type': 'ViT'},
        {'name': 'swin', 'type': 'swin'},
        {'name': 'resnet50', 'type': 'CNN'},
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
        # lab-pc
        #checkpoint_path = f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/classification/{model_name}/{model_name}_best_model.pth"
        # cluster
        checkpoint_path = f"/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/classification/{model_name}/{model_name}_best_model.pth"
        # Load model
        model = load_model(model_name)
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint for {model_name} from {checkpoint_path}...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Checkpoint not found for {model_name}. Using pretrained weights instead.")
        model.to(device)
        model.eval()

        print(f"Testing {model_name}...")
        
        model_output_folder = os.path.join(output_folder, model_name)
        os.makedirs(model_output_folder, exist_ok=True)
    
        target_layers = layer_selection(model, model_name)

        results, avg_metrics = evaluate_model_with_cam_sam(
            test_image_folder, gt_folder, model_output_folder,
            num_points=10, weight_exponent=0.4, sam_checkpoint=sam_checkpoint,
            model=model, model_type=model_type, target_layers=target_layers, device=device
        )

if __name__ == "__main__":
    main()
