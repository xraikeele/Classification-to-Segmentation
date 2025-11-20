import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from glob import glob
import gc
import numpy as np
import random
import cv2
from PIL import Image
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def layer_selection(model: nn.Module, layer: str):
    if layer == "resnet50":
        return [model.layer4[-1]]
    elif layer == "mobilenet_v2":
        return [model.features[-1]]
    elif layer == "efficientnet_b0":
        return [model.features[-1]]
    elif layer == "ViT":
        #for name, module in model.encoder.layers[-1].named_children():
        #    print(f"{name}: {module}")  
        return [model.encoder.layers[-1].ln_1]
    elif layer == "swin":
        return [model.features[-1][1].norm1]
    else:
        raise NotImplementedError(f"Layer selection not implemented for: {layer}")

def reshape_transform_vit(tensor, height=14, width=14):
    """
    Reshape tensor for compatibility with Vision Transformer (ViT).
    Args:
        tensor (torch.Tensor): The input tensor of shape [batch_size, num_patches, embedding_dim].
        height (int): The height of the spatial grid.
        width (int): The width of the spatial grid.
    Returns:
        torch.Tensor: The reshaped tensor of shape [batch_size, embedding_dim, height, width].
    """
    # Exclude the class token if present
    tensor = tensor[:, 1:, :]  # Remove the class token (if applicable)

    # Reshape to [batch_size, height, width, embedding_dim]
    batch_size, num_patches, embedding_dim = tensor.size()
    reshaped = tensor.view(batch_size, height, width, embedding_dim)

    # Permute to [batch_size, embedding_dim, height, width]
    reshaped = reshaped.permute(0, 3, 1, 2)
    print(f"Original tensor shape: {tensor.shape}")
    print(f"Reshaped tensor shape: {reshaped.shape}")
    return reshaped

def reshape_transform_swin(tensor, feature_map_size=None):
    """
    Reshape tensor for compatibility with Swin Transformer.
    Handles both square and non-square grid sizes.
    Args:
        tensor (torch.Tensor): The input tensor of shape [batch_size, num_patches, embedding_dim].
        feature_map_size (tuple, optional): The (height, width) of the feature map.
            If not provided, it will be inferred assuming a square grid.
    Returns:
        torch.Tensor: The reshaped tensor of shape [batch_size, embedding_dim, height, width].
    """
    print(f"Original tensor shape: {tensor.shape}")
    if len(tensor.shape) == 3:  # [batch_size, num_patches, embedding_dim]
        batch_size, num_patches, embedding_dim = tensor.size()

        # Infer height and width if not provided
        if feature_map_size is None:
            height = width = int(num_patches**0.5)
            if height * width != num_patches:
                raise ValueError(
                    f"Non-square patch grid detected. Provide feature_map_size explicitly. "
                    f"num_patches={num_patches}, inferred grid={height}x{width}."
                )
        else:
            height, width = feature_map_size

        # Reshape to [batch_size, height, width, embedding_dim]
        reshaped = tensor.view(batch_size, height, width, embedding_dim)
        print(f"Reshaped tensor shape: {reshaped.shape}")
        # Permute to [batch_size, embedding_dim, height, width]
        reshaped = reshaped.permute(0, 3, 1, 2)
        return reshaped

    elif len(tensor.shape) == 4:  # [batch_size, height, width, embedding_dim]
        return tensor.permute(0, 3, 1, 2)  # [batch_size, embedding_dim, height, width]

    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

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

def calculate_metrics(predicted, ground_truth):
    print("Predicted mask shape:", predicted.shape)
    print("Ground truth shape:", ground_truth.shape)

    # Ensure both masks are binary
    predicted = (predicted > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)

    # Flatten both arrays for metric calculation
    predicted_flat = predicted.flatten()
    ground_truth_flat = ground_truth.flatten()

    # Calculate metrics
    precision = precision_score(ground_truth_flat, predicted_flat, zero_division=0)
    recall = recall_score(ground_truth_flat, predicted_flat, zero_division=0)
    f1 = f1_score(ground_truth_flat, predicted_flat, zero_division=0)
    iou = jaccard_score(ground_truth_flat, predicted_flat, zero_division=0)

    return {"Precision": precision, "Recall": recall, "F1": f1, "IoU": iou}

def post_process_mask(mask, ground_truth, output_path_contours, image_name):
    mask = (mask * 255).astype(np.uint8)
    print("Mask shape:", mask.shape)

    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)  # Use a smaller kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Debug: Save the binary mask before contours calculation
    mask_before_contours_path = os.path.join(output_path_contours, f"{image_name}_mask_before_contours.png")
    cv2.imwrite(mask_before_contours_path, mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours found: {len(contours)}")

    # Handle empty contours
    if not contours:
        print("No contours found. Returning the original binary mask.")
        return mask

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a new mask to keep only the largest contour
    largest_contour_mask = np.zeros_like(mask)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Debug: Save the mask with the largest contour
    largest_contour_path = os.path.join(output_path_contours, f"{image_name}_largest_contours.png")
    cv2.imwrite(largest_contour_path, largest_contour_mask)
    print(f"Largest contour mask saved to {largest_contour_path}")

    return largest_contour_mask

def process_image(img_path, model, cam_extractor):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_tensor)

    predicted_category = out.argmax().item()
    print(f"input_tensor shape: {input_tensor.shape}")
    activation_map = cam_extractor(input_tensor, targets=[ClassifierOutputTarget(predicted_category)])
    if activation_map is None:
        print("Activation map is None.")
        return None, None

    heatmap = activation_map[0]
    print(f"Activation map shape: {activation_map.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    heatmap = np.clip(heatmap, 0, None)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))

    return heatmap_resized, img

def sample_points(heatmap, num_points, amplification_factor=30):
    height, width = heatmap.shape
    flat_heatmap = heatmap.flatten()

    # Amplify high values but avoid numerical instability
    flat_heatmap = np.power(flat_heatmap, amplification_factor)
    flat_heatmap += 1e-10  # Avoid division by zero

    # Get valid indices (nonzero values)
    valid_indices = np.nonzero(flat_heatmap)[0]
    if len(valid_indices) == 0:
        print("Warning: No valid heatmap points found. Returning random points.")
        return np.random.randint(0, width, size=(num_points, 2))

    # Compute probabilities safely
    probabilities = flat_heatmap[valid_indices] / flat_heatmap[valid_indices].sum()

    # Sample points based on heatmap probabilities
    sampled_indices = np.random.choice(valid_indices, size=min(num_points, len(valid_indices)), replace=False, p=probabilities)

    # Convert to (x, y) coordinates efficiently
    sampled_coords = np.column_stack((sampled_indices % width, sampled_indices // width))

    return sampled_coords

def draw_bounding_box(image, box, sampled_points, output_path=None):
    """
    Draws a bounding box and sampled points on the given image.

    Args:
        image (numpy.ndarray): The image on which to draw the bounding box (HxWxC).
        box (numpy.ndarray): The bounding box in format [[x_min, y_min, x_max, y_max]].
        sampled_points (numpy.ndarray): Sampled points as (N, 2) array with (x, y) coordinates.
        output_path (str, optional): Path to save the image. If None, it is displayed.
    """
    # Ensure the image is in RGB format
    if len(image.shape) == 2:  # Convert grayscale image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert bounding box coordinates to integers
    x_min, y_min, x_max, y_max = map(int, box[0])

    # Draw bounding box (Green)
    image_with_overlay = cv2.rectangle(image.copy(), (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw sampled points (Red circles)
    for (x, y) in sampled_points:
        x, y = int(x), int(y)  # Ensure integer coordinates
        cv2.circle(image_with_overlay, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red dot

    # Save or display the image
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image_with_overlay)
        print(f"Image with bounding box and points saved to {output_path}")
    else:
        cv2.imshow("Bounding Box with Sampled Points", image_with_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def calculate_iou(pred_mask, gt_mask):
    intersection = np.sum((pred_mask == 1) & (gt_mask == 1))
    union = np.sum((pred_mask == 1) | (gt_mask == 1))
    return intersection / union if union != 0 else 0

def medsam_inference(sam_model, img_embed, box_1024, input_points, ground_truth, resized_image_np, num_masks=10):
    print("### DEBUG: Inside medsam_inference ###")

    # Prepare for storing multiple masks and IoU values
    all_masks = []
    iou_scores = []

    # Debug: Check input shapes
    print(f"Image embedding shape: {img_embed.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Bounding box shape: {box_1024.shape} | Values: {box_1024}")

    # Convert bounding box to tensor
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    # Expand the bounding box to match the number of points
    box_torch = box_torch.expand(input_points.shape[0], -1, -1)  # Match batch size of coords
    print(f"Transformed box shape after expansion: {box_torch.shape}")

    # Convert input points to tensor
    coords = torch.tensor(input_points, dtype=torch.float, device=img_embed.device)
    coords = coords.unsqueeze(1)

    # Create labels for the points
    labels = torch.ones((input_points.shape[0],), dtype=torch.long, device=img_embed.device)  # All points labeled as foreground (1)
    print(f"Coordinates shape: {coords.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Generate multiple masks by perturbing the input points or bounding box
    for mask_idx in range(num_masks):
        print(f"Generating mask {mask_idx + 1}/{num_masks}")

        # Perturb the input points with a scale factor based on the mask index
        perturbation_scale = (mask_idx + 1) / num_masks  # Scale factor grows with mask_idx
        perturbed_coords = coords.clone()  # Example: Add small noise for diversity
        #perturbed_coords += torch.randn_like(perturbed_coords) * 0.1 * perturbation_scale  # Perturb coordinates slightly
        perturbed_coords += torch.randn_like(perturbed_coords) * perturbation_scale

        # Perturb the bounding box as well
        perturbed_box_torch = box_torch.clone()
        #perturb_bbox = torch.randn_like(perturbed_box_torch) * 0.05 * perturbation_scale  # Add noise to the bbox
        perturb_bbox = torch.randn_like(perturbed_box_torch) * perturbation_scale
        perturbed_box_torch += perturb_bbox

        # Pass perturbed coordinates and bounding box to the prompt encoder
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=(perturbed_coords, labels),
            boxes=perturbed_box_torch,
            masks=None,
        )

        # Decode mask
        low_res_logits, scores = sam_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )
        # Select the highest IoU scoring mask from the multi-masks
        scores_cpu = scores.cpu().detach().numpy()
        if low_res_logits.size(0) > 1:  # When multimask_output=True
            best_mask_index = np.argmax(scores_cpu[1:]) + 1  # Skip the first mask (index 0)
        else:  # When multimask_output=False
            best_mask_index = 0  # Only one mask is available

        # Ensure the index is within bounds
        best_mask_index = min(best_mask_index, low_res_logits.size(0) - 1)

        low_res_logits = low_res_logits[best_mask_index]

        # Convert to probabilities
        low_res_pred = torch.sigmoid(low_res_logits)  # (10, 1, 256, 256)

        # Aggregate the batch dimension (e.g., take the mean or max across the batch)
        low_res_pred = low_res_pred.mean(dim=0, keepdim=True)  # Aggregate predictions (1, 1, 256, 256)
        low_res_pred = low_res_pred.squeeze(1).unsqueeze(1)  # (1, 1, 256, 256)

        # Interpolate to match ground truth dimensions
        low_res_pred = F.interpolate(
            low_res_pred,
            size=(ground_truth.shape[1], ground_truth.shape[2]),
            mode='bilinear',
            align_corners=False
        )

        # Convert to NumPy
        low_res_pred = low_res_pred.squeeze().cpu().detach().numpy()  # (H, W)

        # Post-process the predictions into binary
        threshold = np.percentile(low_res_pred, 90)  # Use the 90th percentile as the threshold
        medsam_seg = (low_res_pred > threshold).astype(np.uint8)

        # Calculate IoU for this mask
        iou = calculate_iou(medsam_seg, ground_truth)
        iou_scores.append(iou)
        all_masks.append(medsam_seg)

        print(f"IoU for mask {mask_idx + 1}: {iou}")

    # Select the mask with the highest IoU
    best_mask_idx = np.argmax(iou_scores)
    best_mask = all_masks[best_mask_idx]
    best_iou = iou_scores[best_mask_idx]

    print(f"Best mask selected with IoU: {best_iou}")

    # Generate the heatmap for the best mask
    best_low_res_pred = all_masks[best_mask_idx]
    low_res_pred_normalized = (best_low_res_pred - best_low_res_pred.min()) / (best_low_res_pred.max() - best_low_res_pred.min())
    low_res_pred_normalized = (low_res_pred_normalized * 255).astype(np.uint8)

    # Convert to heatmap
    heatmap = cv2.applyColorMap(low_res_pred_normalized, cv2.COLORMAP_JET)

    # Resize the original image to match the heatmap size
    original_image_resized = cv2.resize(resized_image_np, (best_low_res_pred.shape[1], best_low_res_pred.shape[0]))

    # Blend the heatmap with the original image
    overlay = cv2.addWeighted(original_image_resized, 0.6, heatmap, 0.4, 0)

    return best_mask, overlay

# Evaluate CAM-SAM
def evaluate_model_with_cam_sam(
    test_image_folder, gt_folder, output_folder, num_points=3, weight_exponent=0.4,
    sam_checkpoint="sam_vit_h_4b8939.pth", model=None, model_type=None, target_layers=None, sam_model=None, device="cuda", image_index=0
):
    os.makedirs(output_folder, exist_ok=True)
    
    if model_type in ['ViT']:
        reshape_transform = reshape_transform_vit
        #lambda tensor: reshape_transform_vit(tensor, patch_size=16, image_size=(224, 224))
    elif model_type in ['swin']:
        #reshape_transform = None
        reshape_transform = reshape_transform_swin
        #lambda tensor: reshape_transform_swin(tensor, feature_map_size=(14, 14))
    else:
        reshape_transform = None  # CNNs don't need reshape

    # Define CAM methods, including reshape_transform when needed
    methods = [
        ("GradCAM", GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) if reshape_transform else GradCAM(model=model, target_layers=target_layers)),
        ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform) if reshape_transform else GradCAMPlusPlus(model=model, target_layers=target_layers)),
        ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) if reshape_transform else EigenGradCAM(model=model, target_layers=target_layers)),
        ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform, ablation_layer=AblationLayerVit()) if reshape_transform else AblationCAM(model=model, target_layers=target_layers)),
        ("RandomCAM", RandomCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) if reshape_transform else RandomCAM(model=model, target_layers=target_layers)),
    ]

    print("Methods to be evaluated:", [method_name for method_name, _ in methods])

    results = {method_name: [] for method_name, _ in methods}

    image_files = sorted([f for f in os.listdir(test_image_folder) if f.endswith('.jpg')])
    if image_index < 0 or image_index >= len(image_files):
        print(f"Invalid image index {image_index}, skipping...")
        return {}

    image_name = image_files[image_index]
    set_seed()
    img_path = os.path.join(test_image_folder, image_name)
    gt_image_name = image_name.replace('.jpg', '_segmentation.png')
    gt_path = os.path.join(gt_folder, gt_image_name)

    if not os.path.exists(gt_path):
        print(f"Ground truth not found for {image_name}, skipping...")
        return {}

    ground_truth = transform_mask(gt_path)
    print("Ground truth shape:", ground_truth.shape)
    img = cv2.imread(img_path)
    transformed_image = transform_image(img_path)

    for method_name, cam_method in methods:
        method_folder = os.path.join(output_folder, method_name)
        heatmap_resized, _ = process_image(img_path, model, cam_method)
        if (heatmap_resized is None or np.isnan(heatmap_resized).any() or 
            heatmap_resized.max() == heatmap_resized.min()):
            print(f"Warning: Skipping {image_name} due to invalid heatmap from {method_name}")
            continue
        
        heatmap_height, heatmap_width = heatmap_resized.shape
        print(f"Heatmap size: {heatmap_width}x{heatmap_height}")

        # Sample points in heatmap space
        sampled_points = sample_points(heatmap_resized, num_points,amplification_factor=5)
        print("Sampled points (in heatmap space):", sampled_points)

        # Normalize sampled points to [0, 1] range
        input_points = np.array(sampled_points, dtype=np.float32)
        input_points[:, 0] /= heatmap_width  # Normalize x-coordinates
        input_points[:, 1] /= heatmap_height  # Normalize y-coordinates
        print("Normalized input points:", input_points)

        # Scale normalized points to 1024x1024 resolution
        input_points[:, 0] *= 1024  # Scale x-coordinates
        input_points[:, 1] *= 1024  # Scale y-coordinates
        print("Scaled input points (mapped to 1024x1024):", input_points)

        # Convert sampled points into bounding box for MedSAM
        x_min, y_min = np.min(input_points, axis=0)
        x_max, y_max = np.max(input_points, axis=0)
        box_1024 = np.array([[x_min, y_min, x_max, y_max]])
        print("Scaled bounding box (box_1024):", box_1024)

        # Debugging: Print scaled points and bounding box
        print("Scaled input points:", input_points)
        print("Scaled bounding box (box_1024):", box_1024)

        # Generate segmentation mask using MedSAM
        #img_embed = sam_model.get_image_embedding(transformed_image.unsqueeze(0).to(device))
        print("Image shape:", transformed_image.shape)
        #transformed_image = transformed_image.permute(0, 2, 1)
        print("Transformed image shape:", transformed_image.shape)
        # Resize the image to 1024x1024 (or any expected input size of the model)
        resized_image = F.interpolate(transformed_image.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False)
        print("Resized image shape:", resized_image.shape)
        resized_image_np = resized_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HxWxC format for OpenCV
        resized_image_np = (resized_image_np * 255).astype(np.uint8)  # Convert to uint8 for visualization

        # Draw the bounding box on the resized image
        output_path_bbox = os.path.join(method_folder, f"{image_name}_bbox.png")
        draw_bounding_box(resized_image_np, box_1024, input_points, output_path=output_path_bbox)

        img_embed = sam_model.image_encoder(resized_image.to(device))
        print("Image embedding shape:", img_embed.shape)
        predicted_mask, overlay = medsam_inference(sam_model, img_embed, box_1024, input_points, ground_truth,resized_image_np)
        # Save the overlay
        output_path_overlay = output_path_bbox = os.path.join(method_folder, f"{image_name}_low_res_pred_overlay.png")
        cv2.imwrite(output_path_overlay, overlay)
        print(f"Low res pred overlay saved to {output_path_overlay}")
        
        predicted_mask = post_process_mask(predicted_mask, ground_truth,method_folder, image_name)  
        print("Predicted mask shape:", predicted_mask.shape)
        metrics = calculate_metrics(predicted_mask, ground_truth)
        results[method_name].append({"image": image_name, "metrics": metrics})
        
        # Save the heatmap as an image
        heatmap_path = os.path.join(method_folder, f"{image_name}_heatmap.png")
        heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap_colored)
  
        original_image = cv2.imread(img_path)
        heatmap_overlay = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)
        heatmap_overlay_path = os.path.join(method_folder, f"{image_name}_heatmap_overlay.png")
        cv2.imwrite(heatmap_overlay_path, heatmap_overlay)
    
        
        # Save the final mask with metrics overlay
        os.makedirs(method_folder, exist_ok=True)
        output_image_path = os.path.join(method_folder, f"{image_name}_result.png")

        # Convert the predicted mask to a visualization-friendly format
        visualized_mask_colored = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR for text overlay

        # Overlay metrics on the mask
        metrics_text = [
            f"Precision: {metrics['Precision']:.2f}",
            f"Recall: {metrics['Recall']:.2f}",
            f"F1 Score: {metrics['F1']:.2f}",
            f"IoU: {metrics['IoU']:.2f}"
        ]
        
        font_scale = 2.0  # Increase the font size
        font_thickness = 2  # Adjust thickness for better visibility
        y_offset = 50  # Increased initial y-coordinate
        line_spacing = int(50 * font_scale)  # Dynamic spacing based on font size

        for i, text in enumerate(metrics_text):
            cv2.putText(
                visualized_mask_colored, text, (10, y_offset + i * line_spacing),  # Adjusted spacing
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
            )

        # Save the final mask with metrics overlay
        cv2.imwrite(output_image_path, visualized_mask_colored)
        print(f"Final mask with metrics saved to {output_image_path}")

        # Create a figure of the original image, predicted mask, and ground truth mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the original image with the predicted mask overlaid
        original_with_overlay = cv2.addWeighted(original_image, 0.6, cv2.applyColorMap(predicted_mask, cv2.COLORMAP_JET), 0.4, 0)
        axes[0].imshow(cv2.cvtColor(original_with_overlay, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image + Prediction")
        axes[0].axis("off")

        # Plot the predicted mask
        axes[1].imshow(predicted_mask, cmap="gray")
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        ground_truth_vis = ground_truth.squeeze(0)
        # Plot the ground truth mask
        axes[2].imshow(ground_truth_vis, cmap="gray")
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

        # Save and show the visualization
        visualization_path = os.path.join(method_folder, f"{image_name}_comparison.png")
        plt.savefig(visualization_path, bbox_inches="tight")
        plt.show()
        # Convert grayscale predicted mask to color
        predicted_mask_colored = cv2.applyColorMap(predicted_mask, cv2.COLORMAP_JET)

        # Overlay the mask on the original image
        original_with_overlay = cv2.addWeighted(original_image, 0.6, predicted_mask_colored, 0.4, 0)

        # Draw bounding box on the image
        # Find contours of the mask
        contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ensure at least one contour is found
        if contours:
            # Get the bounding box that encloses the mask
            x_min, y_min, w, h = cv2.boundingRect(contours[0])  # (x, y, width, height)
            x_max, y_max = x_min + w, y_min + h

            # Draw bounding box on the original image (without overlaying the mask)
            original_with_bbox = original_image.copy()  # Make a copy to preserve the original image
            cv2.rectangle(original_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

            # Save the updated image with just the bounding box
            bbox_only_path = os.path.join(method_folder, f"{image_name}_original_image_bbox.png")
            cv2.imwrite(bbox_only_path, original_with_bbox)
            print(f"Original image with bounding box saved to {bbox_only_path}")
            cv2.rectangle(original_with_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) 

        # Save the updated visualization
        overlay_bbox_path = os.path.join(method_folder, f"{image_name}_overlay_bbox.png")
        cv2.imwrite(overlay_bbox_path, original_with_overlay)
        print(f"Overlay with bounding box saved to {overlay_bbox_path}")
        
        # 'Unsupervised learning' vis
        # Convert grayscale predicted mask to color (for foreground)
        predicted_mask_colored = cv2.applyColorMap(predicted_mask, cv2.COLORMAP_JET)

        # Create a background mask (inverse of the predicted mask)
        background_mask = cv2.bitwise_not(predicted_mask)  # Invert mask to get the background

        # Color the foreground in blue (255, 0, 0) and the background in yellow (0, 255, 255)
        foreground_color = np.zeros_like(original_image)  # Black background initially
        foreground_color[predicted_mask == 255] = [255, 0, 0]  # Blue for foreground (1)

        background_color = np.zeros_like(original_image)  # Black background initially
        background_color[background_mask == 255] = [0, 255, 255]  # Yellow for background (0)

        # Apply transparency by adding an alpha channel (0.6 for foreground, 0.4 for background)
        alpha = 0.6  # Transparency for foreground
        beta = 0.4   # Transparency for background

        # Overlay foreground with transparency
        foreground_with_alpha = cv2.addWeighted(original_image, 1 - alpha, foreground_color, alpha, 0)

        # Overlay background with transparency
        original_with_overlay = cv2.addWeighted(foreground_with_alpha, 1 - beta, background_color, beta, 0)

        # Add text labels for foreground (1) and background (0)
        cv2.putText(original_with_overlay, 'Foreground (1)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(original_with_overlay, 'Background (0)', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Save the updated visualization
        overlay_unsupervised_path = os.path.join(method_folder, f"{image_name}_overlay_unsupervised.png")
        cv2.imwrite(overlay_unsupervised_path, original_with_overlay)
        print(f"Overlay with bounding box and labels saved to {overlay_unsupervised_path}")
    return results


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Lab-pc
    #root_dir = "/home/matthewcockayne/Documents/PhD/data/ISIC_2017/"
    #test_image_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Data/")
    #gt_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Part1_GroundTruth")
    #output_folder = "/home/matthewcockayne/Documents/PhD/Zero-Shot_SkinLesion_Segmentation/results/test_images/results"
    # cluster
    root_dir = "/home/xrai/datasets/ISIC2017/"
    test_image_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Data/")
    gt_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Part1_GroundTruth")
    output_folder = "/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/CAM-MedSAM/test/vis"

    os.makedirs(output_folder, exist_ok=True)
    
    sam_checkpoint = "./results/model_checkpoints/medsam_vit_b.pth"
    medsam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
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
        print(f"Processing model: {model_name}")
        # Path to the saved checkpoint
        # If zero-shot
        checkpoint_path = None
        # Else if using ISIC-trained, use path
        # lab-pc
        #checkpoint_path = f"/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/classification/{model_name}/{model_name}_best_model.pth"
        # cluster
        #checkpoint_path = f"/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/classification/{model_name}/{model_name}_best_model.pth"
        # Load model
        model = load_model(model_name)
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint for {model_name} from {checkpoint_path}...")
            model = model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Checkpoint not found for {model_name}. Using pretrained weights instead.")
        model.to(device)
        model.eval()

        print(f"Testing {model_name}...")
        
        model_output_folder = os.path.join(output_folder, model_name)
        os.makedirs(model_output_folder, exist_ok=True)
    
        target_layers = layer_selection(model, model_name)

        results = evaluate_model_with_cam_sam(
            test_image_folder, gt_folder, model_output_folder,
            num_points=20, weight_exponent=0.4, sam_checkpoint=sam_checkpoint,
            model=model, model_type=model_type, target_layers=target_layers, sam_model=medsam_model, device=device, image_index=1
        )
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()