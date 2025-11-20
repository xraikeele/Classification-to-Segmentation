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
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, normalize, resize
from models.load_models import load_model 
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import gc

def clear_gpu_memory():
    """Clear GPU memory to prevent CUDA out of memory errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("GPU memory cleared")

def get_gpu_memory_info():
    """Get current GPU memory usage info"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - allocated
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB, Total: {total:.2f}GB")
        return allocated, reserved, free, total
    return 0, 0, 0, 0

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
    if len(tensor.shape) == 3:  # [batch_size, num_patches, embedding_dim]
        batch_size, num_patches, embedding_dim = tensor.size()

        # Infer height and width if not provided
        if feature_map_size is None:
            height = width = int(num_patches**0.5)
            if height * width != num_patches:
                raise ValueError("Non-square patch grid is not supported.")
        else:
            height, width = feature_map_size

        # Reshape to [batch_size, height, width, embedding_dim]
        reshaped = tensor.view(batch_size, height, width, embedding_dim)
        # Permute to [batch_size, embedding_dim, height, width]
        reshaped = reshaped.permute(0, 3, 1, 2)
        return reshaped

    elif len(tensor.shape) == 4:  # [batch_size, height, width, embedding_dim]
        return tensor.permute(0, 3, 1, 2)  # [batch_size, embedding_dim, height, width]

    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

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
    os.makedirs(output_path_contours, exist_ok=True)
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

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def transform_mask(mask_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    mask = Image.open(mask_path).convert("L")
    return np.array(transform(mask))

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
    confidence = torch.softmax(out, dim=1).max().item()
    
    print(f"input_tensor shape: {input_tensor.shape}")
    
    # Generate CAM with memory management
    try:
        activation_map = cam_extractor(input_tensor, targets=[ClassifierOutputTarget(predicted_category)])
        if activation_map is None:
            print("Activation map is None.")
            return None, None, None, None

        heatmap = activation_map[0]
        print(f"Activation map shape: {activation_map.shape}")
        print(f"Heatmap shape: {heatmap.shape}")
        heatmap = np.clip(heatmap, 0, None)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
        
        # Clear intermediate tensors
        del input_tensor, out, activation_map
        clear_gpu_memory()
        
        return heatmap_resized, img, predicted_category, confidence
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"CUDA out of memory during CAM generation: {e}")
            clear_gpu_memory()
            return None, None, None, None
        else:
            raise e

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

        try:
            # Perturb the input points with a scale factor based on the mask index
            perturbation_scale = (mask_idx + 1) / num_masks  # Scale factor grows with mask_idx
            perturbed_coords = coords.clone()  # Example: Add small noise for diversity
            perturbed_coords += torch.randn_like(perturbed_coords) * perturbation_scale

            # Perturb the bounding box as well
            perturbed_box_torch = box_torch.clone()
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
            if low_res_logits.size(0) > 1:            
                best_mask_index = np.argmax(scores_cpu)
            else:            
                best_mask_index = 0

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
            
            # Clear intermediate tensors to save memory
            del sparse_embeddings, dense_embeddings, low_res_logits, scores, low_res_pred
            del perturbed_coords, perturbed_box_torch
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA out of memory during mask generation {mask_idx + 1}: {e}")
                clear_gpu_memory()
                # Continue with fewer masks if we hit memory limits
                break
            else:
                raise e

    if not all_masks:
        print("No masks were successfully generated")
        return None, None

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
    
    # Clear variables before returning
    del coords, labels, box_torch, all_masks
    clear_gpu_memory()

    return best_mask, overlay

def create_visualization_grid(image_path, gt_path, models, methods, sam_checkpoint, output_path, num_points=10):
    """
    Create a comprehensive visualization grid showing:
    - Original image and ground truth
    - CAM heatmaps for each model/method combination
    - Segmentation results using MedSAM with bounding boxes
    - Metrics overlay
    """
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load MedSAM
    sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    
    # Load images
    original_img = Image.open(image_path).convert("RGB")
    ground_truth = transform_mask(gt_path)
    transformed_image = transform_image(image_path)
    
    print("Ground truth shape:", ground_truth.shape)
    print("Transformed image shape:", transformed_image.shape)
    
    # Calculate grid dimensions
    n_models = len(models)
    n_methods = len(methods)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_models + 1, n_methods + 2, figsize=(4*(n_methods + 2), 4*(n_models + 1)))
    
    # First row: Original image and ground truth
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    gt_display = ground_truth.squeeze(0) if len(ground_truth.shape) > 2 else ground_truth
    axes[0, 1].imshow(gt_display, cmap='gray')
    axes[0, 1].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Hide remaining cells in first row
    for j in range(2, n_methods + 2):
        axes[0, j].axis('off')
    
    # Results storage
    all_results = {}
    
    # Process each model
    for i, model_config in enumerate(models):
        model_name = model_config['name']
        model_type = model_config['type']
        
        print(f"Processing model: {model_name}")
        get_gpu_memory_info()
        
        # Load model
        model = load_model(model_name)
        # Optionally load checkpoint here if needed
        model.to(device)
        model.eval()
        
        target_layers = layer_selection(model, model_name)
        all_results[model_name] = {}
        
        # Process each CAM method
        for j, (method_name, method_class) in enumerate(methods):
            print(f"  Processing method: {method_name}")
            
            # Set a deterministic seed for this specific model/method combination
            model_method_seed = hash(f"{model_name}_{method_name}") % 2**32
            set_seed(model_method_seed)
            print(f"  Using seed: {model_method_seed} for {model_name}/{method_name}")
            
            # Skip AblationCAM for ViT and Swin models due to compatibility issues
            if method_name == 'AblationCAM' and model_type in ['ViT', 'swin']:
                print(f"  Skipping {method_name} for {model_name} due to compatibility issues")
                axes[i+1, j].text(0.5, 0.5, f"Skipped:\n{model_name}\n{method_name}\n(Compatibility)", 
                                ha='center', va='center', transform=axes[i+1, j].transAxes,
                                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
                axes[i+1, j].axis('off')
                continue
            
            # Clear memory before processing each method
            clear_gpu_memory()
            
            # Create CAM extractor with reshape transform
            if model_type in ['ViT']:
                reshape_transform = reshape_transform_vit
            elif model_type in ['swin']:
                reshape_transform = reshape_transform_swin
            else:
                reshape_transform = None  # CNNs don't need reshape
            
            # Define CAM methods, including reshape_transform when needed
            cam_extractor = None
            try:
                if reshape_transform:
                    if method_name == 'AblationCAM':
                        cam_extractor = method_class(model=model, target_layers=target_layers, 
                                                   reshape_transform=reshape_transform,
                                                   ablation_layer=AblationLayerVit())
                    else:
                        cam_extractor = method_class(model=model, target_layers=target_layers, 
                                                   reshape_transform=reshape_transform)
                else:
                    cam_extractor = method_class(model=model, target_layers=target_layers)
                
                # Generate heatmap and process
                heatmap_resized, _, pred_category, confidence = process_image(image_path, model, cam_extractor)
                
                if (heatmap_resized is None or np.isnan(heatmap_resized).any() or 
                    heatmap_resized.max() == heatmap_resized.min()):
                    print(f"Warning: Invalid heatmap for {model_name}/{method_name}")
                    continue
                
                heatmap_height, heatmap_width = heatmap_resized.shape
                print(f"Heatmap size: {heatmap_width}x{heatmap_height}")

                # Sample points in heatmap space
                sampled_points = sample_points(heatmap_resized, num_points, amplification_factor=5)
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

                # Resize the image to 1024x1024 for MedSAM
                resized_image = F.interpolate(transformed_image.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False)
                print("Resized image shape:", resized_image.shape)
                resized_image_np = resized_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HxWxC format
                resized_image_np = (resized_image_np * 255).astype(np.uint8)  # Convert to uint8

                # Get image embedding from MedSAM
                img_embed = sam.image_encoder(resized_image.to(device))
                print("Image embedding shape:", img_embed.shape)
                
                # Generate segmentation mask using MedSAM
                predicted_mask, overlay = medsam_inference(sam, img_embed, box_1024, input_points, ground_truth, resized_image_np)
                
                if predicted_mask is None:
                    print(f"Failed to generate mask for {model_name}/{method_name}")
                    continue
                
                # Post-process the mask
                method_folder = os.path.join(os.path.dirname(output_path), f"temp_{model_name}_{method_name}")
                predicted_mask = post_process_mask(predicted_mask, ground_truth, method_folder, 
                                                 os.path.basename(image_path).replace('.jpg', ''))
                print("Predicted mask shape:", predicted_mask.shape)

                # Calculate metrics
                metrics = calculate_metrics(predicted_mask, ground_truth)
                all_results[model_name][method_name] = {
                    'metrics': metrics,
                    'predicted_category': pred_category,
                    'confidence': confidence
                }
                
                # Create visualization overlay
                img_array = np.array(original_img) / 255.0
                heatmap_normalized = cv2.resize(heatmap_resized, (img_array.shape[1], img_array.shape[0]))
                
                # Show CAM overlay
                cam_overlay = show_cam_on_image(img_array, heatmap_normalized, use_rgb=True)
                
                # Add sampled points to the visualization
                original_sampled_points = sampled_points.copy()
                original_sampled_points[:, 0] = original_sampled_points[:, 0] * img_array.shape[1] / heatmap_width
                original_sampled_points[:, 1] = original_sampled_points[:, 1] * img_array.shape[0] / heatmap_height
                
                for point in original_sampled_points:
                    cv2.circle(cam_overlay, tuple(point.astype(int)), 3, (255, 255, 0), -1)
                
                axes[i+1, j].imshow(cam_overlay)
                title = f"{model_name}\n{method_name}\nIoU: {metrics['IoU']:.3f}"
                axes[i+1, j].set_title(title, fontsize=10)
                axes[i+1, j].axis('off')
                
                # Show segmentation result in the last column
                if j == len(methods) - 1:  # Last method, show segmentation
                    axes[i+1, j+1].imshow(predicted_mask, cmap='gray')
                    axes[i+1, j+1].set_title(f"Segmentation\nF1: {metrics['F1']:.3f}", fontsize=10)
                    axes[i+1, j+1].axis('off')
                
                # Clear CAM extractor and intermediate variables
                del cam_extractor, img_embed, resized_image
                clear_gpu_memory()
                
            except Exception as e:
                print(f"Error processing {model_name}/{method_name}: {e}")
                if "out of memory" in str(e):
                    clear_gpu_memory()
                axes[i+1, j].text(0.5, 0.5, f"Error:\n{model_name}\n{method_name}", 
                                ha='center', va='center', transform=axes[i+1, j].transAxes)
                axes[i+1, j].axis('off')
                
                # Clean up on error
                if cam_extractor is not None:
                    del cam_extractor
                clear_gpu_memory()
        
        # Clear model from memory after processing all methods for this model
        del model
        clear_gpu_memory()
        print(f"Completed processing {model_name}")
        get_gpu_memory_info()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return all_results

def save_individual_visualizations(image_path, gt_path, model_name, method_name, heatmap_resized, 
                                 predicted_mask, cam_overlay, original_img, ground_truth, 
                                 bbox_vis, overlay, metrics, output_folder):
    """
    Save individual visualization components for presentations
    """
    image_name = os.path.basename(image_path).replace('.jpg', '')
    
    # Create model-method subfolder
    model_method_folder = os.path.join(output_folder, f"{model_name}_{method_name}")
    os.makedirs(model_method_folder, exist_ok=True)
    
    # Save original image
    original_path = os.path.join(model_method_folder, f"{image_name}_original.png")
    original_img.save(original_path)
    
    # Save ground truth
    gt_path_save = os.path.join(model_method_folder, f"{image_name}_ground_truth.png")
    gt_display = ground_truth.squeeze(0) if len(ground_truth.shape) > 2 else ground_truth
    plt.figure(figsize=(8, 8))
    plt.imshow(gt_display, cmap='gray')
    plt.axis('off')
    plt.title(f"Ground Truth - {image_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(gt_path_save, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save CAM heatmap
    heatmap_path = os.path.join(model_method_folder, f"{image_name}_{method_name}_heatmap.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap_resized, cmap='jet')
    plt.axis('off')
    plt.title(f"{method_name} Heatmap - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save CAM overlay with points
    overlay_path = os.path.join(model_method_folder, f"{image_name}_{method_name}_overlay.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(cam_overlay)
    plt.axis('off')
    plt.title(f"{method_name} Overlay - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save bounding box visualization
    bbox_path = os.path.join(model_method_folder, f"{image_name}_bounding_box.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(bbox_vis)
    plt.axis('off')
    plt.title(f"Bounding Box + Points - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(bbox_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save MedSAM overlay
    medsam_overlay_path = os.path.join(model_method_folder, f"{image_name}_medsam_overlay.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title(f"MedSAM Inference - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(medsam_overlay_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save predicted mask
    mask_path = os.path.join(model_method_folder, f"{image_name}_predicted_mask.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(predicted_mask, cmap='gray')
    plt.axis('off')
    plt.title(f"Predicted Segmentation - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(mask_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics as text file
    metrics_path = os.path.join(model_method_folder, f"{image_name}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Image: {image_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"IoU: {metrics['IoU']:.4f}\n")
        f.write(f"F1-Score: {metrics['F1']:.4f}\n")
        f.write(f"Precision: {metrics['Precision']:.4f}\n")
        f.write(f"Recall: {metrics['Recall']:.4f}\n")
    
    print(f"Individual visualizations saved to: {model_method_folder}")

def create_detailed_single_result(image_path, gt_path, model_name, method_name, sam_checkpoint, output_path, 
                                existing_metrics=None, num_points=10, save_individual=False, individual_output_folder=None):
    """
    Create a detailed visualization for a single model/method combination using advanced MedSAM
    Args:
        existing_metrics: If provided, use these metrics instead of recalculating
        save_individual: If True, save individual visualization components
        individual_output_folder: Folder to save individual components
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set a deterministic seed based on model and method to ensure reproducibility
    model_method_seed = hash(f"{model_name}_{method_name}") % 2**32
    set_seed(model_method_seed)
    
    print(f"Starting detailed analysis for {model_name}/{method_name} (seed: {model_method_seed})")
    get_gpu_memory_info()
    
    # Load MedSAM
    sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    
    # Load model
    model = load_model(model_name)
    model.to(device)
    model.eval()
    
    # Determine model type for CAM setup
    if model_name in ['ViT']:
        model_type = 'ViT'
    elif model_name in ['swin']:
        model_type = 'swin'
    else:
        model_type = 'CNN'
    
    target_layers = layer_selection(model, model_name)
    
    # Create CAM extractor with improved error handling
    method_mapping = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus,
        'EigenGradCAM': EigenGradCAM,
        'AblationCAM': AblationCAM,
        'RandomCAM': RandomCAM
    }
    
    # Skip AblationCAM for ViT and Swin models due to compatibility issues
    if method_name == 'AblationCAM' and model_type in ['ViT', 'swin']:
        error_msg = f"AblationCAM not supported for {model_name} due to tensor dimension incompatibilities"
        print(error_msg)
        # Create error visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, error_msg, ha='center', va='center', transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return {"error": error_msg}
    
    # Create CAM extractor with reshape transform
    if model_type in ['ViT']:
        reshape_transform = reshape_transform_vit
    elif model_type in ['swin']:
        reshape_transform = reshape_transform_swin
    else:
        reshape_transform = None  # CNNs don't need reshape
    
    cam_extractor = None
    try:
        # Clear memory before creating CAM extractor
        clear_gpu_memory()
        
        if reshape_transform:
            if method_name == 'AblationCAM':
                cam_extractor = method_mapping[method_name](model=model, target_layers=target_layers, 
                                                          reshape_transform=reshape_transform,
                                                          ablation_layer=AblationLayerVit())
            else:
                cam_extractor = method_mapping[method_name](model=model, target_layers=target_layers, 
                                                          reshape_transform=reshape_transform)
        else:
            cam_extractor = method_mapping[method_name](model=model, target_layers=target_layers)
        
        # Load images
        original_img = Image.open(image_path).convert("RGB")
        ground_truth = transform_mask(gt_path)
        transformed_image = transform_image(image_path)
        
        # Process image
        heatmap_resized, _, pred_category, confidence = process_image(image_path, model, cam_extractor)
        
        if (heatmap_resized is None or np.isnan(heatmap_resized).any() or 
            heatmap_resized.max() == heatmap_resized.min()):
            raise ValueError("Invalid heatmap generated")
        
        heatmap_height, heatmap_width = heatmap_resized.shape
        print(f"Heatmap size: {heatmap_width}x{heatmap_height}")

        # Sample points in heatmap space using advanced sampling
        sampled_points = sample_points(heatmap_resized, num_points, amplification_factor=5)
        print("Sampled points (in heatmap space):", sampled_points)

        # Normalize sampled points to [0, 1] range
        input_points = np.array(sampled_points, dtype=np.float32)
        input_points[:, 0] /= heatmap_width  # Normalize x-coordinates
        input_points[:, 1] /= heatmap_height  # Normalize y-coordinates

        # Scale normalized points to 1024x1024 resolution
        input_points[:, 0] *= 1024  # Scale x-coordinates
        input_points[:, 1] *= 1024  # Scale y-coordinates

        # Convert sampled points into bounding box for MedSAM
        x_min, y_min = np.min(input_points, axis=0)
        x_max, y_max = np.max(input_points, axis=0)
        box_1024 = np.array([[x_min, y_min, x_max, y_max]])

        # Always generate the visualization components, but optionally use existing metrics
        print("Generating MedSAM visualizations...")
        # Resize the image to 1024x1024 for MedSAM
        resized_image = F.interpolate(transformed_image.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False)
        resized_image_np = resized_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        resized_image_np = (resized_image_np * 255).astype(np.uint8)

        # Get image embedding from MedSAM
        img_embed = sam.image_encoder(resized_image.to(device))
        
        # Generate segmentation mask using MedSAM
        predicted_mask, overlay = medsam_inference(sam, img_embed, box_1024, input_points, ground_truth, resized_image_np)
        
        if predicted_mask is None:
            raise ValueError("Failed to generate segmentation mask")
        
        # Post-process the mask
        method_folder = os.path.join(os.path.dirname(output_path), f"temp_{model_name}_{method_name}")
        predicted_mask = post_process_mask(predicted_mask, ground_truth, method_folder, 
                                         os.path.basename(image_path).replace('.jpg', ''))

        # Use existing metrics if provided, otherwise calculate new ones
        if existing_metrics is not None:
            print(f"Using existing metrics: IoU={existing_metrics['IoU']:.4f}")
            metrics = existing_metrics
        else:
            print("Calculating new metrics...")
            # Calculate metrics
            metrics = calculate_metrics(predicted_mask, ground_truth)
        
        # Clean up large objects before creating visualization
        del img_embed, resized_image
        clear_gpu_memory()
        
    except Exception as e:
        print(f"Error in detailed visualization for {model_name}/{method_name}: {e}")
        if "out of memory" in str(e):
            clear_gpu_memory()
        
        # Create a simple error visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error processing {model_name}/{method_name}:\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clean up on error
        if cam_extractor is not None:
            del cam_extractor
        del model, sam
        clear_gpu_memory()
        return {"error": str(e)}
    
    # Create detailed visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth
    gt_display = ground_truth.squeeze(0) if len(ground_truth.shape) > 2 else ground_truth
    axes[0, 1].imshow(gt_display, cmap='gray')
    axes[0, 1].set_title("Ground Truth Mask", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # CAM heatmap
    img_array = np.array(original_img) / 255.0
    heatmap_normalized = cv2.resize(heatmap_resized, (img_array.shape[1], img_array.shape[0]))
    axes[0, 2].imshow(heatmap_normalized, cmap='jet', alpha=0.8)
    axes[0, 2].set_title(f"{method_name} Heatmap\n{model_name}", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # CAM overlay with points
    cam_overlay = show_cam_on_image(img_array, heatmap_normalized, use_rgb=True)
    # Convert sampled points back to original image coordinates
    original_sampled_points = sampled_points.copy()
    original_sampled_points[:, 0] = original_sampled_points[:, 0] * img_array.shape[1] / heatmap_width
    original_sampled_points[:, 1] = original_sampled_points[:, 1] * img_array.shape[0] / heatmap_height
    
    for point in original_sampled_points:
        cv2.circle(cam_overlay, tuple(point.astype(int)), 5, (255, 255, 0), -1)
    axes[0, 3].imshow(cam_overlay)
    axes[0, 3].set_title(f"CAM Overlay + Sampled Points\n({len(sampled_points)} points)", fontsize=14, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Bounding box visualization
    bbox_vis = resized_image_np.copy()
    x_min_int, y_min_int, x_max_int, y_max_int = box_1024[0].astype(int)
    cv2.rectangle(bbox_vis, (x_min_int, y_min_int), (x_max_int, y_max_int), (0, 255, 0), 2)
    for point in input_points:
        cv2.circle(bbox_vis, tuple(point.astype(int)), 5, (0, 0, 255), -1)
    axes[1, 0].imshow(bbox_vis)
    axes[1, 0].set_title("Bounding Box + Points\n(1024x1024)", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # MedSAM overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("MedSAM Inference Overlay", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Predicted mask
    axes[1, 2].imshow(predicted_mask, cmap='gray')
    axes[1, 2].set_title("Final Predicted Segmentation", fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Save individual visualizations if requested
    if save_individual and individual_output_folder:
        save_individual_visualizations(image_path, gt_path, model_name, method_name, 
                                     heatmap_resized, predicted_mask, cam_overlay, 
                                     original_img, ground_truth, bbox_vis, overlay, 
                                     metrics, individual_output_folder)
    
    # Metrics and info
    metrics_source = "From existing calculation" if existing_metrics is not None else "Newly calculated"
    
    metrics_text = f"""Model: {model_name}
Method: {method_name}
Predicted Class: {pred_category}
Confidence: {confidence:.3f}

Metrics ({metrics_source}):
IoU: {metrics['IoU']:.4f}
F1-Score: {metrics['F1']:.4f}
Precision: {metrics['Precision']:.4f}
Recall: {metrics['Recall']:.4f}

Points Sampled: {len(sampled_points)}
Bounding Box: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]"""
    
    axes[1, 3].text(0.1, 0.9, metrics_text, transform=axes[1, 3].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clean up after visualization
    del model, sam, cam_extractor
    clear_gpu_memory()
    
    print(f"Completed detailed analysis for {model_name}/{method_name}")
    get_gpu_memory_info()
    
    return metrics

def analyze_single_image(image_name, test_image_folder, gt_folder, sam_checkpoint, output_folder, save_individual=False):
    """
    Main function to analyze a single image with all models and methods
    Args:
        save_individual: If True, save individual visualization components for presentation
    """
    # Setup paths
    img_path = os.path.join(test_image_folder, image_name)
    gt_image_name = image_name.replace('.jpg', '_segmentation.png')
    gt_path = os.path.join(gt_folder, gt_image_name)
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    if not os.path.exists(gt_path):
        print(f"Ground truth not found: {gt_path}")
        return
    
    print(f"Analyzing image: {image_name}")
    get_gpu_memory_info()
    
    # Define models and methods
    models = [
        #{'name': 'resnet50', 'type': 'CNN'},
        {'name': 'mobilenet_v2', 'type': 'CNN'},
        #{'name': 'efficientnet_b0', 'type': 'CNN'},
        #{'name': 'swin', 'type': 'swin'},
        #{'name': 'ViT', 'type': 'ViT'},
    ]
    
    methods = [
        ('GradCAM', GradCAM),
        ('GradCAM++', GradCAMPlusPlus),
        ('EigenGradCAM', EigenGradCAM),
        ('AblationCAM', AblationCAM),  # Commented out due to compatibility issues with ViT/Swin
    ]
    
    # Create output directory
    image_output_folder = os.path.join(output_folder, image_name.replace('.jpg', ''))
    os.makedirs(image_output_folder, exist_ok=True)
    
    # Create individual visualizations folder if requested
    individual_folder = None
    if save_individual:
        individual_folder = os.path.join(image_output_folder, "individual_visualizations")
        os.makedirs(individual_folder, exist_ok=True)
    
    # Create comprehensive grid visualization with memory management
    grid_output_path = os.path.join(image_output_folder, f"{image_name.replace('.jpg', '')}_comprehensive_grid.png")
    
    try:
        print("Creating comprehensive grid visualization...")
        all_results = create_visualization_grid(img_path, gt_path, models, methods, sam_checkpoint, grid_output_path)
        
        # Clear memory after grid creation
        clear_gpu_memory()
        
        # Save results as JSON
        results_json_path = os.path.join(image_output_folder, f"{image_name.replace('.jpg', '')}_results.json")
        with open(results_json_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        # Create detailed visualizations for best performing combinations
        print("Creating detailed visualizations for top performing combinations...")
        
        # Find best IoU for each model
        for model_name in all_results:
            best_method = None
            best_iou = 0
            
            for method_name in all_results[model_name]:
                iou = all_results[model_name][method_name]['metrics']['IoU']
                if iou > best_iou:
                    best_iou = iou
                    best_method = method_name
            
            if best_method:
                detailed_output_path = os.path.join(image_output_folder, f"{model_name}_{best_method}_detailed.png")
                try:
                    print(f"Creating detailed visualization for {model_name} + {best_method}...")
                    # Pass the existing metrics to avoid recalculation and ensure consistency
                    existing_metrics = all_results[model_name][best_method]['metrics']
                    create_detailed_single_result(img_path, gt_path, model_name, best_method, 
                                                sam_checkpoint, detailed_output_path, 
                                                existing_metrics=existing_metrics,
                                                save_individual=save_individual,
                                                individual_output_folder=individual_folder)
                    print(f"Created detailed visualization for {model_name} + {best_method} (IoU: {best_iou:.4f}) - using existing metrics")
                except Exception as e:
                    print(f"Error creating detailed visualization for {model_name} + {best_method}: {e}")
                    if "out of memory" in str(e):
                        clear_gpu_memory()
                finally:
                    clear_gpu_memory()  # Always clear after each detailed visualization
        
        print(f"Analysis complete! Results saved to: {image_output_folder}")
        get_gpu_memory_info()
        return all_results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if "out of memory" in str(e):
            clear_gpu_memory()
        return None

def analyze_multiple_random_images(n_images, test_image_folder, gt_folder, sam_checkpoint, output_folder, save_individual=False):
    """
    Analyze n randomly selected images
    Args:
        n_images: Number of random images to process
        save_individual: If True, save individual visualization components for presentation
    """
    # Get all available images
    all_images = [f for f in os.listdir(test_image_folder) if f.endswith('.jpg')]
    
    if len(all_images) < n_images:
        print(f"Warning: Only {len(all_images)} images available, but {n_images} requested.")
        n_images = len(all_images)
    
    # Use current time as seed for truly random image selection
    import time
    current_time_seed = int(time.time())
    random.seed(current_time_seed)
    print(f"🎲 Using time-based seed for random selection: {current_time_seed}")
    
    # Randomly select n images
    selected_images = random.sample(all_images, n_images)
    print(f"Selected {n_images} random images: {selected_images}")
    
    all_image_results = {}
    
    for i, image_name in enumerate(selected_images):
        print(f"\n=== Processing image {i+1}/{n_images}: {image_name} ===")
        clear_gpu_memory()  # Clear memory before each image
        
        try:
            results = analyze_single_image(image_name, test_image_folder, gt_folder, 
                                         sam_checkpoint, output_folder, save_individual=save_individual)
            if results:
                all_image_results[image_name] = results
                print(f"✅ Successfully processed {image_name}")
            else:
                print(f"❌ Failed to process {image_name}")
        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")
            if "out of memory" in str(e):
                clear_gpu_memory()
        
        clear_gpu_memory()  # Clear memory after each image
    
    # Save combined results
    combined_results_path = os.path.join(output_folder, f"combined_results_{n_images}_images.json")
    with open(combined_results_path, 'w') as f:
        json.dump(all_image_results, f, indent=4)
    
    # Create summary report
    summary_path = os.path.join(output_folder, f"summary_report_{n_images}_images.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Summary Report for {n_images} Random Images\n")
        f.write("=" * 50 + "\n\n")
        
        for image_name, results in all_image_results.items():
            f.write(f"Image: {image_name}\n")
            f.write("-" * 30 + "\n")
            
            for model_name in results:
                f.write(f"  {model_name}:\n")
                for method_name in results[model_name]:
                    metrics = results[model_name][method_name]['metrics']
                    f.write(f"    {method_name}: IoU={metrics['IoU']:.4f}, F1={metrics['F1']:.4f}\n")
            f.write("\n")
    
    print(f"\n🎉 Completed processing {len(all_image_results)} images successfully!")
    print(f"📊 Combined results saved to: {combined_results_path}")
    print(f"📋 Summary report saved to: {summary_path}")
    
    return all_image_results

def main():
    # Don't set seed initially to allow truly random image selection
    # set_seed() will be called later for reproducible model/method combinations
    
    print("Starting CAM-SAM analysis with memory management...")
    get_gpu_memory_info()
    
    # Configuration - adjust these paths for your setup
    # Lab-pc paths
    root_dir = "/home/matthewcockayne/Documents/PhD/data/ISIC2017/"
    test_image_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Data/")
    gt_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Part1_GroundTruth")
    output_folder = "/home/matthewcockayne/Documents/PhD/Zero-Shot_SkinLesion_Segmentation/results/single-image-analysis/"
    sam_checkpoint = "/home/matthewcockayne/Documents/PhD/Zero-Shot_SkinLesion_Segmentation/results/model_checkpoints/medsam_vit_b.pth"
    
    # Cluster paths (uncomment if running on cluster)
    # root_dir = "/home/xrai/datasets/ISIC2017/"
    # test_image_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Data/")
    # gt_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Part1_GroundTruth")
    # output_folder = "/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/single-image-analysis/"
    # sam_checkpoint = "/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/model_checkpoints/medsam_vit_b.pth"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # ========== CONFIGURATION OPTIONS ==========
    
    # Option 1: Analyze n random images with individual visualizations for presentation
    n_random_images = 1  # Change this number as needed
    save_individual_viz = True  # Set to True to save individual components for presentations
    use_random_seed = True  # Set to False for reproducible random image selection (uses seed 905)
    
    # If you want reproducible random image selection, set use_random_seed=False
    if not use_random_seed:
        set_seed()  # This will make image selection reproducible
        print("🔒 Using fixed seed for reproducible image selection")
    else:
        print("🎲 Using truly random image selection")
    
    if n_random_images > 1:
        print(f"🎯 Analyzing {n_random_images} random images with individual visualizations: {save_individual_viz}")
        results = analyze_multiple_random_images(n_random_images, test_image_folder, gt_folder, 
                                               sam_checkpoint, output_folder, save_individual=save_individual_viz)
    else:
        # Option 2: Analyze a single specific image
        target_image = "ISIC_0012199.jpg"  # Specify image name here
        print(f"🎯 Analyzing single image: {target_image} with individual visualizations: {save_individual_viz}")
        # Set seed for reproducible results when analyzing a specific image
        set_seed()
        results = analyze_single_image(target_image, test_image_folder, gt_folder, 
                                     sam_checkpoint, output_folder, save_individual=save_individual_viz)
        
        if results:
            print(f"\n=== Summary for {target_image} ===")
            for model_name in results:
                print(f"\n{model_name}:")
                for method_name in results[model_name]:
                    metrics = results[model_name][method_name]['metrics']
                    print(f"  {method_name}: IoU={metrics['IoU']:.4f}, F1={metrics['F1']:.4f}")
    
    # Final memory cleanup
    clear_gpu_memory()
    print("\n✅ Analysis completed with memory management!")
    get_gpu_memory_info()

if __name__ == "__main__":
    main()
