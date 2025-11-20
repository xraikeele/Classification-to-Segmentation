import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from glob import glob
import numpy as np
import random
import cv2 as cv
from PIL import Image
import json
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, normalize, resize
from torchvision.models import resnet18
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
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

def calculate_metrics(predicted, ground_truth):
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
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask

def transform_image(image_path):
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

def evaluate_model_on_test_set(seg_predictor, test_image_folder, gt_folder, output_folder):
    results = []
    total_metrics = {"Precision": 0, "Recall": 0, "F1": 0, "IoU": 0}
    num_images = 0
    
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
        
        seg_predictor.set_image(transformed_image.numpy()) 
        sampled_points = np.array([[112, 112]])
        input_labels = np.ones(sampled_points.shape[0])
        predicted_mask, _, _ = seg_predictor.predict(point_coords=sampled_points, point_labels=input_labels, multimask_output=False)
        
        predicted_mask = post_process_mask(predicted_mask[0])
        metrics = calculate_metrics(predicted_mask, ground_truth)
        results.append((image_name, metrics))
        
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_images += 1
        
        output_image_path = os.path.join(output_folder, f"{image_name}_result.png")
        cv.imwrite(output_image_path, predicted_mask)
        
        print(f"Processed {num_images} images")
    
    average_metrics = {key: total_metrics[key] / num_images for key in total_metrics}
    return results, average_metrics

def main():
    set_seed()
    root_dir = "/home/matthewcockayne/Documents/PhD/data/ISIC_2017/"
    test_image_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Data/")
    gt_folder = os.path.join(root_dir, "test/ISIC-2017_Test_v2_Part1_GroundTruth")
    output_folder = "/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/SAM/"
    os.makedirs(output_folder, exist_ok=True)
    
    sam_model = "vit_h"  
    checkpoint = "./results/model_checkpoints/sam_vit_h_4b8939.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[sam_model](checkpoint=checkpoint)  
    sam.to(device)
    sam_predictor = SamPredictor(sam)

    results, avg_metrics = evaluate_model_on_test_set(sam_predictor, test_image_folder, gt_folder, output_folder)
    print("Final Results:", avg_metrics)
    
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
if __name__ == "__main__":
    main()
