"""
Demo Sample Preparation Script

This script prepares 5 diverse test examples from the ISIC 2017 dataset
for use in the zero-shot segmentation demo notebook. It selects examples
that showcase different characteristics (good/poor segmentation, melanoma/benign).

Usage:
    python demo_samples.py
    
Output:
    - Copies 5 images and their corresponding masks to demo_data/
    - Creates demo_info.csv with metadata about each sample
"""

import os
import shutil
import pandas as pd
from PIL import Image
import numpy as np

def calculate_lesion_properties(mask_path):
    """Calculate basic properties of the lesion from the mask."""
    mask = np.array(Image.open(mask_path).convert("L"))
    mask_binary = (mask > 128).astype(np.uint8)
    
    lesion_area = np.sum(mask_binary)
    total_area = mask_binary.size
    lesion_ratio = lesion_area / total_area
    
    # Calculate compactness (circularity)
    if lesion_area > 0:
        contours = np.sum(mask_binary[:-1, :] != mask_binary[1:, :]) + \
                   np.sum(mask_binary[:, :-1] != mask_binary[:, 1:])
        compactness = (4 * np.pi * lesion_area) / (contours ** 2) if contours > 0 else 0
    else:
        compactness = 0
    
    return {
        'lesion_ratio': lesion_ratio,
        'compactness': compactness,
        'area_pixels': lesion_area
    }

def select_diverse_samples(root_dir, num_samples=5):
    """
    Select specific pre-defined samples from test set.
    """
    test_label_csv = os.path.join(root_dir, "test", "ISIC-2017_Test_v2_Part3_GroundTruth.csv")
    test_mask_dir = os.path.join(root_dir, "test", "ISIC-2017_Test_v2_Part1_GroundTruth")
    
    # Load labels
    labels_df = pd.read_csv(test_label_csv, index_col=0)
    
    # Pre-defined sample IDs
    sample_ids = [
        "ISIC_0012223",
        "ISIC_0014941",
        "ISIC_0015273",
        "ISIC_0016066",
        "ISIC_0016070"
    ]
    
    print("Selecting pre-defined samples...")
    selected = []
    
    for img_id in sample_ids:
        if img_id not in labels_df.index:
            print(f"Warning: {img_id} not found in labels")
            continue
        
        # Get label
        label = labels_df.loc[img_id, "melanoma"]
        
        # Calculate mask properties
        mask_path = os.path.join(test_mask_dir, f"{img_id}_segmentation.png")
        if os.path.exists(mask_path):
            props = calculate_lesion_properties(mask_path)
        else:
            print(f"Warning: Mask not found for {img_id}")
            props = {'lesion_ratio': 0, 'compactness': 0, 'area_pixels': 0}
        
        selected.append({
            'image_id': img_id,
            'label': label,
            'lesion_ratio': props['lesion_ratio'],
            'compactness': props['compactness'],
            'area_pixels': props['area_pixels']
        })
    
    return pd.DataFrame(selected).head(num_samples)

def prepare_demo_data(root_dir, output_dir="demo_data", num_samples=5):
    """
    Prepare demo dataset by copying selected samples.
    
    Args:
        root_dir: Root directory of ISIC 2017 dataset
        output_dir: Output directory for demo samples
        num_samples: Number of samples to prepare
    """
    # Create output directories
    demo_images_dir = os.path.join(output_dir, "images")
    demo_masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(demo_images_dir, exist_ok=True)
    os.makedirs(demo_masks_dir, exist_ok=True)
    
    # Select diverse samples
    selected_samples = select_diverse_samples(root_dir, num_samples)
    
    test_image_dir = os.path.join(root_dir, "test", "ISIC-2017_Test_v2_Data")
    test_mask_dir = os.path.join(root_dir, "test", "ISIC-2017_Test_v2_Part1_GroundTruth")
    
    # Copy files and prepare metadata
    demo_info = []
    
    print(f"\nPreparing {num_samples} demo samples...")
    print("=" * 60)
    
    for idx, row in selected_samples.iterrows():
        img_id = row['image_id']
        
        # Source paths
        src_img = os.path.join(test_image_dir, f"{img_id}.jpg")
        src_mask = os.path.join(test_mask_dir, f"{img_id}_segmentation.png")
        
        # Destination paths
        dst_img = os.path.join(demo_images_dir, f"{img_id}.jpg")
        dst_mask = os.path.join(demo_masks_dir, f"{img_id}_segmentation.png")
        
        # Copy files
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)
        
        # Prepare metadata
        demo_info.append({
            'image_id': img_id,
            'image_path': dst_img,
            'mask_path': dst_mask,
            'label': 'Melanoma' if row['label'] == 1 else 'Benign',
            'label_binary': int(row['label']),
            'lesion_ratio': f"{row['lesion_ratio']:.3f}",
            'compactness': f"{row['compactness']:.3f}",
            'description': get_description(row)
        })
        
        print(f"✓ {img_id}: {demo_info[-1]['description']}")
    
    # Save metadata
    demo_info_df = pd.DataFrame(demo_info)
    demo_info_path = os.path.join(output_dir, "demo_info.csv")
    demo_info_df.to_csv(demo_info_path, index=False)
    
    print("=" * 60)
    print(f"\n✅ Demo dataset prepared successfully!")
    print(f"   Output directory: {os.path.abspath(output_dir)}")
    print(f"   Images: {demo_images_dir}")
    print(f"   Masks: {demo_masks_dir}")
    print(f"   Metadata: {demo_info_path}")
    print(f"\n📊 Label distribution:")
    print(f"   Melanoma: {sum(1 for x in demo_info if x['label'] == 'Melanoma')}")
    print(f"   Benign: {sum(1 for x in demo_info if x['label'] == 'Benign')}")
    
    return demo_info_df

def get_description(row):
    """Generate human-readable description for each sample."""
    label = "Melanoma" if row['label'] == 1 else "Benign"
    
    if row['lesion_ratio'] > 0.25:
        size = "large"
    elif row['lesion_ratio'] > 0.15:
        size = "medium"
    else:
        size = "small"
    
    if row['compactness'] > 0.6:
        shape = "compact/circular"
    elif row['compactness'] > 0.4:
        shape = "moderate shape"
    else:
        shape = "irregular shape"
    
    return f"{label}, {size} lesion, {shape}"

def load_demo_samples(demo_dir="demo_data"):
    """
    Load demo samples for use in notebook/script.
    
    Returns:
        List of dictionaries with image_path, mask_path, and metadata
    """
    demo_info_path = os.path.join(demo_dir, "demo_info.csv")
    
    if not os.path.exists(demo_info_path):
        raise FileNotFoundError(
            f"Demo info file not found at {demo_info_path}. "
            "Please run prepare_demo_data() first."
        )
    
    demo_info = pd.read_csv(demo_info_path)
    return demo_info.to_dict('records')

def main():
    """Main function to prepare demo dataset."""
    # Configuration
    ROOT_DIR = "/home/matthewcockayne/Documents/PhD/data/ISIC2017/"
    OUTPUT_DIR = os.path.join("notebooks", "demo_data")  # Create in notebooks directory
    NUM_SAMPLES = 5
    
    # Check if root directory exists
    if not os.path.exists(ROOT_DIR):
        print(f"❌ Error: ISIC 2017 dataset not found at {ROOT_DIR}")
        print("Please update ROOT_DIR in the script to point to your dataset.")
        return
    
    # Prepare demo data
    demo_info = prepare_demo_data(ROOT_DIR, OUTPUT_DIR, NUM_SAMPLES)
    
    # Display summary
    print("\n" + "=" * 60)
    print("DEMO SAMPLES SUMMARY")
    print("=" * 60)
    print(demo_info[['image_id', 'label', 'description']].to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    main()
