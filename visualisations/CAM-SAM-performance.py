import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Models and file paths
models = ['resnet50', 'ViT', 'swin', 'mobilenet_v2', 'efficientnet_b0']
base_path = '/home/matthewcockayne/Documents/PhD/zero-shot-segmentation-results/CAM-SAM-finetuned'

# Dictionary to store average IoU scores
average_iou_scores = {}

# Read and process each model's IoU scores
for model in models:
    file_path = os.path.join(base_path, model, 'average_results.json')
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract IoU scores for each method
        avg_scores = {method: metrics['IoU'] for method, metrics in data.items()}
        average_iou_scores[model] = avg_scores
    else:
        print(f"Warning: File not found for model {model}")

# Convert to DataFrame
average_iou_df = pd.DataFrame(average_iou_scores)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
average_iou_df.T.plot(kind='bar', ax=ax)

# Customization
ax.set_title('Average IoU Scores by Method for Each Model', fontsize=14)
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Average IoU Score', fontsize=12)
ax.legend(title='Methods', fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()

# Save and show plot
output_path = os.path.join(base_path, 'average_iou_scores.png')
plt.savefig(output_path)
plt.show()
