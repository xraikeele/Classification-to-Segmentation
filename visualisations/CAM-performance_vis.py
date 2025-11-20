import pandas as pd
import matplotlib.pyplot as plt
import os

# Models and file paths
models = ['resnet50', 'ViT', 'swin', 'mobilenet_v2', 'efficientnet_b0']
base_path = '/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/CAM'

# Dictionary to store average IoU scores
average_iou_scores = {}

# Read and process each model's IoU scores
for model in models:
    file_path = os.path.join(base_path, model, f'iou_scores_{model}.csv')
    data = pd.read_csv(file_path)
    avg_scores = data.groupby('Method')['IoU Score'].mean()
    average_iou_scores[model] = avg_scores

# Combine data into a single DataFrame
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