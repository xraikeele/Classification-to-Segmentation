import json

# Load JSON from a file
file_path = "/home/matthewcockayne/Documents/PhD/Zero-Shot_SkinLesion_Segmentation/results/experiment_results.json"  # Change this to your actual file path

with open(file_path, "r") as file:
    data = json.load(file)

# Initialize sums and count
precision_sum = recall_sum = f1_sum = iou_sum = 0.0
count = len(data)

# Loop through data to accumulate values
for _, metrics in data:
    precision_sum += metrics["Precision"]
    recall_sum += metrics["Recall"]
    f1_sum += metrics["F1"]
    iou_sum += metrics["IoU"]

# Compute averages
avg_precision = precision_sum / count
avg_recall = recall_sum / count
avg_f1 = f1_sum / count
avg_iou = iou_sum / count

# Print results
print(f"Average Precision: {avg_precision:.6f}")
print(f"Average Recall: {avg_recall:.6f}")
print(f"Average F1 Score: {avg_f1:.6f}")
print(f"Average IoU: {avg_iou:.6f}")