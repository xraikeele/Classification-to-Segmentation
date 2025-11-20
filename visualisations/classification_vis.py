import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["resnet50", "efficientnet_b0", "mobilenet_v2", "vit", "swin"]
checkpoint_accuracy = [ 0.8333, 0.8267, 0.8350, 0.8100, 0.8367]
checkpoint_loss = [0.8966, 0.9857, 0.8087, 0.8365, 0.7517]
zero_shot_accuracy = [0.7950, 0.6617, 0.6917, 0.2017, 0.4383]
zero_shot_loss = [0.6268, 0.6380, 0.6591, 0.8892, 0.7131]

# Bar width
bar_width = 0.35

# X-axis positions
x = np.arange(len(models))

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Accuracy subplot
axs[0].bar(x - bar_width/2, checkpoint_accuracy, bar_width, label='Checkpoint Accuracy', color='blue')
axs[0].bar(x + bar_width/2, zero_shot_accuracy, bar_width, label='Zero-Shot Accuracy', color='orange')
axs[0].set_title('Model Test Accuracy')
axs[0].set_xlabel('Models')
axs[0].set_ylabel('Accuracy')
axs[0].set_xticks(x)
axs[0].set_xticklabels(models)
axs[0].legend()

# Loss subplot
axs[1].bar(x - bar_width/2, checkpoint_loss, bar_width, label='Checkpoint Loss', color='blue')
axs[1].bar(x + bar_width/2, zero_shot_loss, bar_width, label='Zero-Shot Loss', color='orange')
axs[1].set_title('Model Test Loss')
axs[1].set_xlabel('Models')
axs[1].set_ylabel('Loss')
axs[1].set_xticks(x)
axs[1].set_xticklabels(models)
axs[1].legend()

# Layout and save
plt.tight_layout()
# Lab PC
#plt.savefig('/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/classification/model_performance.png')
# Cluster
plt.savefig('/home/csc29/projects/Zero-Shot_SkinLesion_Segmentation/results/classification/model_performance.png')
plt.show()