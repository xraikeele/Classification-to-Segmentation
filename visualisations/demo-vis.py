import matplotlib.pyplot as plt
# Save visualization results
"""
def save_visualization(image, heatmap, sampled_points, mask, processed_mask, save_path):
    plt.figure(figsize=(10, 50))

    # Original image
    plt.subplot(1, 5, 1)
    plt.imshow(to_pil_image(image))
    plt.title('Original')

    # Heatmap overlay
    result = overlay_mask(to_pil_image(image), to_pil_image(heatmap, mode='F'), alpha=0.5)
    plt.subplot(1, 5, 2)
    plt.imshow(result)
    plt.title('CAM Overlay')
    plt.axis('off')

    # Sampled points
    plt.subplot(1, 5, 3)
    plt.imshow(to_pil_image(image))
    plt.scatter(sampled_points[:, 0], sampled_points[:, 1], color='red', marker='o')
    plt.title('Sampled Points')
    plt.axis('off')

    # Segmented mask
    plt.subplot(1, 5, 4)
    plt.imshow(to_pil_image(image))
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title('SAM Segmented')
    plt.axis('off')

    # Post-processed mask
    plt.subplot(1, 5, 5)
    plt.imshow(to_pil_image(image))
    plt.imshow(processed_mask, cmap='jet', alpha=0.5)
    plt.title('Post-Processed')
    plt.axis('off')

    plt.savefig(save_path)
    plt.close()"
"""