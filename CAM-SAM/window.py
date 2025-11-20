import cv2
import numpy as np

# Global variables
drawing = False
ix, iy, fx, fy = -1, -1, -1, -1
image = None
overlay = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, image, overlay

    if event == cv2.EVENT_LBUTTONDOWN:  # Mouse click: start drawing
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse drag: update rectangle
        if drawing:
            temp_image = overlay.copy()
            cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw Bounding Box", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:  # Mouse release: finalize rectangle
        drawing = False
        fx, fy = x, y
        cv2.rectangle(image, (ix, iy), (fx, fy), (0, 255, 0), 2)
        cv2.imshow("Draw Bounding Box", image)
        print(f"Bounding Box Coordinates: {ix, iy, fx, fy}")

# Load an example image
image = cv2.imread("/home/matthewcockayne/Documents/PhD/data/ISIC_2017/test/ISIC-2017_Test_v2_Data/ISIC_0012092.jpg")
if image is None:
    raise FileNotFoundError("Image file not found. Replace 'example.jpg' with a valid path.")

# Simulate an initial segmentation mask overlay
mask = np.zeros_like(image, dtype=np.uint8)
mask[:, :, 1] = 100  # Green tint for visualization
overlay = cv2.addWeighted(image, 0.6, mask, 0.4, 0)

cv2.imshow("Draw Bounding Box", overlay)
cv2.setMouseCallback("Draw Bounding Box", draw_rectangle)

cv2.waitKey(0)
cv2.destroyAllWindows()