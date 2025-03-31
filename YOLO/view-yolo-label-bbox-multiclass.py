import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# === PATHS (modify these) ===
image_path = "/content/drive/MyDrive/filament-detection-project/data/YOLO-data/gong-processed-jpgs/2011/01/09/030101-20110109104734Ch.jpg"
label_path = "/content/drive/MyDrive/filament-detection-project/data/YOLO-data/labels/030101-20110109104734Ch.txt"

# === Load image ===
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")
img_height, img_width = image.shape[:2]

# === Read YOLO label ===
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Label file not found: {label_path}")

with open(label_path, "r") as f:
    lines = f.readlines()

# === Define distinct colors per class ===
def get_class_color(class_id):
    color_map = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
    ]
    return color_map[int(class_id) % len(color_map)]

# === Define class label map ===
class_labels = {
    0: "L",
    1: "R",
    2: "U"
}

# === Draw boxes ===
for line in lines:
    parts = line.strip().split()
    if len(parts) != 5:
        continue

    class_id, x_center, y_center, w, h = map(float, parts)
    class_id = int(class_id)

    # Convert normalized coordinates to absolute pixel values
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    color = get_class_color(class_id)
    label = class_labels.get(class_id, f"Class {class_id}")

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=6)

    # Compute text size
    font_scale = 1.5
    font_thickness = 3
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_origin = (x1, y1 - 12)

    # Draw filled rectangle for label background
    cv2.rectangle(image,
                  (text_origin[0], text_origin[1] - text_height - 4),
                  (text_origin[0] + text_width + 6, text_origin[1] + baseline),
                  color,
                  thickness=cv2.FILLED)

    # Draw label text on top
    cv2.putText(image,
                label,
                (text_origin[0] + 3, text_origin[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA)

# === Show result ===
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("YOLO Bounding Boxes with Labels")
plt.show()
