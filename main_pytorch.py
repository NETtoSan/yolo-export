import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

# ------------------------------
# Define your same model
# ------------------------------
class SimpleDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone: deeper with more channels
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # [B,32,H/2,W/2]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B,64,H/4,W/4]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [B,128,H/8,W/8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # [B,256,H/16,W/16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

        # BBox regression head
        self.bbox_regressor = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),  # automatically outputs 1x1 per channel
            nn.Flatten(),
            nn.Linear(128, 4),  # now input is 128, independent of H/W
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        class_logits = self.classifier(x)
        bbox = self.bbox_regressor(x)
        return class_logits, bbox

# ------------------------------
# Device & model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleDetectionModel(num_classes=1).to(device)
model.load_state_dict(torch.load("custom_detection_model.pt", map_location=device))
model.eval()

# ------------------------------
# Image folder & transforms
# ------------------------------
img_folder = "./yolov11/bottles/test/images"  # change this
transform = transforms.Compose([
    transforms.ToTensor()
])

# ------------------------------
# Run inference
# ------------------------------
for img_name in os.listdir(img_folder):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # Load image
    img_path = os.path.join(img_folder, img_name)
    pil_img = Image.open(img_path).convert("RGB")
    w, h = pil_img.size
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, bbox_pred = model(img_tensor)
        score = torch.sigmoid(class_logits).item()  # binary class score
        label = 1 if score > 0.5 else 0

    # Convert bbox to pixel coordinates
    x_c, y_c, bw, bh = bbox_pred[0].cpu().numpy()
    x_min = int((x_c - bw/2) * w)
    y_min = int((y_c - bh/2) * h)
    x_max = int((x_c + bw/2) * w)
    y_max = int((y_c + bh/2) * h)

    # Load image for OpenCV display
    img_cv = cv2.imread(img_path)
    cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(img_cv, f"Label: {label} | Score: {score:.2f}",
                (x_min, max(y_min-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show for 500ms
    cv2.imshow("Inference", img_cv)
    key = cv2.waitKey(500)
    if key == 27:  # ESC to exit early
        break

cv2.destroyAllWindows()
