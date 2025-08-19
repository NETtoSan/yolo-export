import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import time

# Import the model definition
import torch.nn as nn

class SimpleDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(32, num_classes)
        self.bbox_regressor = nn.Linear(32, 4)  # [x, y, w, h]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        class_logits = self.classifier(x)
        bbox = self.bbox_regressor(x)
        return class_logits, bbox

# Load the model
model = SimpleDetectionModel(num_classes=2)
model.load_state_dict(torch.load('./custom_detection_model.pt', map_location='cpu'))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

def detect_objects(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        class_logits, bbox = model(input_tensor)
        probs = torch.softmax(class_logits, dim=1)
        score, label = torch.max(probs, dim=1)
        box = bbox[0]

    results = []
    if score.item() > 0.5:  # Confidence threshold
        results.append({
            'box': box.tolist(),
            'label': int(label.item()),
            'score': float(score.item())
        })
    return results

if __name__ == "__main__":
    images_dir = './yolov8/bottle/train/images'
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, filename)
            detections = detect_objects(image_path)
            print(f"{filename}: {detections}")

            # Show image and draw bounding box
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image {image_path}")
                continue
            h, w = image.shape[:2]
            for det in detections:
                x, y, bw, bh = det['box']
                # Convert normalized [x, y, w, h] to pixel coordinates
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Label:{det['label']} Score:{det['score']:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imshow('Detection', image)
            cv2.waitKey(1)
            time.sleep(0.5)
    cv2.destroyAllWindows()