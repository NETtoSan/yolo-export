import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, time, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import cv2
import numpy as np

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
time.sleep(1)


class CustomDetectionDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.img_files = []

        print(f"\n=== Dataset Cleanup Log ===")
        print(f"Image directory: {img_dir}")
        print(f"Annotation directory: {annotations_dir}\n")

        total_files = 0
        skip_reasons = {}  # dictionary to count skip reasons

        for f in os.listdir(img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_files += 1
                ann_path = os.path.join(annotations_dir, os.path.splitext(f)[0] + '.txt')

                reason = None
                if not os.path.exists(ann_path):
                    reason = "No label file"
                elif os.path.getsize(ann_path) == 0:
                    reason = "Empty label file"
                else:
                    try:
                        with open(ann_path, "r") as fh:
                            line = fh.readline().strip()
                        if not line:
                            reason = "Blank line in label file"
                        elif len(line.split()) < 5:
                            reason = f"Not enough values ({len(line.split())})"
                    except Exception as e:
                        reason = f"Error reading file: {e}"

                if reason:
                    print(f"[SKIPPED] {f} → {reason}")
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue

                # Passed all checks → valid
                self.img_files.append(f)

        print(f"\n✅ Loaded {len(self.img_files)} valid images out of {total_files} total\n")

        # Print skip summary
        if skip_reasons:
            print("Summary of skipped images:")
            for reason, count in skip_reasons.items():
                print(f" - {count} images skipped due to: {reason}")
        else:
            print("No images were skipped.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.annotations_dir, os.path.splitext(img_name)[0] + '.txt')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        with open(ann_path, 'r') as f:
            line = f.readline().strip()
        parts = line.split()
        label = int(parts[0])
        bbox = torch.tensor([float(x) for x in parts[1:5]])

        return image, torch.tensor(label, dtype=torch.long), bbox

# Simple CNN model
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

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate dataset and dataloader
# Adjust batch_size here to reduce the number of batches per epoch
train_dataset = CustomDetectionDataset('./yolov11/bottles/train/labels', './yolov11/bottles/train/images')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # <-- change batch_size here

val_dataset = CustomDetectionDataset('./yolov11/bottles/valid/labels', './yolov11/bottles/valid/images')
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)    # <-- and here for validation

# Instantiate model, loss, optimizer
model = SimpleDetectionModel(num_classes=1).to(device)
criterion_cls = nn.BCEWithLogitsLoss()  # For binary classification, use BCEWithLogitsLoss if you prefer logits
# If you have multiple classes, use CrossEntropyLoss and adjust num_classes accordingly
#criterion_cls = nn.CrossEntropyLoss()  # For multi-class classification

# Show dataset contents before training
import random
def check_dataset_with_labels(dataset, num_samples=5, name="Dataset"):
    print(f"\n🔍 Checking {name}...")
    print(f"Number of images: {len(dataset)}\n")

    # pick random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        image, label, bbox = dataset[idx]

        # convert tensor -> numpy -> HWC in uint8 format
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert RGB → BGR for OpenCV

        # bbox is [x_center, y_center, w, h] in YOLO format (normalized 0–1)
        h, w = img_np.shape[:2]
        x_c, y_c, bw, bh = bbox.numpy()
        x_min = int((x_c - bw / 2) * w)
        y_min = int((y_c - bh / 2) * h)
        x_max = int((x_c + bw / 2) * w)
        y_max = int((y_c + bh / 2) * h)

        # Draw rectangle
        cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        # Put label text
        cv2.putText(img_np, f"Label: {label.item()}", (x_min, max(y_min-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show image for 500 ms
        cv2.imshow(f"{name} Sample #{idx}", img_np)
        cv2.waitKey(500)
        cv2.destroyAllWindows()


# Run dataset checks before training
check_dataset_with_labels(train_dataset, name="Training Dataset")
check_dataset_with_labels(val_dataset, name="Validation Dataset")


criterion_bbox = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def compute_iou(box1, box2):
    # box: [x, y, w, h] normalized
    # Convert to [x1, y1, x2, y2]
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2

    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def compute_map50(model, dataloader, device=torch.device('cpu')):
    model.eval()
    all_true = 0
    all_pred = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (images, labels, bboxes) in enumerate(dataloader, 1):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            class_logits, bbox_preds = model(images)
            probs = torch.sigmoid(class_logits).squeeze(1)  # shape [B]
            pred_labels = (probs > 0.5).long()              # 0 or 1

            for j in range(images.size(0)):
                if pred_labels[j] == 1:                    # predicted positive
                    all_pred += 1
                    iou = compute_iou(bbox_preds[j].cpu(), bboxes[j].cpu())
                    if pred_labels[j] == labels[j] and iou > 0.5:
                        correct += 1
                all_true += 1

            sys.stdout.write(f"\rmAP50: [{batch_idx}/{len(dataloader)}] batches processed")
            sys.stdout.flush()

        print()

    precision = correct / all_pred if all_pred > 0 else 0
    recall = correct / all_true if all_true > 0 else 0
    map50 = precision
    return map50


def evaluate(model, dataloader, criterion_cls, criterion_bbox, device):
    model.eval()
    total_loss = 0.0
    i = 0
    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            class_logits, bbox_preds = model(images)
            labels = labels.float().unsqueeze(1)  # Ensure labels are 2D for BCEWithLogitsLoss
            loss_cls = criterion_cls(class_logits, labels)
            loss_bbox = criterion_bbox(bbox_preds, bboxes)
            loss = loss_cls + loss_bbox
            total_loss += loss.item() * images.size(0)

            i += 1
            sys.stdout.write(f"\rEvaluating batch [{i}/{len(dataloader)}]: Loss: {loss.item():.4f}")
            sys.stdout.flush()
        print()

    avg_loss = total_loss / len(dataloader.dataset)
    map50 = compute_map50(model, dataloader, device)
    return avg_loss, map50

def visualize_predictions(model, dataset, device, score_thresh=0.5, num_samples=5):
    """
    Visualize predictions on random samples from a dataset.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        image, label, bbox_gt = dataset[idx]

        # Move image to device and add batch dimension
        img_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            class_logits, bbox_pred = model(img_tensor)
            score = torch.sigmoid(class_logits)[0].item()
            pred_label = 1 if score > score_thresh else 0
            bbox_pred = bbox_pred[0].cpu().numpy()

        # Convert tensor -> numpy -> HWC in uint8
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        h, w = img_np.shape[:2]

        # Ground truth box
        x_c, y_c, bw, bh = bbox_gt.numpy()
        x_min_gt = int((x_c - bw / 2) * w)
        y_min_gt = int((y_c - bh / 2) * h)
        x_max_gt = int((x_c + bw / 2) * w)
        y_max_gt = int((y_c + bh / 2) * h)
        cv2.rectangle(img_np, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 255, 0), 2)
        cv2.putText(img_np, f"GT: {label.item()}", (x_min_gt, max(y_min_gt-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Predicted box
        x_c, y_c, bw, bh = bbox_pred
        x_min = int((x_c - bw / 2) * w)
        y_min = int((y_c - bh / 2) * h)
        x_max = int((x_c + bw / 2) * w)
        y_max = int((y_c + bh / 2) * h)
        cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(img_np, f"Pred: {pred_label} ({score:.2f})", (x_min, max(y_min-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Show the image
        cv2.imshow(f"Validation Sample", img_np)
        cv2.waitKey(250)
    cv2.destroyAllWindows()
        

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batchsize = len(train_loader)
    for batch_idx, (images, labels, bboxes) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)

        class_logits, bbox_preds = model(images)
        optimizer.zero_grad()
        labels = labels.float().unsqueeze(1).to(device)  # Ensure labels are 2D for BCEWithLogitsLoss
        loss_cls = criterion_cls(class_logits, labels)
        loss_bbox = criterion_bbox(bbox_preds, bboxes)
        loss = loss_cls + loss_bbox
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        sys.stdout.write(
            f"\rEpoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{batchsize}] Loss: {loss.item():.4f}"
        )
        sys.stdout.flush()

    print("\nValidating...")

    epoch_loss = running_loss / len(train_dataset)
    val_loss, val_map50 = evaluate(model, val_loader, criterion_cls, criterion_bbox, device)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {epoch_loss:.4f} | "
          f"Val Loss: {val_loss:.4f}, mAP@0.5: {val_map50:.4f} | "
          f"Loss_bbox: {loss_bbox.item():.4f}\n")
    
    visualize_predictions(model, val_dataset, device, num_samples=30)

# Save model
torch.save(model.state_dict(), 'custom_detection_model.pt')