import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomDetectionDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.annotations_dir, os.path.splitext(img_name)[0] + '.txt')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        # Read annotation (YOLO format: class x y w h)
        try:
            with open(ann_path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    raise ValueError("Empty annotation file")
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError("Annotation file does not have enough values")
                label = int(parts[0])
                bbox = torch.tensor([float(x) for x in parts[1:5]])  # [x, y, w, h] normalized
        except Exception as e:
            # If annotation is missing or invalid, return a dummy label and bbox
            label = 0
            bbox = torch.tensor([0.5, 0.5, 0.1, 0.1])
        return image, torch.tensor(label), bbox

# Simple CNN model for detection (for illustration)
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

def compute_map50(model, dataloader, device='cpu'):
    model.eval()
    all_true = 0
    all_pred = 0
    correct = 0
    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            class_logits, bbox_preds = model(images)
            probs = torch.softmax(class_logits, dim=1)
            scores, pred_labels = torch.max(probs, dim=1)
            for i in range(images.size(0)):
                if scores[i] > 0.5:
                    all_pred += 1
                    iou = compute_iou(bbox_preds[i].cpu(), bboxes[i].cpu())
                    if pred_labels[i] == labels[i] and iou > 0.5:
                        correct += 1
                all_true += 1
    model.train()
    precision = correct / all_pred if all_pred > 0 else 0
    recall = correct / all_true if all_true > 0 else 0
    map50 = precision  # For single prediction per image, mAP@0.5 ≈ precision
    return map50

# Instantiate dataset and dataloader
# Adjust batch_size here to reduce the number of batches per epoch
train_dataset = CustomDetectionDataset('./yolov8/bottle/train/labels', './yolov8/bottle/train/images')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # <-- change batch_size here

val_dataset = CustomDetectionDataset('./yolov8/bottle/val/labels', './yolov8/bottle/valid/images')
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)    # <-- and here for validation

# Instantiate model, loss, optimizer
model = SimpleDetectionModel(num_classes=2)
criterion_cls = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    num_batches = len(train_loader)
    print(f"Epoch {epoch} starting... ({num_batches} batches)")
    for batch_idx, (images, labels, bboxes) in enumerate(train_loader):
        optimizer.zero_grad()
        class_logits, bbox_preds = model(images)
        loss_cls = criterion_cls(class_logits, labels)
        loss_bbox = criterion_bbox(bbox_preds, bboxes)
        loss = loss_cls + loss_bbox
        loss.backward()
        optimizer.step()
        # Compute mAP@0.5 for current batch
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(class_logits, dim=1)
            scores, pred_labels = torch.max(probs, dim=1)
            correct = 0
            all_pred = 0
            for i in range(images.size(0)):
                if scores[i] > 0.5:
                    all_pred += 1
                    iou = compute_iou(bbox_preds[i].cpu(), bboxes[i].cpu())
                    if pred_labels[i] == labels[i] and iou > 0.5:
                        correct += 1
            batch_map50 = correct / all_pred if all_pred > 0 else 0
        model.train()
        print(f"  Batch {batch_idx}: Loss {loss.item():.4f} mAP@0.5 {batch_map50:.4f}")
    map50 = compute_map50(model, train_loader)
    print(f"Epoch {epoch}: Final Loss {loss.item():.4f} mAP@0.5 {map50:.4f}")

# Validation after training
model.eval()
val_loss = 0
with torch.no_grad():
    for images, labels, bboxes in val_loader:
        class_logits, bbox_preds = model(images)
        loss_cls = criterion_cls(class_logits, labels)
        loss_bbox = criterion_bbox(bbox_preds, bboxes)
        loss = loss_cls + loss_bbox
        val_loss += loss.item() * images.size(0)
val_loss /= len(val_dataset)
val_map50 = compute_map50(model, val_loader)
print(f"Validation Loss: {val_loss:.4f} Validation mAP@0.5: {val_map50:.4f}")

# Save model
torch.save(model.state_dict(), 'custom_detection_model.pt')