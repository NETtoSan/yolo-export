import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import cv2
import numpy as np

img_loss = 0
class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.missing_labels = []
        self.cleaned_files = []
        for f in self.img_files:
            label_path = os.path.join(self.label_dir, os.path.splitext(f)[0] + '.txt')
            if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                self.missing_labels.append(f)
                # Mark image file for exclusion if label is missing/empty
                self.cleaned_files.append(f)
        # Remove marked files from img_files (do not delete from disk)
        self.img_files = [f for f in self.img_files if f not in self.cleaned_files]
        print(f"Total images after cleanup: {len(self.img_files)}")
        print(f"Images excluded due to missing/empty label files: {len(self.cleaned_files)}")
        if self.cleaned_files:
            print("Sample excluded files:")
            for fname in self.cleaned_files[:10]:
                print(f" - {fname}")
        else:
            print("No files excluded.")
        print()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                line = f.readline().strip().split()
            # Treat any object as label=1 for training
            #print(int(line[0]))
            label = 1
            bbox = torch.tensor([float(x) for x in line[1:5]], dtype=torch.float32)
        else:
            print(f"Warning: Missing or empty label file for {img_name}. Using default values.")
            label = 0
            bbox = torch.zeros(4)
        return image, torch.tensor(label, dtype=torch.float32), bbox

class SimpleYoloNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(256, 1)      # Objectness score
        # Improved bounding box regressor: small MLP + sigmoid
        self.bbox_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()  # constrain outputs to [0, 1] for normalized bbox
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        class_logits = self.classifier(x)
        bbox = self.bbox_regressor(x)
        return class_logits, bbox

def validate(model, dataloader, criterion_cls, criterion_bbox, device):
    model.eval()
    total_loss = 0.0
    batchsize = len(dataloader)
    with torch.no_grad():
        for batch_idx, (images, labels, bboxes) in enumerate(dataloader):
            images, labels, bboxes = images.to(device), labels.to(device).unsqueeze(1), bboxes.to(device)
            class_logits, bbox_preds = model(images)
            loss_cls = criterion_cls(class_logits, labels)
            loss_bbox = criterion_bbox(bbox_preds, bboxes)
            loss = loss_cls + 0.5 * loss_bbox
            total_loss += loss.item() * images.size(0)
            sys.stdout.write(
                f"\rValidating Batch [{batch_idx+1}/{batchsize}] Loss: {loss.item():.4f} | loss_cls: {loss_cls.item():.4f}, loss_bbox: {loss_bbox.item():.4f}"
            )
            sys.stdout.flush()
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"\nValidation Loss: {avg_loss:.4f}")
    return avg_loss

def calculate_map(model, dataloader, device, iou_threshold=0.5, score_thresh=0.5):
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_gt = 0
    batchsize = len(dataloader)
    with torch.no_grad():
        for batch_idx, (images, labels, bboxes) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            class_logits, bbox_preds = model(images)
            scores = torch.sigmoid(class_logits).squeeze(1)
            pred_labels = (scores >= score_thresh).long()
            for i in range(images.size(0)):
                gt_label = int(labels[i].item())
                pred_label = int(pred_labels[i].item())
                gt_bbox = bboxes[i].cpu().numpy()
                pred_bbox = bbox_preds[i].cpu().numpy()
                # Calculate IoU
                x1_gt = gt_bbox[0] - gt_bbox[2] / 2
                y1_gt = gt_bbox[1] - gt_bbox[3] / 2
                x2_gt = gt_bbox[0] + gt_bbox[2] / 2
                y2_gt = gt_bbox[1] + gt_bbox[3] / 2
                x1_pred = pred_bbox[0] - pred_bbox[2] / 2
                y1_pred = pred_bbox[1] - pred_bbox[3] / 2
                x2_pred = pred_bbox[0] + pred_bbox[2] / 2
                y2_pred = pred_bbox[1] + pred_bbox[3] / 2
                xi1 = max(x1_gt, x1_pred)
                yi1 = max(y1_gt, y1_pred)
                xi2 = min(x2_gt, x2_pred)
                yi2 = min(y2_gt, y2_pred)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box1_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
                box2_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
                union_area = box1_area + box2_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if gt_label == 1:
                    total_gt += 1
                    if pred_label == 1 and iou >= iou_threshold:
                        true_positives += 1
                    elif pred_label == 1 and iou < iou_threshold:
                        false_positives += 1
                    elif pred_label == 0:
                        false_negatives += 1
                elif gt_label == 0 and pred_label == 1:
                    false_positives += 1
            sys.stdout.write(f"\rCalculating mAP: Batch [{batch_idx+1}/{batchsize}]")
            sys.stdout.flush()
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    print(f"\nmAP@{iou_threshold}: Precision={precision:.4f}, Recall={recall:.4f}, TP={true_positives}, FP={false_positives}, FN={false_negatives}")
    return precision, recall

def detect_and_visualize(model, dataset, device, num_images=5, score_thresh=0.5):
    model.eval()
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    correct = 0
    total = 0
    for idx in indices:
        image, label, bbox_gt = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            class_logits, bbox_pred = model(img_tensor)
            score = torch.sigmoid(class_logits).squeeze().item()
            pred_label = 1 if score >= score_thresh else 0  # Fix threshold logic
        total += 1
        is_correct = pred_label == int(label.item())
        if is_correct:
            correct += 1
        print(f"Image {idx}: GT label={int(label.item())}, Pred label={pred_label}, Score={score:.4f}, Correct={is_correct}")
        # Convert tensor to numpy image
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_np.shape[:2]
        # Draw ground truth box in green
        x_c, y_c, bw, bh = bbox_gt.numpy()
        x_min_gt = int((x_c - bw / 2) * w)
        y_min_gt = int((y_c - bh / 2) * h)
        x_max_gt = int((x_c + bw / 2) * w)
        y_max_gt = int((y_c + bh / 2) * h)
        cv2.rectangle(img_np, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 255, 0), 2)
        cv2.putText(img_np, f"GT: {label.item()}", (x_min_gt, max(y_min_gt-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Draw predicted box in red if above threshold
        if score >= score_thresh:
            x_c, y_c, bw, bh = bbox_pred.squeeze().cpu().numpy()
            x_min = int((x_c - bw / 2) * w)
            y_min = int((y_c - bh / 2) * h)
            x_max = int((x_c + bw / 2) * w)
            y_max = int((y_c + bh / 2) * h)
            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(img_np, f"Pred: {pred_label} ({score:.2f})", (x_min, max(y_min-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("Detection", img_np)
        cv2.waitKey(50)
    print(f"Detection accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)\nClosing window....\n")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Paths
    img_dir = './yolov11/bottlesv11/train/images'
    label_dir = './yolov11/bottlesv11/train/labels'
    val_img_dir = './yolov11/bottlesv11/valid/images'
    val_label_dir = './yolov11/bottlesv11/valid/labels'

    # Dataset and DataLoader
    train_dataset = YoloDataset(img_dir, label_dir)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataset = YoloDataset(val_img_dir, val_label_dir)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Model, Loss, Optimizer
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA for training.')
    else:
        device = torch.device('cpu')
        print('CUDA not available. Using CPU for training.')
    model = SimpleYoloNet().to(device)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_bbox = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    try:
        for epoch in range(10):
            model.train()
            batchsize = len(train_loader)
            img_loss = len(train_dataset.missing_labels)
            for batch_idx, (images, labels, bboxes) in enumerate(train_loader):
                images, labels, bboxes = images.to(device), labels.to(device).unsqueeze(1), bboxes.to(device)
                class_logits, bbox_preds = model(images)
                loss_cls = criterion_cls(class_logits, labels)
                loss_bbox = criterion_bbox(bbox_preds, bboxes)
                loss = loss_cls + 0.5 * loss_bbox
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sys.stdout.write(
                    f"\rEpoch [{epoch+1}/10] Batch [{batch_idx+1}/{batchsize}] [Missing labels: {img_loss}] Loss: {loss.item():.4f} | loss_cls: {loss_cls.item():.4f}, loss_bbox: {loss_bbox.item():.4f}"
                )
                sys.stdout.flush()
            print(f"\nEpoch {epoch+1} Loss: {loss.item():.4f}")
            # Calculate mAP before validation
            print(f"Calculating mAP on validation set after epoch {epoch+1}...")
            calculate_map(model, val_loader, device)
            validate(model, val_loader, criterion_cls, criterion_bbox, device)
            print(f"Detecting 30 images from validation set after epoch {epoch+1}...")
            detect_and_visualize(model, val_dataset, device, num_images=30)
        # Save model
        torch.save(model.state_dict(), 'simple_yolo_model.pt')
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting cleanly...")