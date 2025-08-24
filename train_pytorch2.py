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
import time

def nms(boxes, scores, iou_threshold=0.5):
    # boxes: [N, 4] in xywh normalized
    # scores: [N]
    if boxes.size(0) == 0:
        return []
    # Convert xywh to xyxy
    xyxy = torch.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        ious = box_iou(xyxy[i].unsqueeze(0), xyxy[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def box_iou(box1, box2):
    # box1: [N, 4], box2: [M, 4] in xyxy
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / union_area
    return iou

img_loss = 0

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, max_objects=10, clean_missing_labels=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.max_objects = max_objects
        print(f"Total images: {len(self.img_files)}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        boxes = []
        labels = []
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append(int(parts[0]) + 1)
                        boxes.append([float(x) for x in parts[1:5]])

        # Pad to max_objects
        while len(boxes) < self.max_objects:
            boxes.append([0,0,0,0])
            labels.append(-1)  # -1 for "no object"

        boxes = torch.tensor(boxes[:self.max_objects], dtype=torch.float32)
        labels = torch.tensor(labels[:self.max_objects], dtype=torch.long)
        return image, labels, boxes

class SimpleYoloNet(nn.Module):
    def __init__(self, num_classes=2, max_objects=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.max_objects = max_objects
        self.num_classes = num_classes
        self.classifier = nn.Linear(512, num_classes * max_objects)
        self.bbox_regressor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * max_objects)
            #nn.Sigmoid()  # constrain outputs to [0, 1] for normalized bbox
            )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        class_logits = self.classifier(x).view(x.size(0), self.max_objects, self.num_classes)
        bbox = self.bbox_regressor(x).view(x.size(0), self.max_objects, 4)
        return class_logits, bbox

def validate(model, dataloader, criterion_cls, criterion_bbox, device):
    model.eval()
    total_loss = 0.0
    batchsize = len(dataloader)
    with torch.no_grad():
        for batch_idx, (images, labels, bboxes) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            class_logits, bbox_preds = model(images)
            # Flatten for loss
            class_logits_flat = class_logits.view(-1, class_logits.size(-1))
            labels_flat = labels.view(-1)
            bbox_preds_flat = bbox_preds.view(-1, 4)
            bboxes_flat = bboxes.view(-1, 4)
            obj_mask = labels_flat != -1
            if obj_mask.sum() > 0:
                loss_cls = criterion_cls(class_logits_flat[obj_mask], labels_flat[obj_mask])
                loss_bbox = criterion_bbox(bbox_preds_flat[obj_mask], bboxes_flat[obj_mask])
                loss = loss_cls + 2.0 * loss_bbox
                total_loss += loss.item() * images.size(0)
                bbox_min = bbox_preds.min().item()
                bbox_max = bbox_preds.max().item()
                bbox_mean = bbox_preds.mean().item()
                sys.stdout.write(
                    f"\rValidating Batch [{batch_idx+1}/{batchsize}] Loss: {loss.item():.4f} | loss_cls: {loss_cls.item():.4f}, loss_bbox: {loss_bbox.item():.4f} | bbox_out: min={bbox_min:.3f}, max={bbox_max:.3f}, mean={bbox_mean:.3f}"
                )
                sys.stdout.flush()
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"\n      >Validation Loss: {avg_loss:.4f}")
    #return avg_loss

def calculate_map(model, dataloader, device, iou_threshold=0.5, score_thresh=0.5):
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_gt = 0
    batchsize = len(dataloader) if hasattr(dataloader, '__len__') else None
    with torch.no_grad():
        for batch_idx, (images, labels, bboxes) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            class_logits, bbox_preds = model(images)
            probs = torch.softmax(class_logits, dim=2)
            for i in range(images.size(0)):
                gt_boxes = bboxes[i][labels[i] != -1]
                gt_labels = labels[i][labels[i] != -1]
                pred_boxes = bbox_preds[i]
                pred_probs = probs[i]
                pred_scores, pred_classes = torch.max(pred_probs, dim=1)
                # Filter predictions by score threshold
                keep = pred_scores >= score_thresh
                pred_boxes = pred_boxes[keep]
                pred_classes = pred_classes[keep]
                pred_scores = pred_scores[keep]
                # Apply NMS per class
                final_preds = []
                for cls in torch.unique(pred_classes):
                    cls_mask = pred_classes == cls
                    if cls_mask.sum() == 0:
                        continue
                    boxes_cls = pred_boxes[cls_mask]
                    scores_cls = pred_scores[cls_mask]
                    keep_idx = nms(boxes_cls, scores_cls, iou_threshold)
                    for k in keep_idx:
                        final_preds.append((boxes_cls[k], int(cls.item()), scores_cls[k].item()))
                matched_gt = set()
                matched_pred = set()
                # For each gt, find best matching pred
                for gt_idx, gt_box in enumerate(gt_boxes):
                    gt_label = gt_labels[gt_idx].item()
                    best_iou = 0
                    best_pred = -1
                    for pred_idx, (pred_box, pred_cls, pred_score) in enumerate(final_preds):
                        if pred_cls != gt_label:
                            continue
                        # IoU calculation
                        x1_gt = gt_box[0] - gt_box[2] / 2
                        y1_gt = gt_box[1] - gt_box[3] / 2
                        x2_gt = gt_box[0] + gt_box[2] / 2
                        y2_gt = gt_box[1] + gt_box[3] / 2
                        x1_pred = pred_box[0] - pred_box[2] / 2
                        y1_pred = pred_box[1] - pred_box[3] / 2
                        x2_pred = pred_box[0] + pred_box[2] / 2
                        y2_pred = pred_box[1] + pred_box[3] / 2
                        xi1 = max(x1_gt, x1_pred)
                        yi1 = max(y1_gt, y1_pred)
                        xi2 = min(x2_gt, x2_pred)
                        yi2 = min(y2_gt, y2_pred)
                        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                        box1_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
                        box2_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
                        union_area = box1_area + box2_area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        if iou > best_iou:
                            best_iou = iou
                            best_pred = pred_idx
                    total_gt += 1
                    if best_iou >= iou_threshold and best_pred != -1:
                        true_positives += 1
                        matched_gt.add(gt_idx)
                        matched_pred.add(best_pred)
                    else:
                        false_negatives += 1
                # Count unmatched preds as FP
                for pred_idx, (pred_box, pred_cls, pred_score) in enumerate(final_preds):
                    if pred_idx in matched_pred:
                        continue
                    false_positives += 1
            sys.stdout.write(f"\rCalculating mAP: Batch [{batch_idx+1}/{batchsize}]")
            sys.stdout.flush()
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    print(f"\nmAP@{iou_threshold}: Precision: {precision:.4f}, Recall: {recall:.4f}, TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
    #return precision, recall

def detect_and_visualize(model, dataset, device, num_images=5, score_thresh=0.3, epoch_num=None):
    model.eval()
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    correct = 0
    total = 0
    # Prepare output directory
    #out_dir = f'./output/epoch{epoch_num}' if epoch_num is not None else './output/epoch'
    #os.makedirs(out_dir, exist_ok=True)
    img_count = 0
    for idx in indices:
        img_count += 1
        image, gt_labels, gt_bboxes = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            class_logits, bbox_preds = model(img_tensor)
            probs = torch.softmax(class_logits, dim=2)[0]  # shape: [max_objects, num_classes]
            pred_scores, pred_labels = torch.max(probs, dim=1)  # shape: [max_objects]
            bbox_preds = bbox_preds[0]  # shape: [max_objects, 4]
        # Convert tensor to numpy image
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_np.shape[:2]
        # Draw ground truth boxes in green
        for obj_idx in range(gt_labels.shape[0]):
            label = gt_labels[obj_idx].item()
            if label == -1:
                continue
            bbox_gt = gt_bboxes[obj_idx].numpy()
            x_c, y_c, bw, bh = bbox_gt
            x_min_gt = int((x_c - bw / 2) * w)
            y_min_gt = int((y_c - bh / 2) * h)
            x_max_gt = int((x_c + bw / 2) * w)
            y_max_gt = int((y_c + bh / 2) * h)
            cv2.rectangle(img_np, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 255, 0), 2)
            cv2.putText(img_np, f"GT: {label}", (x_min_gt, max(y_min_gt-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Apply NMS per class for predictions
        nms_preds = []
        for cls in torch.unique(pred_labels):
            cls_mask = (pred_labels == cls) & (pred_scores >= score_thresh)
            if cls_mask.sum() == 0:
                continue
            boxes_cls = bbox_preds[cls_mask]
            scores_cls = pred_scores[cls_mask]
            keep_idx = nms(boxes_cls, scores_cls, iou_threshold=0.5)
            for k in keep_idx:
                nms_preds.append((boxes_cls[k].cpu().numpy(), int(cls.item()), scores_cls[k].item()))
        # Draw NMS-filtered predicted boxes in red
        detected = 0
        for bbox_pred, pred_label, score in nms_preds:
            x_c, y_c, bw, bh = bbox_pred
            x_min = int((x_c - bw / 2) * w)
            y_min = int((y_c - bh / 2) * h)
            x_max = int((x_c + bw / 2) * w)
            y_max = int((y_c + bh / 2) * h)
            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(img_np, f"Pred: {pred_label} ({score:.2f})", (x_min, max(y_min-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            detected += 1
        # For accuracy, count correct predictions (simple matching)
        # Here, we count a correct if any pred_label matches a gt_label
        total += gt_labels[gt_labels != -1].shape[0]
        correct += sum([1 for obj_idx in range(gt_labels.shape[0]) if gt_labels[obj_idx].item() in pred_labels.tolist() and gt_labels[obj_idx].item() != -1])
        sys.stdout.write(
            f"\rImage [{correct}|{img_count}] {idx}: Scores: {[round(s,2) for s in pred_scores.tolist()]}"
        )
        sys.stdout.flush()
        cv2.imshow("Detection", img_np)
        cv2.waitKey(50)
    print(f"\n      >Detection accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)\n")

def print_label_distribution(dataset, name="Dataset"):
        pos_count = 0
        neg_count = 0
        class_ids = set()
        for i in range(len(dataset)):
            _, labels, _ = dataset[i]
            for obj_idx in range(labels.shape[0]):
                label_int = int(labels[obj_idx].item())
                if label_int == -1:
                    continue
                if label_int == 0:
                    neg_count += 1
                else:
                    pos_count += 1
                    class_ids.add(label_int)
            sys.stdout.write(f"\rChecking [{i+1}/{len(dataset)}] ...")
            sys.stdout.flush()
        all_class_ids = set(class_ids)
        all_class_ids.add(0)  # Include negative class
        num_classes_incl_neg = len(all_class_ids)
        print(f"\n{name} label distribution: {pos_count} positive, {neg_count} negative samples.")
        print(f"        > Number of classes (including negatives): {num_classes_incl_neg}")
        print(f"        > Class IDs (all): {sorted(all_class_ids)}")
        print(f"        > Negative class ID: 0")
        print(f"        > Positive class IDs: {[cid for cid in sorted(all_class_ids) if cid != 0]}")
        return pos_count, neg_count, num_classes_incl_neg


# --- Ground Truth Bounding Box Visualization ---
def visualize_ground_truth(dataset, num_images=10):
    print(f"\nVisualizing {num_images} ground truth bounding boxes...")
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    for idx in indices:
        image, labels, bboxes = dataset[idx]
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_np.shape[:2]
        for obj_idx in range(labels.shape[0]):
            label = labels[obj_idx].item()
            if label == -1:
                continue
            bbox_gt = bboxes[obj_idx].numpy()
            x_c, y_c, bw, bh = bbox_gt
            x_min_gt = int((x_c - bw / 2) * w)
            y_min_gt = int((y_c - bh / 2) * h)
            x_max_gt = int((x_c + bw / 2) * w)
            y_max_gt = int((y_c + bh / 2) * h)
            cv2.rectangle(img_np, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 255, 0), 2)
            cv2.putText(img_np, f"GT: {label}", (x_min_gt, max(y_min_gt-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Ground Truth Bounding Boxes", img_np)
        cv2.waitKey(300)
    cv2.destroyAllWindows()
    print("Visualization complete.\n")

# Visualize ground truth bounding box centers
def visualize_bbox_centers(dataset):
    import matplotlib.pyplot as plt
    x_centers = []
    y_centers = []
    for i in range(len(dataset)):
        _, _, bbox = dataset[i]
        x_c, y_c, _, _ = bbox.numpy()
        x_centers.append(x_c)
        y_centers.append(y_c)

        sys.stdout.write(f"\rProcessing bbox center [{i+1}/{len(dataset)}]: x={x_c:.3f}, y={y_c:.3f}")
        sys.stdout.flush()

    print()
    '''
    plt.figure(figsize=(6,6))
    plt.scatter(x_centers, y_centers, alpha=0.5)
    plt.title('Ground Truth Bounding Box Centers')
    plt.xlabel('x_center (normalized)')
    plt.ylabel('y_center (normalized)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()
    '''


# Paths
img_dir = './yolov11/bottlesv11/train/images'
label_dir = './yolov11/bottlesv11/train/labels'
val_img_dir = './yolov11/bottlesv11/valid/images'
val_label_dir = './yolov11/bottlesv11/valid/labels'

# Dataset and DataLoader
max_objects = 10  # Set max objects per image
train_dataset = YoloDataset(img_dir, label_dir, max_objects=max_objects)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = YoloDataset(val_img_dir, val_label_dir, max_objects=max_objects)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model, Loss, Optimizer
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('[INFO] Using CUDA for training.')
else:
    device = torch.device('cpu')
    print('[INFO] Using CPU for training.')

# Dynamically get class distribution for weighting
num_pos, num_neg, num_classes = print_label_distribution(train_dataset, name="Training set")
if num_pos > 0:
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
else:
    pos_weight = torch.tensor([1.0], device=device)
print_label_distribution(val_dataset, name="Validation set")

model = SimpleYoloNet(num_classes=num_classes, max_objects=max_objects).to(device)
print("--------------------------"); print(model); print("--------------------------")
time.sleep(2)

criterion_cls = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion_bbox = nn.MSELoss()   #nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) #, weight_decay=1e-4)

# Visualize the data
#visualize_bbox_centers(train_dataset)
visualize_ground_truth(train_dataset, num_images=10)


# Train epoch
epoch_loop = 400
try:
    for epoch in range(epoch_loop):
        model.train()
        batchsize = len(train_loader)
        img_loss = 0  # No missing_labels attribute anymore
        
        for batch_idx, (images, labels, bboxes) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            class_logits, bbox_preds = model(images)
            # Flatten for loss
            class_logits_flat = class_logits.view(-1, class_logits.size(-1))
            labels_flat = labels.view(-1)
            bbox_preds_flat = bbox_preds.view(-1, 4)
            bboxes_flat = bboxes.view(-1, 4)
            obj_mask = labels_flat != -1
            if obj_mask.sum() > 0:

                loss_cls = criterion_cls(class_logits_flat[obj_mask], labels_flat[obj_mask])
                loss_bbox = criterion_bbox(bbox_preds_flat[obj_mask], bboxes_flat[obj_mask])
                loss = loss_cls + 5.0 * loss_bbox
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bbox_min = bbox_preds.min().item()
                bbox_max = bbox_preds.max().item()
                bbox_mean = bbox_preds.mean().item()
                sys.stdout.write(
                    f"\rEpoch [{epoch+1}/{epoch_loop}] Batch [{batch_idx+1}/{batchsize}] [Missing labels: {img_loss}] Loss: {loss.item():.4f} | loss_cls: {loss_cls.item():.4f}, loss_bbox: {loss_bbox.item():.4f}"
                )
                sys.stdout.flush()

        print(f"\nEpoch {epoch+1} Loss: {loss.item():.4f}")
        # Calculate mAP before validation
        print(f"Calculating mAP on validation set after epoch {epoch+1}...")
        calculate_map(model, val_loader, device)
        validate(model, val_loader, criterion_cls, criterion_bbox, device)

        print(f"Detecting 30 images from validation set after epoch {epoch+1}...")
        detect_and_visualize(model, val_dataset, device, num_images=30, epoch_num=epoch+1)
    # Save model
    torch.save(model.state_dict(), 'simple_custom_model.pt')
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Exiting cleanly...")