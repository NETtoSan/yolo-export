import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import cv2
import graphviz
import numpy as np

img_loss = 0
class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, clean_missing_labels=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.missing_labels = []
        self.cleaned_files = []
        if clean_missing_labels:
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
        else:
            print(f"Total images (no cleanup): {len(self.img_files)}")

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
            label = 1 #(int(line[0]))
            bbox = torch.tensor([float(x) for x in line[1:5]], dtype=torch.float32)
        else:
            #print(f"Warning: Missing or empty label file for {img_name}. Using default values.")
            label = 0
            bbox = torch.zeros(4)
        return image, torch.tensor(label, dtype=torch.float32), bbox

class SimpleYoloNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Dropout(0.3),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(0.3),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(512, 1)      # Objectness score
        
        # Improved bounding box regressor: deeper MLP + sigmoid
        self.bbox_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
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
            loss = loss_cls + 2.0 * loss_bbox  # Increased bbox loss weight
            total_loss += loss.item() * images.size(0)
            # Print bbox output stats for debugging
            bbox_min = bbox_preds.min().item()
            bbox_max = bbox_preds.max().item()
            bbox_mean = bbox_preds.mean().item()
            sys.stdout.write(
                f"\rValidating Batch [{batch_idx+1}/{batchsize}] Loss: {loss.item():.4f} | loss_cls: {loss_cls.item():.4f}, loss_bbox: {loss_bbox.item():.4f} | bbox_out: min={bbox_min:.3f}, max={bbox_max:.3f}, mean={bbox_mean:.3f}"
            )
            sys.stdout.flush()
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"\nValidation Loss: {avg_loss:.4f}")
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
            scores = torch.sigmoid(class_logits).squeeze(1)
            pred_labels = (scores >= score_thresh).long()
            if batch_idx == 0:
                print("\n[DEBUG] First batch logits:", class_logits.squeeze().detach().cpu().numpy())
                print("[DEBUG] First batch labels:", labels.squeeze().detach().cpu().numpy())
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
    #return precision, recall

def visualize_model_layers_and_weights(model, img_tensor):
    import torch.nn as nn
    import cv2
    import numpy as np
    feature_imgs = []
    feature_names = []
    overlay_imgs = []
    overlay_names = []
    with torch.no_grad():
        x = img_tensor
        # Prepare input image for overlay
        input_img = img_tensor.squeeze().cpu().numpy()
        if input_img.ndim == 3:
            input_img = input_img[:3]
            input_img = (input_img - input_img.min()) / (input_img.ptp() + 1e-5) * 255
            input_img = input_img.astype(np.uint8)
            input_img = np.transpose(input_img, (1,2,0))
        else:
            input_img = (input_img - input_img.min()) / (input_img.ptp() + 1e-5) * 255
            input_img = input_img.astype(np.uint8)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        # Save input image for overlays
        base_img = input_img.copy()
        # For each layer, get feature map and overlay
        for i, layer in enumerate(model.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ReLU):
                act = x.squeeze().detach().cpu().numpy()
                if act.ndim == 3:
                    # Feature map visualization
                    act_img = act[:3]
                    act_img = (act_img - act_img.min()) / (act_img.ptp() + 1e-5) * 255
                    act_img = act_img.astype(np.uint8)
                    act_img = np.transpose(act_img, (1,2,0))
                    feature_imgs.append(act_img)
                elif act.ndim == 2:
                    act_img = (act - act.min()) / (act.ptp() + 1e-5) * 255
                    act_img = act_img.astype(np.uint8)
                    feature_imgs.append(cv2.cvtColor(act_img, cv2.COLOR_GRAY2BGR))
                feature_names.append(f"{layer.__class__.__name__}_{i}")
                # Overlay feature map (first channel as heatmap) on input image
                fmap = act[0] if act.ndim == 3 else act
                fmap = (fmap - fmap.min()) / (fmap.ptp() + 1e-5)
                fmap = (fmap * 255).astype(np.uint8)
                fmap = cv2.resize(fmap, (base_img.shape[1], base_img.shape[0]))
                heatmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(base_img, 0.6, heatmap, 0.4, 0)
                overlay_imgs.append(overlay)
                overlay_names.append(f"Overlay_{layer.__class__.__name__}_{i}")
        # Final output: get bounding box and confidence
        x_flat = x.view(x.size(0), -1)
        class_logits = model.classifier(x_flat)
        bbox = model.bbox_regressor(x_flat)
        score = torch.sigmoid(class_logits).squeeze().item()
        pred_label = 1 if score >= 0.5 else 0
        bbox_np = bbox.squeeze().detach().cpu().numpy()
        h, w = base_img.shape[:2]
        x_c, y_c, bw, bh = bbox_np
        x_min = int((x_c - bw / 2) * w)
        y_min = int((y_c - bh / 2) * h)
        x_max = int((x_c + bw / 2) * w)
        y_max = int((y_c + bh / 2) * h)
        img_bbox = base_img.copy()
        cv2.rectangle(img_bbox, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(img_bbox, f"Conf: {score:.2f}", (x_min, max(y_min-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        # Show all visualizations
        target_h = 128  # Increased from 64
        name_bar_h = 28 # Increased from 18
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 # Increased from 0.5
        thickness = 2    # Increased from 1
        # Feature maps
        all_imgs = feature_imgs
        all_names = feature_names
        resized_imgs = [cv2.resize(img, (target_h, target_h)) for img in all_imgs]
        img_with_names = []
        for img, name in zip(resized_imgs, all_names):
            bar = np.ones((name_bar_h, target_h, 3), dtype=np.uint8) * 255
            cv2.putText(bar, name, (4, name_bar_h-6), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
            img_with_names.append(np.vstack([img, bar]))
        tiled_features = cv2.hconcat(img_with_names) if img_with_names else None
        # Overlays
        overlay_h = 256  # Increased from 128
        overlay_imgs_resized = [cv2.resize(img, (overlay_h, overlay_h)) for img in overlay_imgs]
        overlay_with_names = []
        for img, name in zip(overlay_imgs_resized, overlay_names):
            bar = np.ones((name_bar_h, overlay_h, 3), dtype=np.uint8) * 255
            cv2.putText(bar, name, (4, name_bar_h-6), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
            overlay_with_names.append(np.vstack([img, bar]))
        tiled_overlays = cv2.hconcat(overlay_with_names) if overlay_with_names else None
        # Show input with bbox
        img_bbox_disp = cv2.resize(img_bbox, (256,256)) # Increased from 128
        bar = np.ones((name_bar_h, 256, 3), dtype=np.uint8) * 255
        cv2.putText(bar, "Input+BBox", (4, name_bar_h-6), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
        img_bbox_disp = np.vstack([img_bbox_disp, bar])
        # Combine all
        visuals = [img_bbox_disp]
        if tiled_features is not None:
            visuals.append(tiled_features)
        if tiled_overlays is not None:
            visuals.append(tiled_overlays)
        # Ensure all visuals have the same width and type
        target_w = 256 # Increased from 128
        visuals_fixed = []
        for vis in visuals:
            # Convert to 3-channel uint8 if needed
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if vis.shape[1] != target_w:
                vis = cv2.resize(vis, (target_w, vis.shape[0]))
            if vis.dtype != np.uint8:
                vis = vis.astype(np.uint8)
            visuals_fixed.append(vis)
        final_vis = cv2.vconcat(visuals_fixed)
        cv2.imshow("Model Layers & BBox Visualization", final_vis)

        
def detect_and_visualize(model, dataset, device, num_images=5, score_thresh=0.5, epoch_num=None):
    model.eval()
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    correct = 0
    total = 0
    # Prepare output directory
    #out_dir = f'./output/epoch{epoch_num}' if epoch_num is not None else './output/epoch'
    #os.makedirs(out_dir, exist_ok=True)
    for idx in indices:
        image, label, bbox_gt = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            class_logits, bbox_pred = model(img_tensor)
            score = torch.sigmoid(class_logits).squeeze().item()
            pred_label = 1 if score >= score_thresh else 0
        total += 1
        is_correct = pred_label == int(label.item())
        if is_correct:
            correct += 1

            sys.stdout.write(
                f"\rImage {idx}: GT label={int(label.item())}, Pred label={pred_label}, Score={score:.4f}, Correct={is_correct}"
            )
        sys.stdout.flush()

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
        # Save image to output directory
        #out_path = os.path.join(out_dir, f'detect_{idx}.jpg')
        #cv2.imwrite(out_path, img_np)
        cv2.imshow("Detection", img_np)
        
        #visualize_model_layers_and_weights(model, img_tensor)
        #visualize_model_graph(model, img_tensor)
        cv2.waitKey(50)
    print(f"Detection accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)\n")

def print_label_distribution(dataset, name="Dataset"):
    pos_count = 0
    neg_count = 0
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        if int(label.item()) == 1:
            pos_count += 1
        else:
            neg_count += 1
        sys.stdout.write(f"\rChecking [{i+1}/{len(dataset)}] ...")
        sys.stdout.flush()
    print(f"\n{name} label distribution: {pos_count} positive, {neg_count} negative samples.")
    return pos_count, neg_count

# --- Ground Truth Bounding Box Visualization ---
def visualize_ground_truth(dataset, num_images=10):
    print(f"\nVisualizing {num_images} ground truth bounding boxes...")
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    for idx in indices:
        image, label, bbox_gt = dataset[idx]
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_np.shape[:2]
        x_c, y_c, bw, bh = bbox_gt.numpy()
        x_min_gt = int((x_c - bw / 2) * w)
        y_min_gt = int((y_c - bh / 2) * h)
        x_max_gt = int((x_c + bw / 2) * w)
        y_max_gt = int((y_c + bh / 2) * h)
        cv2.rectangle(img_np, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 255, 0), 2)
        cv2.putText(img_np, f"GT: {label.item()}", (x_min_gt, max(y_min_gt-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Ground Truth Bounding Box", img_np)
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
train_dataset = YoloDataset(img_dir, label_dir)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = YoloDataset(val_img_dir, val_label_dir)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model, Loss, Optimizer
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA for training.')
else:
    device = torch.device('cpu')
    print('Using CPU for training.')

model = SimpleYoloNet().to(device)

# Dynamically get class distribution for weighting
#num_pos, num_neg = 
#if num_pos > 0:
#    pos_weight = torch.tensor([num_neg / num_pos], device=device)
#else:
#    pos_weight = torch.tensor([1.0], device=device)
print_label_distribution(train_dataset, name="Training set")
print_label_distribution(val_dataset, name="Validation set")

criterion_cls = nn.BCEWithLogitsLoss()#pos_weight=pos_weight)
criterion_bbox = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Visualize the data
#visualize_bbox_centers(train_dataset)
visualize_ground_truth(train_dataset, num_images=10)

epoch_loop = 400
try:
    for epoch in range(epoch_loop):
        model.train()
        batchsize = len(train_loader)
        img_loss = 0  # No missing_labels attribute anymore
        for batch_idx, (images, labels, bboxes) in enumerate(train_loader):
            images, labels, bboxes = images.to(device), labels.to(device).unsqueeze(1), bboxes.to(device)
            class_logits, bbox_preds = model(images)
            loss_cls = criterion_cls(class_logits, labels)
            loss_bbox = criterion_bbox(bbox_preds, bboxes)
            loss = loss_cls + 2.0 * loss_bbox  # Increased bbox loss weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print bbox output stats for debugging
            bbox_min = bbox_preds.min().item()
            bbox_max = bbox_preds.max().item()
            bbox_mean = bbox_preds.mean().item()
            sys.stdout.write(
                f"\rEpoch [{epoch+1}/{epoch_loop}] Batch [{batch_idx+1}/{batchsize}] [Missing labels: {img_loss}] Loss: {loss.item():.4f} | loss_cls: {loss_cls.item():.4f}, loss_bbox: {loss_bbox.item():.4f}" #| bbox_out: min={bbox_min:.3f}, max={bbox_max:.3f}, mean={bbox_mean:.3f}
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