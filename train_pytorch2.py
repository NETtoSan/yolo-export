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
    def encode_yolo_grid(self, label_path, S=None, B=2, C=1):
        if S is None:
            S = self.grid_size
        """
        Encode ground truth boxes into YOLO grid format.
        Returns: grid tensor of shape [S, S, B*(5+C)]
        """
        grid = np.zeros((S, S, B * (5 + C)), dtype=np.float32)
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0]) if C > 1 else 0
                x_c, y_c, w, h = [float(x) for x in parts[1:5]]
                # Find grid cell
                cell_x = int(x_c * S)
                cell_y = int(y_c * S)
                cell_x = min(cell_x, S-1)
                cell_y = min(cell_y, S-1)
                # Find first available box slot in cell
                for b in range(B):
                    obj_idx = b * (5 + C) + 4
                    if grid[cell_y, cell_x, obj_idx] == 0:
                        # Assign box
                        box_idx = b * (5 + C)
                        grid[cell_y, cell_x, box_idx:box_idx+4] = [x_c, y_c, w, h]
                        grid[cell_y, cell_x, box_idx+4] = 1  # objectness
                        if C > 1:
                            grid[cell_y, cell_x, box_idx+5+class_id] = 1  # one-hot class
                        break
        return torch.tensor(grid)
    def __init__(self, img_dir, label_dir, transform=None, grid_size=15):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.grid_size = grid_size
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
        # Use YOLO grid encoding for multi-object detection
        grid = self.encode_yolo_grid(label_path, S=self.grid_size, B=2, C=1)
        return image, grid

class SimpleYoloNet(nn.Module):
    def __init__(self, num_classes=1, grid_size=7, num_boxes=2):
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
        )
        # Output: [batch, S, S, B*(5+C)]
        self.pred = nn.Conv2d(512, num_boxes * (5 + num_classes), 1)  # 1x1 conv

        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)
        # x shape: [batch, 512, S, S]
        x = self.pred(x)
        # x shape: [batch, B*(5+C), S, S]
        x = x.permute(0, 2, 3, 1)  # [batch, S, S, B*(5+C)]
        return x  # Only return the grid tensor


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    batchsize = len(dataloader)
    with torch.no_grad():
        for batch_idx, (images, grids) in enumerate(dataloader):
            images = images.to(device)
            grids = grids.to(device)
            preds = model(images)
            loss = nn.functional.mse_loss(preds, grids)
            total_loss += loss.item() * images.size(0)
            pred_min = preds.min().item()
            pred_max = preds.max().item()
            pred_mean = preds.mean().item()
            sys.stdout.write(
                f"\rValidating Batch [{batch_idx+1}/{batchsize}] Loss: {loss.item():.4f} | pred_out: min={pred_min:.3f}, max={pred_max:.3f}, mean={pred_mean:.3f}"
            )
            sys.stdout.flush()
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"\nValidation Loss: {avg_loss:.4f}")

def calculate_map(model, dataloader, device, iou_threshold=0.5, obj_thresh=0.5):
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_gt = 0
    with torch.no_grad():
        for batch_idx, (images, grids) in enumerate(dataloader):
            images = images.to(device)
            grids = grids.to(device)
            preds = model(images)
            batch_size = images.size(0)
            S = preds.size(1)
            B = preds.size(3) // 5  # assuming C=1
            for i in range(batch_size):
                pred_boxes = []
                gt_boxes = []
                # Collect predicted boxes above threshold
                for y in range(S):
                    for x in range(S):
                        for b in range(B):
                            obj_score = preds[i, y, x, b*5+4].item()
                            if obj_score > obj_thresh:
                                box = preds[i, y, x, b*5:b*5+4].cpu().numpy()
                                pred_boxes.append(box)
                        for b in range(B):
                            gt_obj = grids[i, y, x, b*5+4].item()
                            if gt_obj > 0:
                                box = grids[i, y, x, b*5:b*5+4].cpu().numpy()
                                gt_boxes.append(box)
                matched_gt = set()
                for pb in pred_boxes:
                    found_match = False
                    for j, gb in enumerate(gt_boxes):
                        if j in matched_gt:
                            continue
                        # Compute IoU
                        iou = compute_iou(pb, gb)
                        if iou >= iou_threshold:
                            true_positives += 1
                            matched_gt.add(j)
                            found_match = True
                            break
                    if not found_match:
                        false_positives += 1
                false_negatives += len(gt_boxes) - len(matched_gt)
                total_gt += len(gt_boxes)
            sys.stdout.write(f"\rBatch [{batch_idx+1}/{len(dataloader)}] TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
            sys.stdout.flush()
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")

def compute_iou(box1, box2):
    # box = [x_c, y_c, w, h] normalized
    x1_min = box1[0] - box1[2]/2
    y1_min = box1[1] - box1[3]/2
    x1_max = box1[0] + box1[2]/2
    y1_max = box1[1] + box1[3]/2
    x2_min = box2[0] - box2[2]/2
    y2_min = box2[1] - box2[3]/2
    x2_max = box2[0] + box2[2]/2
    y2_max = box2[1] + box2[3]/2
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def detect_and_visualize(model, dataset, device, num_images=5, obj_thresh=0):
    model.eval()
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    for idx in indices:
        image, grid = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_tensor)
        preds_np = preds.squeeze(0).cpu().numpy()  # [S, S, B*(5+C)]
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_np.shape[:2]
        S = preds_np.shape[0]
        B = preds_np.shape[2] // 5  # assuming C=1
        # Draw predicted boxes in red
        for y in range(S):
            for x in range(S):
                for b in range(B):
                    obj_idx = b * 5 + 4
                    if preds_np[y, x, obj_idx] >= obj_thresh:
                        box_idx = b * 5
                        x_c = preds_np[y, x, box_idx]
                        y_c = preds_np[y, x, box_idx+1]
                        bw = preds_np[y, x, box_idx+2]
                        bh = preds_np[y, x, box_idx+3]
                        x_min = int((x_c - bw / 2) * w)
                        y_min = int((y_c - bh / 2) * h)
                        x_max = int((x_c + bw / 2) * w)
                        y_max = int((y_c + bh / 2) * h)
                        cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.imshow("YOLO Detection (Grid)", img_np)
        cv2.waitKey(150)
    cv2.destroyAllWindows()
    print("Detection visualization complete.\n")


# Paths
img_dir = './yolov11/bottlesv11/train/images'
label_dir = './yolov11/bottlesv11/train/labels'
val_img_dir = './yolov11/bottlesv11/valid/images'
val_label_dir = './yolov11/bottlesv11/valid/labels'


# Dynamically set grid_size so it divides image_size exactly
image_size = 480
grid_size = image_size // 32  # 480 // 32 = 15
print(f"Using image_size={image_size}, grid_size={grid_size}")
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# Dataset and DataLoader
train_dataset = YoloDataset(img_dir, label_dir, transform=transform, grid_size=grid_size)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = YoloDataset(val_img_dir, val_label_dir, transform=transform, grid_size=grid_size)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Print label distribution in validation set
pos_count = 0
neg_count = 0
for i in range(len(val_dataset)):
    _, grid = val_dataset[i]
    # Count cells with objectness > 0
    obj_cells = (grid[..., 4::5] > 0).sum().item()
    pos_count += obj_cells
    neg_count += grid.shape[0] * grid.shape[1] * (grid.shape[2] // 5) - obj_cells
print(f"Validation set label distribution: {pos_count} positive, {neg_count} negative cells.")

# Model, Loss, Optimizer

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA for training.')
else:
    device = torch.device('cpu')
    print('CUDA not available. Using CPU for training.')





model = SimpleYoloNet(num_classes=1, grid_size=grid_size, num_boxes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# --- Ground Truth Bounding Box Visualization ---
def visualize_ground_truth(dataset, num_images=10):
    import cv2
    import numpy as np
    print(f"\nVisualizing {num_images} ground truth bounding boxes...")
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    for idx in indices:
        image, grid = dataset[idx]
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_np.shape[:2]
        grid_np = grid.numpy()
        S = grid_np.shape[0]
        B = grid_np.shape[2] // 5  # assuming C=1
        for y in range(S):
            for x in range(S):
                for b in range(B):
                    obj_idx = b * 5 + 4
                    if grid_np[y, x, obj_idx] > 0:
                        box_idx = b * 5
                        x_c = grid_np[y, x, box_idx]
                        y_c = grid_np[y, x, box_idx+1]
                        bw = grid_np[y, x, box_idx+2]
                        bh = grid_np[y, x, box_idx+3]
                        x_min_gt = int((x_c - bw / 2) * w)
                        y_min_gt = int((y_c - bh / 2) * h)
                        x_max_gt = int((x_c + bw / 2) * w)
                        y_max_gt = int((y_c + bh / 2) * h)
                        cv2.rectangle(img_np, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 255, 0), 2)
        cv2.imshow("Ground Truth Bounding Boxes (YOLO Grid)", img_np)
        cv2.waitKey(300)
    cv2.destroyAllWindows()
    print("Visualization complete.\n")


# Visualize ground truth bounding box centers
def visualize_bbox_centers(dataset):
    import matplotlib.pyplot as plt
    x_centers = []
    y_centers = []
    for i in range(len(dataset)):
        _, grid = dataset[i]
        grid_np = grid.numpy()
        S = grid_np.shape[0]
        B = grid_np.shape[2] // 5  # assuming C=1
        for y in range(S):
            for x in range(S):
                for b in range(B):
                    obj_idx = b * 5 + 4
                    if grid_np[y, x, obj_idx] > 0:
                        box_idx = b * 5
                        x_c = grid_np[y, x, box_idx]
                        y_c = grid_np[y, x, box_idx+1]
                        x_centers.append(x_c)
                        y_centers.append(y_c)
        sys.stdout.write(f"\rProcessed image {i+1}/{len(dataset)}")
        sys.stdout.flush()
    print()
    plt.figure(figsize=(6,6))
    plt.scatter(x_centers, y_centers, alpha=0.5)
    plt.title('Ground Truth Bounding Box Centers (YOLO Grid)')
    plt.xlabel('x_center (normalized)')
    plt.ylabel('y_center (normalized)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

#visualize_bbox_centers(train_dataset)
visualize_ground_truth(train_dataset, num_images=10)

epoch_loop = 50
try:
    for epoch in range(epoch_loop):
        model.train()
        batchsize = len(train_loader)
        img_loss = len(train_dataset.missing_labels)
        for batch_idx, (images, grids) in enumerate(train_loader):
            images = images.to(device)
            grids = grids.to(device)
            preds = model(images)
            # Simple loss: MSE between prediction and grid (for demonstration)
            loss = nn.functional.mse_loss(preds, grids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print output stats for debugging
            pred_min = preds.min().item()
            pred_max = preds.max().item()
            pred_mean = preds.mean().item()
            sys.stdout.write(
                f"\rEpoch [{epoch+1}/{epoch_loop}] Batch [{batch_idx+1}/{batchsize}] [Missing labels: {img_loss}] Loss: {loss.item():.4f} | pred_out: min={pred_min:.3f}, max={pred_max:.3f}, mean={pred_mean:.3f}"
            )
            sys.stdout.flush()

        print(f"\nEpoch {epoch+1} Loss: {loss.item():.4f}")
        print(f"Calculating mAP on validation set after epoch {epoch+1}...")
        calculate_map(model, val_loader, device)
        validate(model, val_loader, device)

        print(f"Detecting 30 images from validation set after epoch {epoch+1}...")
        detect_and_visualize(model, val_dataset, device, num_images=30)

    # Save model
    torch.save(model.state_dict(), 'simple_yolo_model.pt')
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Exiting cleanly...")