import cv2
import os
import threading
import time
import numpy as np
 
album_path = "./yolov8/bottle/train/images"
label_path = "./yolov8/bottle/train/labels"
is_dir = False
warning_text = None
currnet_file = ""
current_file = ""

# Check if a path is a directory. Otherwise quit
files = []
label_files = []
if os.path.isdir(album_path):
    is_dir = True
    files = sorted(os.listdir(album_path))
    files = [f for f in files if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    if not files:
        print(f"No valid image files found in the directory: {album_path}")
        exit()
else:
    print(f"The path {album_path} is not a directory or does not exist.")
    exit()

if is_dir == True:
    if os.path.isdir(label_path):
        label_files = sorted(os.listdir(label_path))
        label_files = [f for f in label_files if os.path.splitext(f)[1].lower() in ['.txt']]
        if not label_files:
            warning_text = (f"No valid label files found in the directory: {label_path}")
    else:
        warning_text = (f" [!!!!] The path is not a directory or does not exist.")

print("\nUsing the following images/label path:" \
      f"\n[Images]: {album_path}\n[Labels]: {label_path}{warning_text if warning_text is not None else ""}\n")

frame = current_frame = None
frame_lock = threading.Lock()
stop_threads = False

def read_images():
    global album_path, label_path
    # This will only read images
    # And will read 5 images per seceond in the path
    global frame, current_frame, album_path, stop_threads, frame_lock, files, current_file
    #print("Spawned!")
    album_waitmsec = 1.0 / 30.0
    album_size = len(files)
    file_index = 0
    while not stop_threads:
        current_file = files[file_index]
        ext = os.path.splitext(current_file)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(album_path, current_file)
            frame = cv2.imread(img_path)
            print(f"({file_index}/{album_size}): {current_file}")
            if frame is None:
                print(f"Failed to read image: {img_path}")
                file_index = (file_index + 1) % len(files)
                continue

            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)  # Resize to 640x640
            if stop_threads:
                break
            with frame_lock:
                current_frame = frame.copy()
            time.sleep(album_waitmsec)
            
            file_index = (file_index + 1) % len(files)

def draw_polygon(image, points, color=(0, 0, 255), thickness=2):
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

def verify_files():
    is_ok = True
    not_ok_count = 0
    for i in range(len(files)):
        #print(f"Verifying file: {files[i]}")
        #Check whether that file have a corresponding txt label file
        # If not then calculate how healthy the folder is
        label_file = os.path.join(label_path, os.path.splitext(files[i])[0] + '.txt')
        
        if not os.path.exists(label_file):
            not_ok_count += 1
            print(f"{not_ok_count} Label file does not exist!")
            #is_ok = False
        pass
    print(f"{len(files)} Files OK")
    return is_ok
try:
    
    files_ok = verify_files()
    if not files_ok:
        exit()

    threading.Thread(target=read_images, daemon=True).start()

    while True:
        with frame_lock:
            if current_frame is None or current_file is None:
                time.sleep(0.01)
                continue
            img_display = current_frame.copy()
            file_name = current_file
        
        if img_display is None:
            continue
        
        h, w = img_display.shape[:2]
        label_file = os.path.join(label_path, os.path.splitext(file_name)[0] + '.txt')

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # skip invalid lines
                
                cls_id = int(parts[0])

                # Detect if this is a bounding box (5 elements) or polygon (>5)
                if len(parts) == 5:
                    cx, cy, bw, bh = map(float, parts[1:])
                    x_center = int(cx * w)
                    y_center = int(cy * h)
                    box_w = int(bw * w)
                    box_h = int(bh * h)

                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)

                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_display, str(cls_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # Polygon segmentation
                    points = []
                    coords = list(map(float, parts[1:]))
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * w)
                        y = int(coords[i + 1] * h)
                        points.append((x, y))
                    draw_polygon(img_display, points, color=(0, 0, 255))

        else:
            print(f"Label file does not exist: {label_file}")

        # Display the image with rectangles or polygons
        cv2.imshow("Image Viewer", img_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except:
    print("Failed to start the image reading thread.")

finally:
    stop_threads = True
    cv2.destroyAllWindows()
