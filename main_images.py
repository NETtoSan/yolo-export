from collections import defaultdict
from ultralytics import YOLO


import threading
import time
import cv2
import numpy as np
import os
import torch



device_preferences = "intel:gpu" if torch.cuda.is_available() else "cpu"

mod_name = "bottles11-11s-480" #"bottles8-v11_3gflops_480_results"
model = YOLO(f"./runs/detect/{mod_name}/weights/best_openvino_model")

#model = YOLO('./files/bottles9-v11n-480.pt', task='detect')

album_path = "./yolov11/bottlesv11/test/images"
vid_path = "./files/test4.mp4"
cap = cv2.VideoCapture(vid_path)

# set cap resolution to 640x480 and mark frame center
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
cy = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

frame = current_frame = None
frame_lock = threading.Lock()
stop_threads = False

read_frame_start_time = time.time()
read_frame = 0
read_fps = 0
fps = 0
total_frame = 0
prev_time = time.time()
bottles_cnt = 0

track_history = defaultdict(lambda: [])
precision_history = defaultdict(lambda: [])
interest_track = 0

def read_frames():
    global frame, stop_threads, read_frame, read_fps, read_frame_start_time

    cv2.setNumThreads(2)
    frame_interval = 0.5  # 1 second per image

    # Get list of image files in album_path
    image_files = [os.path.join(album_path, f) for f in os.listdir(album_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files.sort()
    idx = 0
    num_images = len(image_files)

    while not stop_threads:
        loop_time = time.time()

        if num_images == 0:
            print("No images found in album_path.")
            time.sleep(1)
            continue

        img = cv2.imread(image_files[idx])
        if img is None:
            print(f"Failed to read {image_files[idx]}")
        else:
            with frame_lock:
                frame = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

        idx = (idx + 1) % num_images

        read_frame += 1
        elapsed = time.time() - loop_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        if time.time() - read_frame_start_time >= 1.0:
            read_fps = read_frame
            read_frame = 0
            read_frame_start_time = time.time()



try:

    threading.Thread(target=read_frames, daemon=True).start()

    while True:
        #ret, frame = cap.read()
        total_frame += 1
        time_now = time.time()
        
        with frame_lock:
            if frame is None:
                continue
            draw_frame = frame.copy()  # Make a copy for drawing overlays
        
        if True:
            cv2.circle(draw_frame, (cx, cy), radius=4, color=(255, 255, 0), thickness=-1)

            results = model.track(draw_frame, persist=False, show=False, verbose=False, tracker="bytetrack.yaml", conf=0.3, device=device_preferences)

            if results[0].boxes and results[0].boxes.is_track:
                max_area = 0
                boxes = results[0].boxes.xywh
                track_ids = results[0].boxes.id
                precision_ids = results[0].boxes.conf

                if track_ids is not None:
                    track_ids = track_ids.tolist()
                else:
                    track_ids = []

                if precision_ids is not None:
                    precision_ids = precision_ids.tolist()
                else:
                    precision_ids = []

                #frame = results[0].plot()

                # Find largest bounding boxes from every detected object
                max_area = 0
                max_area_track_id = None
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    area = int(w * h)
                    if area > max_area:
                        max_area = area
                        max_area_track_id = track_id

                for box, track_id, precision_id in zip(boxes, track_ids, precision_ids):
                    x, y, w, h = box

                    precision = precision_history[precision_id]
                    precision.append(precision_id)  # Store the precision value
                    if len(precision) > 30:  # retain 30 precision values for
                        precision.pop(0)

                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    
                    # Put track_id and (x, y) above the rectangle
                    class_id = int(results[0].boxes.cls[track_ids.index(track_id)]) if hasattr(results[0].boxes, 'cls') else None
                    class_name = model.names[class_id] if class_id is not None and hasattr(model, 'names') else "object"

                    # Mark the largest bounding box with a different color
                    if track_id == max_area_track_id:
                        bound_color = (0, 0, 255)
                        dot_color = (255, 0, 255)
                        interest_track = track_id

                        # Draw arrow from the center of the frame to the center of the bounding box
                        cv2.arrowedLine(draw_frame, (cx, cy), (int(x), int(y)), (0, 0, 255), thickness=2, tipLength=0.1)

                    else:
                        bound_color = (0, 255, 0)
                        dot_color = (0, 255, 0)
                    color = (0, 255, 0)

                    # Assign a color based on the class name
                    if class_name == "bottle":
                        bottles_cnt += 1
                        color = (255, 0, 0)
                    elif class_name == "bottles":
                        bottles_cnt += 1
                        color = (255, 255, 0)

                    
                    label = f"{class_name} {int(track_id)} {int(precision_id * 100)}% ({int(x)}, {int(y)})"
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    # Draw black outline for text
                    cv2.putText(draw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, lineType=cv2.LINE_AA)
                    # Draw colored text on top
                    cv2.putText(draw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)
                    # Draw rectangle for the bounding box
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), bound_color, 2)

                    # Draw a dot at the center of the bounding box
                    center = (int(x), int(y))
                    cv2.circle(draw_frame, center, radius=4, color=dot_color, thickness=-1)


                    # Draw the tracking lines
                    #points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    #cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            else:
                interest_track = 0
                bottles_cnt = 0
        # Show process FPS
        # Reset total_frame every second
        if time_now - prev_time >= 1.0:
            fps = total_frame
            total_frame = 0
            prev_time = time_now

        cv2.putText(draw_frame, f"Processing at {fps:.0f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(draw_frame, f"Reading at {read_fps:.0f} FPS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(draw_frame, f"{total_frame} frames", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        cv2.putText(draw_frame, f"{bottles_cnt if bottles_cnt > 0 else 'No'} bottle{'s' if bottles_cnt > 1 else ''}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(draw_frame, f"{'Not looking at anything...' if interest_track == 0 else 'Looking at bottle ' + str(int(interest_track))}\
                    ", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        cv2.imshow("Tracking", draw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        bottles_cnt = 0
except:
    print("An error occurred. Exiting...")

cap.release()
cv2.destroyAllWindows()