print("------- Tracking software research and development\n\
[2025] Net Zamora, Thai Nichi Institute of Technology\n")

from ultralytics import YOLO
from collections import defaultdict
import threading
import time
import cv2
import numpy as np
import os
import torch
import thop
import serial
import openvino


device_preferences= "cpu" if torch.cuda.is_available() else "cpu"

mod_name = "bottles11-11s-480" #"bottles8-v11_3gflops_480_results"
model = YOLO(f"./runs/detect/{mod_name}/weights/best_int8_openvino_model")

#model = YOLO('./files/bottles9-v11n-480.pt', task='detect')
model_benchmark = YOLO("yolo11n.yaml")


print(f"[OK] Model {mod_name}")
print(f"[OK] Using {device_preferences}")
print(f"OPENVINO devices: {openvino.Core().available_devices}")

# Calculate gflops & params
def validate_model():
    if model is not None:
        input_tensor = torch.randn(1, 3, 480, 480)
        flops, params = thop.profile(model.model, inputs=(input_tensor, )); print("")
        bflops, bparams = thop.profile(model_benchmark.model, inputs=(input_tensor, ))

        gflops = flops / 1e9
        bgflops = bflops / 1e9
        print("---------------------------")
        for name, layer in model.model.named_modules():  # Use named_modules for better layer inspection
            # Check for activation functions
            if isinstance(layer, torch.nn.ReLU):
                print("[OK] Activation: ReLU")
            elif isinstance(layer, torch.nn.LeakyReLU):
                print("[OK] Activation: LeakyReLU")
            elif isinstance(layer, torch.nn.SiLU):  # SiLU is commonly used in YOLO models as the activation
                print("[OK] Activation: SiLU (Swish)")
            elif hasattr(layer, 'activation'):
                print(f"[OK] Activation: {layer.activation}")
            

        print(f"---- benchmark ---- {model_benchmark.info()}")
        print(f"---- model ---- {model.info()}")
        print(f"\n[OK] {bgflops:.2f} gflops, {bparams / 1e6:.2f} million params (yaml build)")
        print(f"[OK] {gflops:.2f} gflops, {params / 1e6:.2f} million params (pytorch model)")


#validate_model()

vid_path = 3
virt_path = str()
# If video path is a directory, set it to the first photo/video in that directory
if os.path.isdir(vid_path) and not isinstance(vid_path, int):
    vid_files = sorted(os.listdir(vid_path))
    vid_files = [f for f in vid_files if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"]]
    if not vid_files:
        print("[!!] No valid video files found in the directory.")
        exit()
    virt_path = os.path.join(vid_path, vid_files[0])
    print(f"[OK] Using path {vid_path}")
else:
    print(f"[OK] Using video {vid_path}")
    virt_path = vid_path

UART_port = ""
ser = None
UART_is_exist = False
try:
    if UART_port:
        ser = serial.Serial(UART_port, 15200)
        UART_is_exist = ser.is_open
        print(f"[OK] UART OK {UART_is_exist}")
    else:
        print("[!!] UART port not specified")
except serial.SerialException:
    print("[!!] UART port not found or not available")
    UART_is_exist = False

cap = cv2.VideoCapture(virt_path)

# set cap resolution to 640x480 and mark frame center
frame_x = 640
frame_y = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

cx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
cy = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
print("---------------------------")
print(f"[OK] Resolution {frame_x}:{frame_y}")

frame = current_frame = None
frame_lock = threading.Lock()
stop_threads = False

read_frame_start_time = time.time()
read_frame = 0
read_fps = 0
fps = 0
total_frame = 0
prev_time = time.time()
read_frame_count = 0
bottles_cnt = 0
nodetect_last = 0

track_history = defaultdict(lambda: [])
precision_history = defaultdict(lambda: [])
interest_track = 0

def read_frames():
    ##############################
    # This must be able to run independently
    # So that it can read frames and inference separately. not tied to each other
    # This must also be able to read images/videos directly from a path, or a directory
    # If it's a path to a directory, it should read all images/videos in that directory
    # Read images and video frame rate must be kept separate. Cycle between images 5 times/ playback videos at 20fps
    ##############################

    print("[OK] Video READY")
    global frame, read_frame, read_frame_count, read_frame_start_time, read_fps, stop_threads, cap
    frame_interval = 1.0 / 99   # 99 FPS for video, if the camera supports it
    album_interval = 1.0 / 5.0  # For images only

    album_files = []
    is_album = False

    if os.path.isdir(vid_path):
        is_album = True
        album_files = sorted(os.listdir(vid_path))
        album_files = [os.path.join(str(vid_path), f) for f in album_files]
        album_files = [f for f in album_files
                       if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"]]

    is_image = False
    if isinstance(vid_path, str) and not is_album:
        ext = os.path.splitext(vid_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            is_image = True

    if is_album:
        file_index = 0
        while not stop_threads:
            current_file = album_files[file_index]
            ext = os.path.splitext(current_file)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                img = cv2.imread(current_file)
                if img is None:
                    print(f"Error reading album image {current_file}")
                    file_index = (file_index + 1) % len(album_files)
                    continue
                img = cv2.resize(img, (frame_x, frame_y), interpolation=cv2.INTER_LINEAR)
                for _ in range(int(5)):  # Display 5 frames worth at 5fps (1s)
                    if stop_threads:
                        break
                    with frame_lock:
                        frame = img.copy()
                    read_frame += 1
                    read_frame_count += 1
                    time.sleep(album_interval)
                    if time.time() - read_frame_start_time >= 1.0:
                        read_fps = read_frame
                        read_frame = 0
                        read_frame_start_time = time.time()
                file_index = (file_index + 1) % len(album_files)

            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                cap.open(current_file)
                if not cap.isOpened():
                    print(f"Error opening album video {current_file}")
                    file_index = (file_index + 1) % len(album_files)
                    continue
                while not stop_threads:
                    loop_start = time.time()
                    cap.grab()
                    ret, new_frame = cap.retrieve()
                    if not ret:
                        cap.release()
                        file_index = (file_index + 1) % len(album_files)
                        break
                    with frame_lock:
                        frame = cv2.resize(new_frame, (frame_x, frame_y), interpolation=cv2.INTER_LINEAR)
                    read_frame += 1
                    read_frame_count += 1
                    elapsed = time.time() - loop_start
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                    if time.time() - read_frame_start_time >= 1.0:
                        read_fps = read_frame
                        read_frame= 0
                        read_frame_start_time = time.time()

    elif is_image:
        img = cv2.imread(str(vid_path))
        if img is None:
            print(f"Error: Unable to read image {vid_path}")
            stop_threads = True
            return
        img = cv2.resize(img, (frame_x, frame_y), interpolation=cv2.INTER_LINEAR)
        while not stop_threads:
            with frame_lock:
                frame = img.copy()
            read_frame += 1
            read_frame_count += 1
            time.sleep(frame_interval)
            if time.time() - read_frame_start_time >= 1.0:
                read_fps = read_frame
                read_frame = 0
                read_frame_start_time = time.time()

    else:
        while not stop_threads:
            loop_start = time.time()
            cap.grab()
            ret, new_frame = cap.retrieve()
            if not ret:
                print("Video ended. Restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.5)
                continue
            with frame_lock:
                frame = cv2.resize(new_frame, (frame_x, frame_y), interpolation=cv2.INTER_LINEAR)
            # Show the raw frame in a separate window
            cv2.imshow("Raw Frame", new_frame)
            cv2.waitKey(1)  # Allow OpenCV to process window events
            read_frame += 1
            read_frame_count += 1
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            if time.time() - read_frame_start_time >= 1.0:
                read_fps = read_frame
                read_frame = 0
                read_frame_start_time = time.time()

def print_uart():
    global ser, UART_is_exist, stop_threads
    # Send serial data X times/ second
    packet_count = 10
    interval = 1.0 / packet_count
    print("[OK] Serial READY")

    uart_fail_count = 0
    while not stop_threads:
        try:
            if ser is not None:
                ser.write(b"Hello!")
                #print(f"[OK] Data sent")
            else:
                uart_fail_count += 1
                #print(f"[FAIL] Cannot send data {uart_fail_count}")

        except Exception as e:
            uart_fail_count += 1
            #print(f"[FAIL] Something went wrong {uart_fail_count} {e}")
            pass

        if uart_fail_count > 10:
            #print("---------------------------\n[INFO] Failed to send 10 packets. Restarting UART.....")
            uart_fail_count = 0
            for i in range(10):
                try:
                    ser = serial.Serial(UART_port, 15200)
                    UART_is_exist = ser.is_open
                except serial.SerialException:
                    #print(f"[FAIL] {i+1}/10 UART port not found or not available")
                    UART_is_exist = False
                time.sleep(0.1)
            
            if UART_is_exist:
                #print("[OK] UART restarted successfully")
                pass
            else:
                #print("[FAIL] UART restart failed.")
                pass
            
        time.sleep(interval)

try:
    print("---------------------------")
    threading.Thread(target=read_frames, daemon=True).start()
    threading.Thread(target=print_uart, daemon=True).start()

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

            results = model.track(draw_frame, persist=True, show=False, verbose=False, tracker="bytetrack.yaml", conf=0.5, device=device_preferences)
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
                    bottles_cnt += 1
                    color = (255, 0, 0)

                    
                    label = f"{int(track_id)} {int(precision_id * 100)}% ({int(x)}, {int(y)})"
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

        if time.time() - nodetect_last > 5 and bottles_cnt == 0:
            nodetect_last = time.time()
            print("[INFO] No bottles detected for 5 seconds.")
            
        # Show process FPS
        # Reset total_frame every second
        if time_now - prev_time >= 1.0:
            fps = total_frame
            total_frame = 0
            prev_time = time_now

        cv2.putText(draw_frame, f"Processing at {fps:.0f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(draw_frame, f"Reading at {read_fps:.0f} FPS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(draw_frame, f"{total_frame} | {read_frame}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        cv2.putText(draw_frame, f"{bottles_cnt if bottles_cnt > 0 else 'No'} bottle{'s' if bottles_cnt > 1 else ''}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(draw_frame, f"{'Not looking at anything...' if interest_track == 0 else 'Looking at bottle ' + str(int(interest_track))}\
                    ", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        cv2.imshow("Tracking", draw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        bottles_cnt = 0
except Exception as e:
    print(f"An error occurred. Exiting...\n{e}")

cap.release()
cv2.destroyAllWindows()