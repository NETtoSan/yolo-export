import cv2
import numpy as np
import torch
import time
import serial
import threading
import multiprocessing
from ultralytics import YOLO
import os

print("-----------------------------\nYOLO Detection for ABU Robocon 2025\n-----------------------------")
print("Loading...........")

device_preferences = 'intel:gpu'
device = device_preferences if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device} Multithreading")

mod_name = "bottles11-11s-480" #"bottles8-v11_3gflops_480_results"
model = YOLO(f"./runs/detect/{mod_name}/weights/best_openvino_model")

vid_path = "./files/test4.mp4" 
cap = cv2.VideoCapture(vid_path)
# Try to set camera buffer size to 1 to minimize latency (may not be supported by all cameras)
if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

on_pi = False
vid_fps = 20

serial_port = "/dev/ttyUSB0"
try:
    ser = serial.Serial(port=serial_port, baudrate=15200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_TWO)
    print("Successfully set serial/uart")
except Exception:
    print("Failed to set serial/uart")

bball_px = bball_py = 0
save_pos = True
bball_pos = []
pad_x1y1 = pad_x2y2 = basket_x1y1 = basket_x2y2 = basketball_x1y1 = basketball_x2y2 = (0, 0)
bball_cx = bball_cy = pad_cx = pad_cy = face_cnt = 0
rsx = rsy = rcx = rcy = 0
rstat = 'failed'

frame_x, frame_y = 640, 480
frame_cx, frame_cy = frame_x // 2, frame_y // 2
fps = 30
prev_frames = frame_count = video_frame = 0
start_time = time.time()
prev_time = frametime = 0

angle_deg = 0
angle_deg_arc = 0

read_frame_count = read_frame_fps = 0
read_frame_start_time = time.time()

if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

frame = current_frame = None
frame_lock = threading.Lock()
stop_threads = False

# On Raspberry Pi 5, set OpenCV to use a single thread to avoid CPU contention.
cpu_threads = 3

# Use all available CPU threads for PyTorch
max_threads = os.cpu_count() or 1
torch.set_num_threads(max_threads)
torch.set_num_interop_threads(max_threads)

def read_frames():
    global frame, video_frame, read_frame_count, read_frame_start_time, read_frame_fps, stop_threads
    cv2.setNumThreads(cpu_threads)  # Limit OpenCV thread for best Pi 5 performance
    frame_interval = 1.0 / vid_fps  # Limit frame rate

    # Detect if input is image or video
    is_image = False
    if isinstance(vid_path, str):
        ext = os.path.splitext(vid_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            is_image = True

    if is_image:
        # Read the image once and keep reusing it
        img = cv2.imread(str(vid_path))
        if img is None:
            print(f"Error: Unable to read image {vid_path}")
            stop_threads = True
            return
        img = cv2.resize(img, (frame_x, frame_y), interpolation=cv2.INTER_LINEAR)
        while not stop_threads:
            with frame_lock:
                frame = img.copy()
            video_frame += 1
            read_frame_count += 1
            # Simulate frame rate for image
            time.sleep(frame_interval)
            if time.time() - read_frame_start_time >= 1.0:
                read_frame_fps = read_frame_count
                read_frame_count = 0
                read_frame_start_time = time.time()

    else:
        while not stop_threads:
            loop_start = time.time()
            # Always grab the latest frame to reduce latency
            for _ in range(1):  # Try to clear buffer (tune as needed)
                cap.grab()
            ret, new_frame = cap.retrieve()
            if not ret:
                print("Video ended. Restarting...")
                bball_pos.clear()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.5)
                continue
            # Rotate the frame 180 degrees
            if on_pi == True:
                new_frame = cv2.rotate(new_frame, cv2.ROTATE_180)
            # Restore resize for model input
            with frame_lock:
                frame = cv2.resize(new_frame, (frame_x, frame_y), interpolation=cv2.INTER_LINEAR)

            video_frame += 1
            read_frame_count += 1
            # Limit frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            if time.time() - read_frame_start_time >= 1.0:
                read_frame_fps = read_frame_count
                read_frame_count = 0
                read_frame_start_time = time.time()

def printUART():
    global rsx, rsy, rcx, rcy, rstat, stop_threads
    while not stop_threads:
        try:
            message = f'{rsx} {rsy} {rcx} {rcy}\n'
            ser.write(message.encode('utf-8'))
            ser.flush()
            rstat = 'send OK'
        except Exception:
            rstat = 'failed'
        time.sleep(0.01)

def bballTrk():
    global bball_px, bball_py, bball_cx, bball_cy, speed, current_frame
    cv2.rectangle(new_frame, pad_x1y1, pad_x2y2, (0, 255, 255), thickness=2)
    cv2.putText(new_frame, f"bball_cx: {rsx} {bball_cx - 320}, bball_cy: {rsy} {bball_cy - 240}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if ((bball_cx == 0 and bball_cy == 0) or not bball_pos):
        cv2.putText(new_frame, "No basketball", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if bball_px != 0 and bball_py != 0:
        distance = ((bball_cx - bball_px) ** 2 + (bball_cy - bball_py) ** 2) ** 0.5
        velocity = ((bball_cx - bball_px) / frametime, (bball_cy - bball_py) / frametime)
        speed = distance / (1 / vid_fps)
        arrow_end = (int(bball_cx + velocity[0] // 2), int(bball_cy + velocity[1] // 2))
        if current_frame is not None:
            #cv2.arrowedLine(current_frame, (int(bball_cx), int(bball_cy)), (int(arrow_end[0]), int(arrow_end[1])), (255, 255, 0), 2)
            pass
        #cv2.arrowedLine(new_frame, (bball_cx, bball_cy), arrow_end, (255, 255, 0), 2)

        if bball_cx != 0 and bball_cy != 0 and (bball_cx != frame_cx or bball_cy != frame_cy):
            if save_pos:
                bball_pos.append((bball_cx, bball_cy))
        if len(bball_pos) > 30:
            bball_pos.pop(0)
        for i, point in enumerate(bball_pos):
            cv2.circle(new_frame, point, 3, (0, 0, 255), -1)
            if i > 0:
                cv2.line(new_frame, bball_pos[i - 1], point, (0, 255, 0), 2)
        
        # Track and predict current's basketball trajectory
        if len(bball_pos) >= 3:
            recent_points = bball_pos[-30:] if len(bball_pos) > 30 else bball_pos
            x_coords, y_coords = zip(*recent_points)
            coeffs = np.polyfit(x_coords, y_coords, 2)
            x_range = range(min(x_coords), max(x_coords) + 1, 5)
            y_values = np.polyval(coeffs, x_range).astype(int)
            valid_points = [(x, y) for x, y in zip(x_range, y_values) if 0 <= x < frame_x and 0 <= y < frame_y]
            for x, y in valid_points:
                cv2.circle(new_frame, (x, y), 2, (0, 255, 255), -1)
            if len(x_coords) > 0:
                last_x = x_coords[-1]
                next_x_values = np.arange(last_x + 5, last_x + 5 * 50, 5)
                next_y_values = np.polyval(coeffs, next_x_values).astype(int)
                next_valid_points = [(x, y) for x, y in zip(next_x_values, next_y_values) if 0 <= x < frame_x and 0 <= y < frame_y]
                red_points_inside_pad = []
                for x, y in next_valid_points:
                    if pad_x1y1[0] <= x <= pad_x2y2[0] and pad_x1y1[1] <= y <= pad_x2y2[1]:
                        cv2.circle(new_frame, (x, y), 2, (0, 0, 255), -1)
                        red_points_inside_pad.append((x, y))
                    else:
                        cv2.circle(new_frame, (x, y), 2, (0, 255, 0), -1)
                if len(red_points_inside_pad) >= 2:
                    first_red_dot = red_points_inside_pad[0]
                    last_red_dot = red_points_inside_pad[-1]
                    top_left = (min(first_red_dot[0], last_red_dot[0]), min(first_red_dot[1], last_red_dot[1]))
                    bottom_right = (max(first_red_dot[0], last_red_dot[0]), max(first_red_dot[1], last_red_dot[1]))
                    if not (bottom_right[0] < basket_x1y1[0] or top_left[0] > basket_x2y2[0] or
                            bottom_right[1] < basket_x1y1[1] or top_left[1] > basket_x2y2[1]):
                        color = (0, 255, 255)
                        cv2.putText(new_frame, "SCORE", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    else:
                        color = (255, 0, 0)
                    cv2.rectangle(new_frame, top_left, bottom_right, color, 2)
                    cv2.rectangle(new_frame, basket_x1y1, basket_x2y2, (255, 0, 255), 2)
                for x, y in next_valid_points:
                    if pad_x1y1[0] <= x <= pad_x2y2[0] and pad_x1y1[1] <= y <= pad_x2y2[1]:
                        cv2.putText(new_frame, "HIT", (pad_x1y1[0] + 10, pad_x1y1[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        break

        # Predict a projectile motion of a basketball from it's launch point
        target_x1y1, target_x2y2 = basket_x1y1, basket_x2y2
        if target_x1y1 == (0, 0) or target_x2y2 == (0, 0):
            target_x1y1, target_x2y2 = pad_x1y1, pad_x2y2

        if (bball_pos or (bball_cx != 0 and bball_cy != 0) and (bball_cx != 320 and bball_cy != 240)) and (target_x1y1 != (0, 0) and target_x2y2 != (0, 0)):
            # Use the first basketball position if available, else current center
            if bball_pos:
                start_x, start_y = bball_pos[0]
            else:
                start_x, start_y = bball_cx, bball_cy
            # Target center (basket or pad)
            target_center_x = (target_x1y1[0] + target_x2y2[0]) // 2
            target_center_y = (target_x1y1[1] + target_x2y2[1]) // 2
            # Control point above the midpoint for the curve
            mid_x = (start_x + target_center_x) // 2
            mid_y = min(start_y, target_center_y) - 50
            curve_points = [(start_x, start_y), (mid_x, mid_y), (target_center_x, target_center_y)]
            curve_x, curve_y = zip(*curve_points)
            coeffs = np.polyfit(curve_x, curve_y, 2)
            x_range = range(min(curve_x), max(curve_x) + 1, 5)
            y_values = np.polyval(coeffs, x_range).astype(int)
            valid_points = [(x, y) for x, y in zip(x_range, y_values) if 0 <= x < frame_x and 0 <= y < frame_y]
            for x, y in valid_points:
                cv2.circle(new_frame, (x, y), 2, (255, 0, 0), -1)

    bball_px = bball_cx
    bball_py = bball_cy

# Track 3D basketball pad and estimate projectile trajectory
def bpadTrk():
    global frame_cx, frame_cy
    global pad_cx, pad_cy, pad_x1y1, pad_x2y2
    global basket_x1y1, basket_x2y2
    global basketball_x1y1, basketball_x2y2
    global angle_deg, angle_deg_arc

    # Draw center and shooter points
    cv2.circle(new_frame, (frame_cx, frame_cy), 6, (255, 0, 0), -1)
    shooter_pts = [(-150, -150), (150, -150)]
    shooter_colors = [(255, 255, 0), (0, 255, 0)]
    for pt, color in zip(shooter_pts, shooter_colors):
        cv2.circle(new_frame, (pt[0] + frame_cx, frame_cy - pt[1]), 6, color, -1)
    cv2.line(new_frame, (shooter_pts[0][0] + frame_cx, frame_cy - shooter_pts[0][1]),
             (shooter_pts[1][0] + frame_cx, frame_cy - shooter_pts[1][1]), (255, 0, 0), 2)
    cv2.rectangle(new_frame, basket_x1y1, basket_x2y2, (255, 0, 255), 2)
    cv2.rectangle(new_frame, basketball_x1y1, basketball_x2y2, (0, 255, 255), 2)

    # Camera tilt angle (not used, placeholder)
    camera_angle_deg = 0

    # Only draw if basket detected

    interest_x1y1 = pad_x1y1 #if pad_x1y1 != (0, 0) else basket_x1y1
    interest_x2y2 = pad_x2y2 #if pad_x2y2 != (0, 0) else basket_x2y2
    if interest_x1y1 != (0, 0) and interest_x2y2 != (0, 0):
        # Estimate distance: large box = near, small box = far
        box_w = interest_x2y2[0] - interest_x1y1[0]
        box_h = interest_x2y2[1] - interest_x1y1[1]
        box_area = box_w * box_h
        min_area, max_area = 500, 30000
        area_norm = np.clip((box_area - min_area) / (max_area - min_area), 0, 1)

        # Text and minimap layout
        bottom_margin, line_height = 10, 20
        y_vals = [frame_y - bottom_margin - i * line_height for i in range(2, 9)]
        x_text = 20 if frame_x > 200 else 5
        minimap_w, minimap_h = 220, 220
        offset_x, offset_y = 10, 10
        cam_pos = (offset_x + 20, offset_y + minimap_h // 2 + 50)
        cv2.circle(new_frame, cam_pos, 8, (255, 255, 255), -1)
        cv2.putText(new_frame, "Camera sideview", (cam_pos[0] - 10, cam_pos[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        min_dist, max_dist = 40, minimap_w - 40
        basket_dist = int(min_dist + (1 - area_norm) * (max_dist - min_dist))
        basket_center_y = (interest_x1y1[1] + interest_x2y2[1]) // 2
        basket_height_px = frame_cy - basket_center_y
        max_offset = frame_y // 2
        minimap_offset = int(np.clip((basket_height_px / max_offset) * 80, -80, 80))
        global basket_pos
        basket_pos = (cam_pos[0] + basket_dist, cam_pos[1] - minimap_offset)

        cv2.rectangle(new_frame, (basket_pos[0] - 15, basket_pos[1] - 10), (basket_pos[0] + 15, basket_pos[1] + 10), (255, 0, 255), 2)
        cv2.putText(new_frame, "Basket", (basket_pos[0] - 30, basket_pos[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.line(new_frame, cam_pos, basket_pos, (255, 255, 0), 2)

        cv2.putText(new_frame, f"Distance: {basket_dist}px", (x_text, y_vals[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(new_frame, f"Height: {basket_height_px:+d}px", (x_text, y_vals[4]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        basket_center_x = (interest_x1y1[0] + interest_x2y2[0]) // 2
        basket_center_y = (interest_x1y1[1] + interest_x2y2[1]) // 2
        rel_x = basket_center_x - frame_x // 2
        rel_y = basket_center_y - frame_y // 2

        # Draw basket info at the bottom of the screen if space allows
        basket_info_1 = f"Basket rel: x={rel_x:+d}, y={-1 * rel_y:+d}"
        basket_info_2 = f"basket_pos on frame: x={basket_pos[0]}, y={basket_pos[1]}"
        # Calculate y position for bottom lines
        bottom_margin = 10
        line_height = 22
        y1 = frame_y - bottom_margin - line_height * 1
        y2 = frame_y - bottom_margin
        cv2.putText(new_frame, basket_info_1, (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(new_frame, basket_info_2, (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Physics-based trajectory
        g, v0, pixels_per_meter = 9.81, 5, 40.0
        dx_px = (basket_pos[0] - cam_pos[0]) + 10
        dy_px =  (basket_pos[1] - cam_pos[1])
        dx = dx_px / pixels_per_meter
        dy = -dy_px / pixels_per_meter

        cv2.putText(new_frame, f"{basket_pos[1]} - {cam_pos[1]} = {basket_pos[1] - cam_pos[1]}", (x_text, y_vals[6]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(new_frame, f"dx: {dx_px} px, dy: {dy_px} px", (x_text, y_vals[5]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        min_deg, max_deg = 45, 90
        min_arc, max_arc = 20, 60
        arc_height = int(min_arc + (max_arc - min_arc) * (basket_dist - min_dist) / (max_dist - min_dist))
        arc_height = np.clip(arc_height, min_arc, max_arc)

        angle_rad = None
        if dx != 0 and v0 > 0:
            v0_sq = v0 * v0
            under_sqrt = v0_sq * v0_sq - g * (g * dx * dx + 2 * dy * v0_sq)
            if under_sqrt >= 0:
                sqrt_val = np.sqrt(under_sqrt)
                theta1 = np.arctan((v0_sq + sqrt_val) / (g * dx))
                theta2 = np.arctan((v0_sq - sqrt_val) / (g * dx))
                min_rad, max_rad = np.radians(min_deg), np.radians(max_deg)
                angles = [theta for theta in [theta1, theta2] if min_rad < theta < max_rad]
                if angles:
                    angle_rad = min(angles)

        if angle_rad is not None:
            num_points = 50
            theta = angle_rad
            angle_deg = np.degrees(theta)
            angle_norm = np.clip((angle_deg - min_deg) / (max_deg - min_deg), 0, 1)
            dist_norm = np.clip((basket_dist - min_dist) / (max_dist - min_dist), 0, 1)
            arc_factor = 0.6 * dist_norm + 0.4 * angle_norm
            arc_offset = int(min_arc + arc_factor * (max_arc - min_arc))
            arc_offset = np.clip(arc_offset, min_arc, max_arc)
            for i in range(num_points + 1):
                t = i / num_points
                x = dx_px * t
                x_m = x / pixels_per_meter
                y_m = x_m * np.tan(theta) - (g * x_m * x_m) / (2 * v0 * v0 * np.cos(theta) ** 2)
                y = -y_m * pixels_per_meter
                arc_y = y - arc_offset * np.sin(np.pi * t)
                px = int(round(cam_pos[0] + x))
                py = int(round(cam_pos[1] + arc_y))
                if 0 <= px < frame_x and 0 <= py < frame_y:
                    cv2.circle(new_frame, (px, py), 2, (0, 128, 255), -1)
            cv2.circle(new_frame, basket_pos, 3, (0, 255, 255), -1)

            cv2.putText(new_frame, f"Possible score at: {angle_deg:.1f} deg", (x_text, y_vals[2]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        else:
            # Visual arc if no valid angle
            mid_x = (cam_pos[0] + basket_pos[0]) // 2
            mid_y = (cam_pos[1] + basket_pos[1]) // 2 - arc_height
            arc_points = [cam_pos, (mid_x, mid_y), basket_pos]
            arc_curve_x, arc_curve_y = zip(*arc_points)
            coeffs = np.polyfit(arc_curve_x, arc_curve_y, 2)
            x_range = np.linspace(cam_pos[0], basket_pos[0], 50)
            y_values = np.polyval(coeffs, x_range)
            for x, y in zip(x_range, y_values):
                px, py = int(round(x)), int(round(y))
                if 0 <= px < frame_x and 0 <= py < frame_y:
                    cv2.circle(new_frame, (px, py), 2, (0, 0, 255), -1)
            a, b, c = coeffs
            x0 = cam_pos[0]
            slope = 2 * a * x0 + b
            angle_rad_arc = np.arctan(slope)
            angle_deg_arc = np.degrees(angle_rad_arc)
            cv2.putText(new_frame, f"Try: {-1 * angle_deg_arc:.1f} deg with higher shooting speed", (x_text, y_vals[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        avg_angle = (angle_deg + (-1 * angle_deg_arc)) / 2
        
            
    pass

try:

    threading.Thread(target=read_frames, daemon=True).start()
    threading.Thread(target=printUART, daemon=True).start()
    while True:
        time_now = time.time()
        elapsed_time = time_now - start_time
        not_ok = ok = 0
        with frame_lock:
            if frame is None:
                continue
            # Only resize if model requires it, otherwise use as is
            current_frame = frame
            
        new_frame = np.zeros((frame_y, frame_x, 3), dtype=np.uint8)
        results = model(current_frame, device=device, verbose=False)
        is_basketball = False
        
        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            conf = detection.conf[0    ].item()
            class_id = int(detection.cls[0].item())
            name = model.names[class_id]
            label = f"{name} {class_id} {conf:.2f}"
            center_color = (0, 0, 255)
            center_radius = 3
            dot_distance_pixels = 5
            dot_radius = 2
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            text_w, text_h = text_size

            if conf > 0.0 and name != 'face':
                if name == 'face':
                    face_cnt += 1
                if conf < 0.5:
                    if name == "face":
                        center_color = (255, 0, 255)
                    not_ok += 1
                    color = (0, 0, 255)
                    text_col = (0, 0, 255)

                else:
                    ok += 1
                    color = (0, 255, 0)
                    text_col = (255, 255, 255)

                    if name == "face":
                        center_color = (255, 255, 0)
                        
                    elif name == "basketball":
                        color = (0, 255, 255)
                        center_color = (0, 255, 255)
                        cv2.line(current_frame, (frame_x // 2, frame_y // 2), (cx, cy), (255, 255, 255), 2)

                        cv2.rectangle(new_frame, (x1, y1), (x2, y2), color, thickness=2)
                        cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, thickness=2)

                    elif name == "pad":
                        center_color = (0, 255, 0)
                        pad_x1y1 = (x1, y1)
                        pad_x2y2 = (x2, y2)
                        pad_cx = cx
                        pad_cy = cy
                        cv2.line(current_frame, (frame_x // 2, frame_y // 2), (cx, cy), (0, 255, 255), 2)

                        cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, thickness=2)

                        #cv2.rectangle(new_frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
                        #cv2.putText(new_frame, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), text_thickness)

                    elif name == "basket":
                        center_color = (255, 0, 255)
                        color = (255, 0, 255)

                        basket_x1y1 = (x1, y1)
                        basket_x2y2 = (x2, y2)

                        cv2.line(current_frame, (frame_x // 2, frame_y // 2), (cx, cy), (0, 255, 255), 2)
                        cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, thickness=2)

                    else:
                        if conf > 0.7:
                            cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, thickness=2)
                    
                if bball_cx != 320 and bball_cy != 240 and pad_cx != 320 and pad_cy != 240:
                    cv2.line(current_frame, (pad_cx, pad_cy), (bball_cx, bball_cy), (255, 255, 0), 2)
                    cv2.putText(current_frame, f"CONNECT", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.rectangle(current_frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)

                if name == "basketball" and conf > 0.5:
                    basketball_x1y1 = (x1, y1)
                    basketball_x2y2 = (x2, y2)
                    bball_cx = cx
                    bball_cy = cy

                cv2.putText(current_frame, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), text_thickness)
                cv2.putText(current_frame, f"{cx - (frame_x // 2)} : {-1 * (cy - (frame_y // 2))}", (x1, y1 - 20), font, 0.5, text_col, 2)
            
                cv2.circle(current_frame, (cx, cy), center_radius, center_color, 2)

        cv2.circle(current_frame, (frame_x // 2, frame_y // 2), 3, (255, 0, 0), 2)

        # Only recreate telemetry_frame if needed
        if 'telemetry_frame' not in locals():
            telemetry_frame = np.zeros((frame_y, frame_y, 3), dtype=np.uint8)
        else:
            telemetry_frame.fill(0)
        
        telemetry_texts = [
            f"Device: {device}",
            f"Frame: {int(frame_count)}",
            f"FPS: {int(prev_frames)}",
            f"Video Frame: {int(video_frame)}",
            f"Read FPS: {int(read_frame_fps)}",
            f"Video Path: {vid_path}",
            f"OK: {ok}",
            f"NOT OK: {not_ok}",
            f"Frame time: {frametime * 1000:.0f}ms"
        ]

        # Toggle save_key every 1 second in non-blocking way if save_pos is True
        if 'save_key' not in globals():
            save_key = False
        if 'last_save_key_toggle' not in globals():
            last_save_key_toggle = time.time()
        if save_pos == True:
            now = time.time()
            if now - last_save_key_toggle >= 0.5:
                save_key = not save_key
                last_save_key_toggle = now

            if save_key:
                cv2.putText(telemetry_frame, "Saving snapshot", (10, 30 + (len(telemetry_texts)) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(telemetry_frame, "Snapshot paused", (10, 30 + (len(telemetry_texts)) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if rstat == 'failed':
            cv2.putText(telemetry_frame, "Robot is not connected", (10, 30 + (len(telemetry_texts) + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif rstat == 'send OK':
            cv2.putText(telemetry_frame, f"Connected to robot via {serial_port}", (10, 30 + (len(telemetry_texts) + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(telemetry_frame, "Connected, but with limited functionality", (10, 30 + (len(telemetry_texts) + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        for i, text in enumerate(telemetry_texts):
            cv2.putText(telemetry_frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(current_frame, "[C] to clear basketball points", (10, frame_y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(current_frame, "[R] to reset video", (10, frame_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(current_frame, "[T] to switch video", (10, frame_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(current_frame, f"[S] to save basketball POS", (10, frame_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            bball_pos.clear()
            print("Basketball points cleared.")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bball_pos.clear()
            print("Video reset.")
        elif key == ord('t'):
            current_video_number = int(vid_path.split("test")[1].split(".mp4")[0]) if isinstance(vid_path, str) else vid_path
            next_video_number = current_video_number + 1 if current_video_number < 10 else 1
            vid_path = f"files/test{next_video_number}.mp4" if isinstance(vid_path, str) else next_video_number
            cap.release()
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video source {vid_path}.")
                while True:
                    current_frame = np.zeros((frame_y, frame_x, 3), dtype=np.uint8)
                    telemetry_frame = np.zeros((frame_y, frame_y, 3), dtype=np.uint8)
                    cv2.putText(current_frame, "Video not found. Press T to retry video path.", (10, frame_y // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(current_frame, "Press V to switch to camera mode.", (10, frame_y // 2 - 0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    srcstr = "Video" if isinstance(vid_path, str) else "Camera"
                    cv2.putText(current_frame, f"Current {srcstr}: {vid_path}", (10, frame_y // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(telemetry_frame, "Refer to main window for instructions.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow('Telemetry', telemetry_frame)
                    cv2.imshow(f'YOLOv8 Detection ({device})', current_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('t'):
                        if isinstance(vid_path, int):
                            vid_path = f"files/test0.mp4"
                        current_video_number = int(vid_path.split("test")[1].split(".mp4")[0])
                        next_video_number = current_video_number + 1 if current_video_number < 10 else 1
                        vid_path = f"files/test{next_video_number}.mp4"
                        cap.release()
                        cap = cv2.VideoCapture(vid_path)
                        if cap.isOpened():
                            print(f"Switched to video: {vid_path}")
                            break
                    if key == ord('v'):
                        if isinstance(vid_path, int):
                            vid_path += 1
                        if isinstance(vid_path, str):
                            vid_path = 0
                        if vid_path > 5:
                            vid_path = 0
                        cap.release()
                        cap = cv2.VideoCapture(vid_path)
                        if cap.isOpened():
                            print(f"Switched to camera: {vid_path}")
                            break
            else:
                bball_pos.clear()
                print(f"Switched to video: {vid_path}")
        if key == ord('s'):
            save_pos = not save_pos
            print(f"save_pos toggled to: {save_pos}")
        
        if len(bball_pos) > 100:
            if bball_pos:
                last_pos = bball_pos[-1]
                bball_pos.clear()
                bball_pos.append(last_pos)

        bballTrk()
        bpadTrk()

        rsx = bball_cx - (frame_x // 2)
        rsy = -1 * (bball_cy - (frame_y // 2))
        rcx = pad_cx - (frame_x // 2)
        rcy = -1 * (pad_cx - (frame_x // 2))
        if rsx > 45:
            rsx = 45
        elif rsx < -45:
            rsx = -45

        frametime = round(elapsed_time - prev_time, 3)
        print(f"[{elapsed_time:.2f}ms] [frame {frame_count}] [{prev_frames} fps] [Δ{frametime:.2}ms] || UART >> {rstat} ; x:{rsx} y:{rsy} | x:{rcx} y:{rcy}")
        frame_count += 1
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            prev_frames = frame_count
            frame_count = 0
            start_time = time.time()

        prev_time = elapsed_time
        cv2.imshow(f'YOLOv8 Detection ({device})', current_frame)
        cv2.imshow(f'Pixel array detect', new_frame)
        cv2.imshow('Telemetry', telemetry_frame)

        pad_cx = frame_cx
        pad_cy = frame_cy
        bball_cx = frame_cx
        bball_cy = frame_cy
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    stop_threads = True
    cap.release()
    cv2.destroyAllWindows()