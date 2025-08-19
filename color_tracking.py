import cv2
import numpy as np
import psutil
import time
from typing import Final
import multiprocessing

# Settings value
expose_value = "AUTO"
RGB = [175, 176, 107]
OG_COLOR = [RGB[0], RGB[1], RGB[2]]
camera_index = "files/test4.mp4"

# UART
send_uart = False
uart_port = "/dev/ttyUSB0"
uart_packet = "default"
#ser = serial.Serial(uart_port, 9600)

# Modes
modes = 0 # 0 norm , 1 auto , 2 = acquis
is_menu = False
select_menu = 0 # Selected menu section
select_header = 0 # Main category section
menu_header = "MENU"
menu_item = [
    ["Device", "UART", "UART path", "UART packet", "White balance", "Exposure" ,"Colors to detect"],
    ["Velocity tracking", "Predictive guidance", "Track new color", "About"]
]
settings_value = [
    [camera_index, "OFF", f"{uart_port}", f"{uart_packet}", "AUTO", f"{expose_value}", f"{RGB[0]} {RGB[1]} {RGB[2]}"],
    ["ON", "ON", ">PRESS ENTER<","Created by Net Zamora"]
]
settings_error = [False, False, False, False, False, False, False]
last_mode = 0
gui = True # Whether the program runs in GUI or not



if gui is not True:
    print("Gui is disabled!")
    time.sleep(1)

# Define a wider range of yellow color to track (in HSV)
col_choice = np.uint8([[[RGB[2], RGB[1], RGB[0]]]])
hsvChoice = cv2.cvtColor(col_choice, cv2.COLOR_BGR2HSV)
lower_col1 = np.array([hsvChoice[0][0][0] - 10, 100, 100])  # Lower bound
upper_col1 = np.array([hsvChoice[0][0][0] + 10, 255, 255])  # Upper bound

def updateColor():
    global lower_col1, upper_col1, col_choice, hsvChoice
    col_choice = np.uint8([[[RGB[2], RGB[1], RGB[0]]]])
    hsvChoice = cv2.cvtColor(col_choice, cv2.COLOR_BGR2HSV)
    lower_col1 = np.array([hsvChoice[0][0][0] - 10, 100, 100])  # Lower bound
    upper_col1 = np.array([hsvChoice[0][0][0] + 10, 255, 255])  # Upper bound

use_col2 = False
lower_col2 = np.array([0, 0, 0])
upper_col2 = np.array([0, 0, 0])

noise_reduct = True

# Start the webcam
cap = cv2.VideoCapture(camera_index)
#cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Set auto white balance off
if expose_value == "AUTO":
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    try:
        cap.set(cv2.CAP_PROP_EXPOSURE, expose_value)
    except:
        pass
kernel = np.ones((5, 5), np.uint8)

# Variables to track previous position and velocity
prev_center = None
velocity = (0, 0)
fps = 30
frame_count = 0
start_time = time.time()

# x & y
i = 0
frame_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_cx = int(frame_x / 2)
frame_cy = int(frame_y / 2)
rect_width, rect_height = 200, 100
center_x, center_y = frame_x // 2, frame_y // 2

col_width, col_height = 100, 100

aquis_top = (center_x - rect_width // 2, center_y - rect_height // 2)
aquis_bot = (center_x + rect_width // 2, center_y + rect_height // 2)
col_top = ((center_x + 100) - col_width // 2, center_y - col_height // 2)
col_bot = ((center_x + 100)+ col_width // 2, center_y + col_height // 2)
real_object_width = 10.0
focal_length = 38

# Tracking
is_trk = False
is_iog = False
trk_x = trk_y = trk_w = trk_h = trk_veloc = 0
trk_ae = trk_cx = trk_cy = 0
los_cnt = 0

def trk_obj():
    global modes, frame_count, fps, start_time, prev_center, velocity, aquis_top, aquis_bot,\
           is_trk, trk_x, trk_y, trk_w, trk_h,\
           trk_veloc, i
    # Find contours in the mask
    contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0

    # First pass: find the largest contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area to consider
            if area > largest_area:
                largest_area = area
                largest_contour = contour

    # Second pass: highlight the largest contour and others
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  
            x, y, w, h = cv2.boundingRect(contour)

        
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            distance = (real_object_width * focal_length) / w if w > 0 else float('inf')

            rgb_value = frame[center_y, center_x - 11]

            
            if contour is largest_contour:
                cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)  # RED
                cv2.circle(hbox, (center_x, center_y), 10, (0, 0, 255), -1)   # RED

                
                if prev_center is not None:
                    distance = np.sqrt((center_x - prev_center[0]) ** 2 + (center_y - prev_center[1]) ** 2)
                    
                    frame_time = 1 / fps
                    velocity = ((center_x - prev_center[0]) / frame_time, (center_y - prev_center[1]) / frame_time)

                    
                    arrow_end = (int(center_x + velocity[0] // 2), int(center_y + velocity[1] // 2))
                    cv2.arrowedLine(frame, (center_x, center_y), arrow_end, (255, 0, 0), 2, tipLength=0.2)
                    cv2.arrowedLine(hbox, (center_x, center_y), arrow_end, (255, 0, 0), 2, tipLength=0.2)

            
                prev_center = (center_x, center_y)
                # Display
                i += 1
                cv2.putText(frame, f"TRK",
                            (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                cv2.putText(frame, f"Width: {w:.2f} px", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Distance: {distance:.2f} cm", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Coords: ({center_x}, {center_y})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"RGB: ({rgb_value[2]}, {rgb_value[1]}, {rgb_value[0]})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Velocity: {np.linalg.norm(velocity):.2f} px/s", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            else:
                i+= 1
                if np.linalg.norm(velocity) < 40:
                    cv2.putText(frame, f"DETECT",
                            (x,y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255 ,0), 2) # GREEN
                    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)  # GREEN
                    cv2.circle(hbox, (center_x, center_y), 10, (0, 255, 0), -1)  # GREEN
                else:
                    cv2.putText(frame, f"MOVING",
                            (x,y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # BLUE
                    cv2.circle(frame, (center_x, center_y), 10, (255, 0, 0), -1)  # BLUE
                    cv2.putText(hbox, f"MOVING",
                            (x,y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # BLUE
                    cv2.circle(hbox, (center_x, center_y), 10, (255, 0, 0), -1)  # BLUE

                cv2.putText(hbox, f"{i}", 
                        (center_x - 20, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.putText(frame, f"Coords: ({center_x}, {center_y})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"RGB: ({rgb_value[2]}, {rgb_value[1]}, {rgb_value[0]})", 
                        (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Velocity: {np.linalg.norm(velocity):.2f} px/s", 
                        (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display coordinates, RGB, velocity
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(hbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)


def trk():
    global modes, frame_count, fps, start_time, prev_center, velocity, aquis_top, aquis_bot,\
           is_trk, trk_x, trk_y, trk_w, trk_h, trk_ae, trk_cx, trk_cy,\
           trk_veloc, i, los_cnt
    # Find contours in the mask
    contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0

    # First pass: find the largest contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area to consider
            if area > largest_area:
                largest_area = area
                largest_contour = contour

    # Second pass: highlight the largest contour and others
    for contour in contours:
        area = cv2.contourArea(contour)

        # Determine shape
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if area > 500:  
            x, y, w, h = cv2.boundingRect(contour)

        
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            distance = (real_object_width * focal_length) / w if w > 0 else float('inf')

            rgb_value = frame[center_y, center_x - 11]

            
            if contour is largest_contour:
                cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)  # RED
                cv2.circle(hbox, (center_x, center_y), 10, (0, 0, 255), -1)   # RED

                
                if prev_center is not None:
                    distance = np.sqrt((center_x - prev_center[0]) ** 2 + (center_y - prev_center[1]) ** 2)
                    
                    frame_time = 1 / fps
                    velocity = ((center_x - prev_center[0]) / frame_time, (center_y - prev_center[1]) / frame_time)

                    
                    arrow_end = (int(center_x + velocity[0] // 2), int(center_y + velocity[1] // 2))
                    trk_ae = arrow_end
                    cv2.arrowedLine(frame, (center_x, center_y), arrow_end, (255, 0, 0), 2, tipLength=0.2)
                    cv2.arrowedLine(hbox, (center_x, center_y), arrow_end, (255, 0, 0), 2, tipLength=0.2)

            
                prev_center = (center_x, center_y)

                if is_trk is False:
                    is_trk = True
                # Update trk variables
                trk_cx = center_x
                trk_cy = center_y
                trk_x = x
                trk_y = y
                trk_w = w
                trk_h = h
                trk_veloc = velocity
                los_cnt = 1

                i += 1

                # Display
                cv2.putText(frame, f"TRK",
                            (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                cv2.putText(frame, f"TRK",
                            (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame, f"Width: {w:.2f} px", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Distance: {distance:.2f} cm", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Coords: ({center_x}, {center_y})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"RGB: ({rgb_value[2]}, {rgb_value[1]}, {rgb_value[0]})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Velocity: {np.linalg.norm(velocity):.2f} px/s", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)
                cv2.drawContours(hbox, [approx], -1, (0, 255, 255), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(hbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

def los_trk():
    global modes, frame_count, fps, start_time, prev_center, velocity, aquis_top, aquis_bot,\
           is_trk, trk_x, trk_y, trk_w, trk_h, trk_ae, trk_cx, trk_cy,\
           trk_veloc, i, los_cnt
    
    if los_cnt > 0 and los_cnt < 10:
        los_cnt += 1

        x = trk_x; y = trk_y; w = trk_w; h = trk_h
        arrow_end = trk_ae; center_x = trk_cx; center_y = trk_cy

        cv2.putText(frame, f"PRED",
                                (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(hbox, f"PRED",
                                (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        try:
            cv2.arrowedLine(frame, (center_x, center_y), arrow_end, (255, 0, 0), 2, tipLength=0.2)
            cv2.arrowedLine(hbox, (center_x, center_y), arrow_end, (255, 0, 0), 2, tipLength=0.2)
        except:
            cv2.putText(frame, f"ERROR", (frame.shape[1] - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.rectangle(hbox, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    else:
        trk_x = 0; trk_y = 0; trk_w = 0; trk_h = 0
        trk_cx = 0; trk_cy = 0; trk_ae = 0
        los_cnt = 0

def user_input(index:int):
    global RGB, OG_COLOR
    in_prompt = True
    submenu = 0
    while in_prompt == True:
        frame[:] = 0
        key =  cv2.waitKey(1)
        if key == 13:
            print("ENTER KEY")
            in_prompt = False
            OG_COLOR = [RGB[0], RGB[1], RGB[2]]
            settings_value[0][6] = f"{RGB[0]} {RGB[1]} {RGB[2]}"
            updateColor()
        if key == 27: # Escape key
            print("ESCAPE KEY")
            RGB = [OG_COLOR[0], OG_COLOR[1], OG_COLOR[2]]   
            in_prompt = False
            
        cv2.putText(frame, f"MENU", (frame_cx - 200, frame_cy - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"{menu_item[select_header][index]}", (frame_cx - 200, (frame_cy - 70) + (0 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"{OG_COLOR} : {RGB}")
        if menu_item[select_header][index] == "Colors to detect":
            if key == ord('a'):
                if RGB[submenu] <= 0:
                    pass
                else:
                    RGB[submenu] -= 1
            elif key == ord('d'):
                if RGB[submenu] >= 255:
                    pass
                else:
                    RGB[submenu] += 1
            for j in range(len(RGB)):
                if submenu == j:
                    cv2.putText(frame, f"{'RGB'[j]}> {RGB[j]}", (frame_cx - 200, (frame_cy - 60) + ((j+1) * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, f"{'RGB'[j]} {RGB[j]}", (frame_cx - 200, (frame_cy - 60) + ((j+1) * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            sample_col = (RGB[2], RGB[1], RGB[0])
            
            cv2.putText(frame, f"Color", (frame_cx + 50, (frame_cy - 70) + (0 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(frame, col_top, col_bot, sample_col, thickness=-1)
        else:
            cv2.putText(frame, f"{settings_value[select_header][index]}_______", (frame_cx - 200, (frame_cy - 70) + (1 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if menu_item[select_header][index] == "Exposure":
                cv2.putText(frame, f"           [v]        ", (frame_cx - 200, (frame_cy - 70) + (3 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"[... -2 -1 0 1 2 ...] [AUTO is enabled]", (frame_cx - 200, (frame_cy - 70) + (4 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if key == ord('w'):  # UP arrow key
            if submenu <= 0:
                pass
            else:
                submenu -= 1
        elif key == ord('s'):  # DOWN arrow key
            if submenu == 2:
                pass
            else:
                submenu += 1
        
                
        cv2.putText(frame, f"[ESC] Discard", (frame_x- 300, frame_y- 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"[ENTER] Save", (frame_x- 150, frame_y- 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Full", frame)

def user_menu():
    cv2.putText(frame, f"{menu_header}", (frame_cx - 200, frame_cy - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame, f"[N] Return", (frame_x- 120, frame_y- 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i in range(len(menu_item[select_header])):
        if select_menu == i:
            cv2.putText(frame, f"> {menu_item[select_header][i]}", (frame_cx - 200, (frame_cy - 70) + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"{menu_item[select_header][i]}", (frame_cx - 200, (frame_cy - 70) + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{settings_value[select_header][i]}", (frame_cx + 50, (frame_cy - 70) + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"----------------------", (frame_cx - 200, (frame_cy - 70) + (7 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if select_header == 0:
        cv2.putText(frame, f"[Camera settings]", (frame_cx - 200, (frame_cy - 70) + (8 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"[D] Features", (frame_cx + 50, (frame_cy - 70) + (8 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif select_header == 1:
        cv2.putText(frame, f"[A] Camera settings", (frame_cx - 200, (frame_cy - 70) + (8 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"[Features]", (frame_cx + 50, (frame_cy - 70) + (8 * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if key == 13:
        user_input(select_menu)

while True:
    # Capture frame-by-frame
    ret, original = cap.read()
    if not ret:
        break

    frame = original.copy()
    key = cv2.waitKey(1) & 0xFF
    
    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # Update FPS every second
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    if is_menu == True:
        user_menu()
        if key == ord('w'):  # UP arrow key
            if select_menu <= 0:
                pass
            else:
                select_menu -= 1
        elif key == ord('s'):  # DOWN arrow key
            if select_menu == len(menu_item[select_header]) - 1:
                pass
            else:
                select_menu += 1
        if key == ord('a'):  # PREVIOUS category
            if select_header <= 0:
                pass
            else:
                select_header -= 1
        elif key == ord('d'):  # NEXT category
            if select_header == len(menu_item) - 1:
                pass
            else:
                select_header += 1

    elif is_menu == False:
        # Hitboxes where it only shows tracking data
        hbox = np.zeros((frame_y, frame_x, 3), dtype=np.uint8)

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the specified yellow color
        mask1 = cv2.inRange(hsv, lower_col1, upper_col1)

        if noise_reduct is True:
            mask1 = cv2.erode(mask1, kernel, iterations=1)  # Erode
            mask1 = cv2.dilate(mask1, kernel, iterations=1)  # Dilate

        if use_col2 == True:
            mask2 = cv2.inRange(hsv, lower_col2, upper_col2)
            mask2 = cv2.erode(mask2, kernel, iterations=1)  # Erode
            mask2 = cv2.dilate(mask2, kernel, iterations=1)  # Dilate
            mask1 = cv2.bitwise_or(mask1, mask2)

        # Bitwise-AND mask and original image
        result = cv2.bitwise_and(frame, frame, mask=mask1)

        if modes == 0:
            trk_obj()
            cv2.putText(frame, f"NORM", (frame.shape[1] - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        elif modes == 1:
            trk()
            if is_trk is False:
                is_iog = True
                if los_cnt > 0:
                    cv2.putText(hbox, f"Inertial tracking", (frame.shape[1] - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(hbox, f"Target lost", (frame.shape[1] - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                los_trk()
            else:
                is_iog = False

            cv2.putText(frame, f"PREDICT_AUTO", (frame.shape[1] - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            pass

        elif modes == 2:
            cv2.putText(frame, f"ACQUIS", (frame.shape[1] - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, aquis_top, aquis_bot, (0, 0, 255), 2)
            cv2.rectangle(hbox, aquis_top, aquis_bot, (0, 0, 255), 2)
            cv2.rectangle(result, aquis_top, aquis_bot, (0, 0, 255), 2)
        cpu_usage = psutil.cpu_percent()
        # Display CPU, GPU usage, and FPS at the top right
        cv2.putText(frame, f"CPU: {cpu_usage}%", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(hbox, f"Objects: {i}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(hbox, f"Track: {is_trk}", (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    TO_SEND = f"{trk_cx - frame_cx} {trk_cy - frame_cy} {los_cnt}"
    #ser.write(TO_SEND.encode('utf-8'))
    
    if gui == True:
        cv2.imshow('Full', frame)
        #cv2.imshow('Feed', original)
        #cv2.imshow('Hitboxes', hbox)
        #cv2.imshow('Mask', mask1)
        #cv2.imshow('Detected Color', result)
    else:
        if last_mode == modes:
            pass
        else:
            print(f"Opencv in mode {modes}!")

    i = 0
    is_trk = False
    if key == ord('q'):
        break

    elif key == ord('n'):
        if is_menu == True:
            is_menu = False
        else:
            is_menu = True


    elif key == ord('m'):
        print("m pressed!")
        while cv2.waitKey(1) & 0xFF == ord('m'):
            pass
        if modes < 2:
            modes += 1
        else:
            modes = 0

# Release the capture and close windows
cap.release()

cv2.destroyAllWindows()