# one lane

# from flask import Flask, render_template, Response, jsonify
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import defaultdict
# import time
# import threading

# app = Flask(__name__)

# # Initialize YOLOv8 model
# model = YOLO('yolov8n.pt')  # Use nano model for speed, good for prototyping

# # Define vehicle classes of interest (COCO Dataset Class IDs)
# # 2: car, 3: motorcycle, 5: bus, 7: truck
# VEHICLE_CLASSES = [2, 3, 5, 7]
# class_names = model.names

# # Global variables to store data for the dashboard
# traffic_data = {
#     'current_counts': {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0},
#     'total_count': 0,
#     'density_history': [],  # List of total vehicle counts over time
#     'timing_history': [],   # List of green light timings over time
#     'time_labels': []       # List of timestamps for the graph
# }
# lock = threading.Lock()

# def calculate_green_time(vehicle_count):
#     """
#     Calculates green signal time based on the number of vehicles.
#     This is a simple heuristic; you can make this more complex.
#     """
#     base_time = 10  # Minimum green time
#     time_per_vehicle = 2  # Seconds per vehicle
#     max_time = 60   # Maximum green time

#     calculated_time = base_time + (vehicle_count * time_per_vehicle)
#     return min(calculated_time, max_time)

# def generate_frames():
#     """
#     Generator function to process video frames and perform vehicle counting.
#     Using traffic_video.mp4 as the video source.
#     """
#     cap = cv2.VideoCapture('traffic_video.mp4')  # Using your video file

#     # Initialize a dictionary to track counts for the current frame
#     current_counts = defaultdict(int)

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             # Loop the video when it ends
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         # Run YOLOv8 inference on the frame
#         results = model.track(frame, persist=True, verbose=False, classes=VEHICLE_CLASSES)

#         if results[0].boxes.id is not None:
#             # Get detected boxes, class IDs, and track IDs
#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
#             track_ids = results[0].boxes.id.cpu().numpy().astype(int)

#             # Reset counts for the current frame
#             current_counts = defaultdict(int)

#             # Count vehicles by type
#             for box, class_id, track_id in zip(boxes, class_ids, track_ids):
#                 label = class_names[class_id]
#                 if label in ['car', 'bus', 'truck', 'motorcycle']:
#                     current_counts[label] += 1

#                 # Optional: Draw bounding boxes and labels (comment out for max performance)
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{label}-{track_id}", (x1, y1-10),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Calculate total vehicles and green time
#         total_vehicles = sum(current_counts.values())
#         green_time = calculate_green_time(total_vehicles)

#         # Update global traffic data with a lock for thread safety
#         with lock:
#             traffic_data['current_counts'] = current_counts
#             traffic_data['total_count'] = total_vehicles
#             # Append to history every 5 seconds (or adjust as needed)
#             if len(traffic_data['time_labels']) == 0 or time.time() - traffic_data['time_labels'][-1] > 5:
#                 traffic_data['density_history'].append(total_vehicles)
#                 traffic_data['timing_history'].append(green_time)
#                 traffic_data['time_labels'].append(time.strftime("%H:%M:%S"))
#                 # Keep history limited to last 20 entries for the graph
#                 if len(traffic_data['density_history']) > 20:
#                     traffic_data['density_history'].pop(0)
#                     traffic_data['timing_history'].pop(0)
#                     traffic_data['time_labels'].pop(0)

#         # Overlay data on the video frame
#         cv2.putText(frame, f"Cars: {current_counts['car']} | Buses: {current_counts['bus']}", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, f"Trucks: {current_counts['truck']} | Bikes: {current_counts['motorcycle']}", (10, 60),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, f"Total: {total_vehicles} | Green Time: {green_time}s", (10, 90),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#         # Encode the frame to JPEG for streaming
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()

# @app.route('/')
# def index():
#     """Route to serve the main dashboard page."""
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     """Route to stream the processed video."""
#     return Response(generate_frames(),
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/traffic_data')
# def get_traffic_data():
#     """API endpoint to get current traffic data for the dashboard."""
#     with lock:
#         # Return a copy of the data to avoid thread issues
#         data = {
#             'counts': dict(traffic_data['current_counts']),
#             'total': traffic_data['total_count'],
#             'green_time': calculate_green_time(traffic_data['total_count']),
#             'density_history': traffic_data['density_history'],
#             'timing_history': traffic_data['timing_history'],
#             'time_labels': traffic_data['time_labels']
#         }
#     return jsonify(data)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=3001, threaded=True)






# four lanes -> incorrect
# from flask import Flask, render_template, Response, jsonify
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import defaultdict
# import time
# import threading

# app = Flask(__name__)

# # Initialize YOLOv8 model
# model = YOLO('yolov8n.pt')

# # Define vehicle classes of interest (COCO Dataset Class IDs)
# VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
# class_names = model.names

# # Traffic light states
# TRAFFIC_LIGHT_STATES = {
#     'RED': 0,
#     'YELLOW': 1, 
#     'GREEN': 2
# }

# # Global data structure for all 4 lanes
# lanes_data = {}
# for lane_id in range(1, 5):
#     lanes_data[lane_id] = {
#         'current_counts': {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0},
#         'total_count': 0,
#         'density_history': [],
#         'timing_history': [],
#         'time_labels': [],
#         'current_light': 'RED',
#         'green_time_remaining': 0,
#         'video_source': f'videos/lane{lane_id}.mp4'
#     }

# lock = threading.Lock()
# current_green_lane = 1
# light_change_time = time.time()
# YELLOW_LIGHT_DURATION = 3  # seconds
# MIN_GREEN_TIME = 10        # seconds
# MAX_GREEN_TIME = 60        # seconds

# def calculate_green_time(vehicle_count):
#     """Calculate green time based on vehicle count with min/max limits"""
#     time_per_vehicle = 2  # seconds per vehicle
#     calculated_time = MIN_GREEN_TIME + (vehicle_count * time_per_vehicle)
#     return min(calculated_time, MAX_GREEN_TIME)

# def manage_traffic_lights():
#     """Manage the traffic light cycle for all lanes"""
#     global current_green_lane, light_change_time, lanes_data
    
#     while True:
#         with lock:
#             current_time = time.time()
#             elapsed = current_time - light_change_time
            
#             # Check if current green light time has expired
#             current_green_time = lanes_data[current_green_lane]['green_time_remaining']
            
#             if elapsed >= current_green_time:
#                 # Switch to next lane
#                 next_lane = current_green_lane % 4 + 1
                
#                 # Set yellow light for current lane
#                 lanes_data[current_green_lane]['current_light'] = 'YELLOW'
#                 lanes_data[current_green_lane]['green_time_remaining'] = YELLOW_LIGHT_DURATION
                
#                 # After yellow duration, switch to green for next lane
#                 time.sleep(YELLOW_LIGHT_DURATION)
                
#                 lanes_data[current_green_lane]['current_light'] = 'RED'
#                 current_green_lane = next_lane
#                 lanes_data[current_green_lane]['current_light'] = 'GREEN'
                
#                 # Calculate new green time based on current traffic
#                 vehicle_count = lanes_data[current_green_lane]['total_count']
#                 new_green_time = calculate_green_time(vehicle_count)
#                 lanes_data[current_green_lane]['green_time_remaining'] = new_green_time
                
#                 light_change_time = time.time()
                
#                 print(f"Switched to Lane {current_green_lane} with {new_green_time}s green time")
        
#         time.sleep(0.1)  # Small delay to prevent CPU overload

# def process_lane_video(lane_id):
#     """Process video for a specific lane and count vehicles"""
#     video_source = lanes_data[lane_id]['video_source']
#     cap = cv2.VideoCapture(video_source)
    
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         # Run YOLOv8 inference
#         results = model.track(frame, persist=True, verbose=False, classes=VEHICLE_CLASSES)
#         current_counts = defaultdict(int)

#         if results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
#             for box, class_id in zip(boxes, class_ids):
#                 label = class_names[class_id]
#                 if label in ['car', 'bus', 'truck', 'motorcycle']:
#                     current_counts[label] += 1

#         # Update lane data
#         total_vehicles = sum(current_counts.values())
        
#         with lock:
#             lanes_data[lane_id]['current_counts'] = current_counts
#             lanes_data[lane_id]['total_count'] = total_vehicles
            
#             # Update history every 5 seconds
#             if len(lanes_data[lane_id]['time_labels']) == 0 or time.time() - lanes_data[lane_id]['time_labels'][-1] > 5:
#                 lanes_data[lane_id]['density_history'].append(total_vehicles)
#                 lanes_data[lane_id]['timing_history'].append(lanes_data[lane_id]['green_time_remaining'])
#                 lanes_data[lane_id]['time_labels'].append(time.strftime("%H:%M:%S"))
                
#                 # Keep limited history
#                 if len(lanes_data[lane_id]['density_history']) > 20:
#                     lanes_data[lane_id]['density_history'].pop(0)
#                     lanes_data[lane_id]['timing_history'].pop(0)
#                     lanes_data[lane_id]['time_labels'].pop(0)

#         # Add traffic light visualization to frame
#         light_color = lanes_data[lane_id]['current_light']
#         light_colors = {'RED': (0, 0, 255), 'YELLOW': (0, 255, 255), 'GREEN': (0, 255, 0)}
        
#         # Draw traffic light
#         cv2.rectangle(frame, (10, 10), (60, 160), (50, 50, 50), -1)
#         cv2.circle(frame, (35, 35), 15, light_colors['RED'], -1 if light_color == 'RED' else -1)
#         cv2.circle(frame, (35, 80), 15, light_colors['YELLOW'], -1 if light_color == 'YELLOW' else -1)
#         cv2.circle(frame, (35, 125), 15, light_colors['GREEN'], -1 if light_color == 'GREEN' else -1)
        
#         # Add info text
#         cv2.putText(frame, f"Lane {lane_id} - {light_color}", (70, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_colors[light_color], 2)
#         cv2.putText(frame, f"Vehicles: {total_vehicles}", (70, 60), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(frame, f"Time: {lanes_data[lane_id]['green_time_remaining']}s", (70, 90), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#         # Encode frame for streaming
#         ret, buffer = cv2.imencode('.jpg', frame)
#         yield buffer.tobytes()

#     cap.release()

# def generate_frames(lane_id):
#     """Generator function for video streaming"""
#     return process_lane_video(lane_id)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed/<int:lane_id>')
# def video_feed(lane_id):
#     return Response(generate_frames(lane_id),
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/traffic_data')
# def get_traffic_data():
#     with lock:
#         data = {
#             'lanes': {},
#             'current_green_lane': current_green_lane,
#             'system_time': time.strftime("%H:%M:%S")
#         }
        
#         for lane_id in range(1, 5):
#             data['lanes'][lane_id] = {
#                 'counts': dict(lanes_data[lane_id]['current_counts']),
#                 'total': lanes_data[lane_id]['total_count'],
#                 'light': lanes_data[lane_id]['current_light'],
#                 'green_time': lanes_data[lane_id]['green_time_remaining'],
#                 'density_history': lanes_data[lane_id]['density_history'],
#                 'timing_history': lanes_data[lane_id]['timing_history'],
#                 'time_labels': lanes_data[lane_id]['time_labels']
#             }
    
#     return jsonify(data)

# # Start traffic light management thread
# traffic_light_thread = threading.Thread(target=manage_traffic_lights, daemon=True)
# traffic_light_thread.start()

# # Initialize first lane as green
# with lock:
#     lanes_data[1]['current_light'] = 'GREEN'
#     lanes_data[1]['green_time_remaining'] = MIN_GREEN_TIME
#     light_change_time = time.time()

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=3001, threaded=True)






from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TrafficSystem')
handler = RotatingFileHandler('traffic_system.log', maxBytes=1000000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize YOLOv8 model with error handling
try:
    model = YOLO('yolov8n.pt')
    logger.info("YOLOv8 model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLOv8 model: {e}")
    # Fallback to a basic detection method if model fails to load
    model = None

# Define vehicle classes of interest (COCO Dataset Class IDs)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
class_names = model.names if model else {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Traffic light states
TRAFFIC_LIGHT_STATES = {
    'RED': 0,
    'YELLOW': 1, 
    'GREEN': 2
}

# Global data structure for all 4 lanes
lanes_data = {}
for lane_id in range(1, 5):
    lanes_data[lane_id] = {
        'current_counts': {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0},
        'total_count': 0,
        'density_history': [],
        'timing_history': [],
        'time_labels': [],
        'current_light': 'RED',
        'green_time_remaining': 0,
        'video_source': f'videos/lane{lane_id}.mp4',
        'last_update': time.time(),
        'status': 'active'
    }

lock = threading.Lock()
current_green_lane = 1
light_change_time = time.time()
YELLOW_LIGHT_DURATION = 3  # seconds
MIN_GREEN_TIME = 10        # seconds
MAX_GREEN_TIME = 60        # seconds
emergency_mode = False
system_start_time = time.time()

def calculate_green_time(vehicle_count):
    """Calculate green time based on vehicle count with min/max limits"""
    time_per_vehicle = 2  # seconds per vehicle
    calculated_time = MIN_GREEN_TIME + (vehicle_count * time_per_vehicle)
    return min(calculated_time, MAX_GREEN_TIME)

def manage_traffic_lights():
    """Manage the traffic light cycle for all lanes"""
    global current_green_lane, light_change_time, lanes_data, emergency_mode
    
    while True:
        with lock:
            current_time = time.time()
            elapsed = current_time - light_change_time
            
            # Emergency mode - all lights red except for emergency vehicle lane
            if emergency_mode:
                for lane_id in range(1, 5):
                    lanes_data[lane_id]['current_light'] = 'RED'
                # In a real system, we'd detect which lane has emergency vehicles
                # For demo, we'll assume lane 1 has emergency vehicles
                lanes_data[1]['current_light'] = 'GREEN'
                lanes_data[1]['green_time_remaining'] = 999  # Indefinite green
                time.sleep(1)
                continue
            
            # Check if current green light time has expired
            current_green_time = lanes_data[current_green_lane]['green_time_remaining']
            
            if elapsed >= current_green_time:
                # Switch to next lane
                next_lane = current_green_lane % 4 + 1
                
                # Set yellow light for current lane
                lanes_data[current_green_lane]['current_light'] = 'YELLOW'
                lanes_data[current_green_lane]['green_time_remaining'] = YELLOW_LIGHT_DURATION
                
                # After yellow duration, switch to green for next lane
                time.sleep(YELLOW_LIGHT_DURATION)
                
                lanes_data[current_green_lane]['current_light'] = 'RED'
                current_green_lane = next_lane
                lanes_data[current_green_lane]['current_light'] = 'GREEN'
                
                # Calculate new green time based on current traffic
                vehicle_count = lanes_data[current_green_lane]['total_count']
                new_green_time = calculate_green_time(vehicle_count)
                lanes_data[current_green_lane]['green_time_remaining'] = new_green_time
                
                light_change_time = time.time()
                
                logger.info(f"Switched to Lane {current_green_lane} with {new_green_time}s green time")
        
        time.sleep(0.1)  # Small delay to prevent CPU overload

def process_lane_video(lane_id):
    """Process video for a specific lane and count vehicles"""
    video_source = lanes_data[lane_id]['video_source']
    
    # Check if video file exists
    if not os.path.exists(video_source):
        logger.error(f"Video file not found: {video_source}")
        lanes_data[lane_id]['status'] = 'error'
        # Generate a placeholder frame with error message
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Video not found: {video_source}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield buffer.tobytes()
            time.sleep(0.1)
    
    cap = cv2.VideoCapture(video_source)
    
    # Check if video opened successfully
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_source}")
        lanes_data[lane_id]['status'] = 'error'
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Failed to open video", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield buffer.tobytes()
            time.sleep(0.1)
    
    # Set a lower resolution for better performance if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    process_every_n_frames = 2  # Process every 2nd frame to improve performance
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # Reset video to start if we reach the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        
        # Skip processing for some frames to improve performance
        if frame_count % process_every_n_frames != 0:
            # Still yield the frame but without processing
            ret, buffer = cv2.imencode('.jpg', frame)
            yield buffer.tobytes()
            continue

        # Run YOLOv8 inference if model is available
        current_counts = defaultdict(int)
        if model:
            try:
                results = model.track(frame, persist=True, verbose=False, classes=VEHICLE_CLASSES)
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for box, class_id in zip(boxes, class_ids):
                        label = class_names[class_id]
                        if label in ['car', 'bus', 'truck', 'motorcycle']:
                            current_counts[label] += 1
            except Exception as e:
                logger.error(f"YOLO inference error in lane {lane_id}: {e}")
        else:
            # Fallback detection if model is not available
            # This is a simple placeholder - in a real scenario you'd implement proper fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count contours of a certain size as vehicles
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:
                    current_counts['car'] += 1

        # Update lane data
        total_vehicles = sum(current_counts.values())
        
        with lock:
            lanes_data[lane_id]['current_counts'] = current_counts
            lanes_data[lane_id]['total_count'] = total_vehicles
            lanes_data[lane_id]['last_update'] = time.time()
            
            # Update history every 5 seconds
            current_time = time.time()
            if (len(lanes_data[lane_id]['time_labels']) == 0 or 
                current_time - lanes_data[lane_id]['time_labels'][-1] > 5):
                lanes_data[lane_id]['density_history'].append(total_vehicles)
                lanes_data[lane_id]['timing_history'].append(lanes_data[lane_id]['green_time_remaining'])
                lanes_data[lane_id]['time_labels'].append(datetime.fromtimestamp(current_time).strftime("%H:%M:%S"))
                
                # Keep limited history
                if len(lanes_data[lane_id]['density_history']) > 20:
                    lanes_data[lane_id]['density_history'].pop(0)
                    lanes_data[lane_id]['timing_history'].pop(0)
                    lanes_data[lane_id]['time_labels'].pop(0)

        # Add traffic light visualization to frame
        light_color = lanes_data[lane_id]['current_light']
        light_colors = {'RED': (0, 0, 255), 'YELLOW': (0, 255, 255), 'GREEN': (0, 255, 0)}
        
        # Draw traffic light
        cv2.rectangle(frame, (10, 10), (60, 160), (50, 50, 50), -1)
        cv2.circle(frame, (35, 35), 15, light_colors['RED'], -1 if light_color == 'RED' else -1)
        cv2.circle(frame, (35, 80), 15, light_colors['YELLOW'], -1 if light_color == 'YELLOW' else -1)
        cv2.circle(frame, (35, 125), 15, light_colors['GREEN'], -1 if light_color == 'GREEN' else -1)
        
        # Add info text
        cv2.putText(frame, f"Lane {lane_id} - {light_color}", (70, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_colors[light_color], 2)
        cv2.putText(frame, f"Vehicles: {total_vehicles}", (70, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {lanes_data[lane_id]['green_time_remaining']}s", (70, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        yield buffer.tobytes()

    cap.release()

def generate_frames(lane_id):
    """Generator function for video streaming"""
    return process_lane_video(lane_id)

@app.route('/')
def index():
    return render_template('index.html', lanes=[1, 2, 3, 4])

@app.route('/video_feed/<int:lane_id>')
def video_feed(lane_id):
    return Response(generate_frames(lane_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/traffic_data')
def get_traffic_data():
    with lock:
        data = {
            'lanes': {},
            'current_green_lane': current_green_lane,
            'system_time': datetime.now().strftime("%H:%M:%S"),
            'system_uptime': int(time.time() - system_start_time),
            'emergency_mode': emergency_mode
        }
        
        for lane_id in range(1, 5):
            data['lanes'][lane_id] = {
                'counts': dict(lanes_data[lane_id]['current_counts']),
                'total': lanes_data[lane_id]['total_count'],
                'light': lanes_data[lane_id]['current_light'],
                'green_time': lanes_data[lane_id]['green_time_remaining'],
                'density_history': lanes_data[lane_id]['density_history'],
                'timing_history': lanes_data[lane_id]['timing_history'],
                'time_labels': lanes_data[lane_id]['time_labels'],
                'status': lanes_data[lane_id]['status'],
                'last_update': lanes_data[lane_id]['last_update']
            }
    
    return jsonify(data)

@app.route('/manual_switch', methods=['POST'])
def manual_switch():
    global current_green_lane, light_change_time
    with lock:
        # Switch to next lane
        next_lane = current_green_lane % 4 + 1
        
        # Set current lane to red
        lanes_data[current_green_lane]['current_light'] = 'RED'
        
        # Set next lane to green
        current_green_lane = next_lane
        lanes_data[current_green_lane]['current_light'] = 'GREEN'
        
        # Calculate new green time based on current traffic
        vehicle_count = lanes_data[current_green_lane]['total_count']
        new_green_time = calculate_green_time(vehicle_count)
        lanes_data[current_green_lane]['green_time_remaining'] = new_green_time
        
        light_change_time = time.time()
        
        logger.info(f"Manual switch to Lane {current_green_lane} with {new_green_time}s green time")
    
    return jsonify({'message': f'Switched to lane {current_green_lane}', 'success': True})

@app.route('/emergency_mode', methods=['POST'])
def toggle_emergency_mode():
    global emergency_mode
    emergency_mode = not emergency_mode
    
    if emergency_mode:
        logger.info("Emergency mode activated")
        return jsonify({'message': 'Emergency mode activated', 'emergency_mode': True})
    else:
        logger.info("Emergency mode deactivated")
        # Reset light change time to resume normal operation
        global light_change_time
        light_change_time = time.time()
        return jsonify({'message': 'Emergency mode deactivated', 'emergency_mode': False})

@app.route('/reset_system', methods=['POST'])
def reset_system():
    global current_green_lane, light_change_time, emergency_mode
    
    with lock:
        emergency_mode = False
        current_green_lane = 1
        light_change_time = time.time()
        
        for lane_id in range(1, 5):
            lanes_data[lane_id]['current_light'] = 'RED'
            lanes_data[lane_id]['green_time_remaining'] = 0
        
        # Set first lane as green
        lanes_data[1]['current_light'] = 'GREEN'
        lanes_data[1]['green_time_remaining'] = MIN_GREEN_TIME
        
        logger.info("System reset")
    
    return jsonify({'message': 'System reset successfully', 'success': True})

@app.route('/set_green_time', methods=['POST'])
def set_green_time():
    lane_id = request.json.get('lane_id')
    green_time = request.json.get('green_time')
    
    if not lane_id or not green_time:
        return jsonify({'message': 'Missing parameters', 'success': False}), 400
    
    if lane_id not in [1, 2, 3, 4]:
        return jsonify({'message': 'Invalid lane ID', 'success': False}), 400
    
    try:
        green_time = int(green_time)
        if green_time < MIN_GREEN_TIME or green_time > MAX_GREEN_TIME:
            return jsonify({'message': f'Green time must be between {MIN_GREEN_TIME} and {MAX_GREEN_TIME}', 'success': False}), 400
    except ValueError:
        return jsonify({'message': 'Green time must be a number', 'success': False}), 400
    
    with lock:
        lanes_data[lane_id]['green_time_remaining'] = green_time
        logger.info(f"Set green time for lane {lane_id} to {green_time}s")
    
    return jsonify({'message': f'Green time for lane {lane_id} set to {green_time}s', 'success': True})

# Start traffic light management thread
traffic_light_thread = threading.Thread(target=manage_traffic_lights, daemon=True)
traffic_light_thread.start()

# Initialize first lane as green
with lock:
    lanes_data[1]['current_light'] = 'GREEN'
    lanes_data[1]['green_time_remaining'] = MIN_GREEN_TIME
    light_change_time = time.time()

if __name__ == '__main__':
    # Check if videos directory exists
    if not os.path.exists('videos'):
        os.makedirs('videos')
        logger.warning("Videos directory created. Please add your video files.")
    
    app.run(debug=True, host='0.0.0.0', port=3001, threaded=True)
