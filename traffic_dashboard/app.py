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






#  four lanes -> correct
# from flask import Flask, render_template, Response, jsonify
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import defaultdict
# import time
# import threading
# import copy

# app = Flask(__name__)

# # --- Configuration ---
# MODEL = YOLO('yolov8n.pt')
# VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
# CLASS_NAMES = MODEL.names
# LANE_IDS = [1, 2, 3, 4]
# VIDEO_SOURCES = {
#     1: 'videos/lane1.mp4',
#     2: 'videos/lane2.mp4',
#     3: 'videos/lane3.mp4',
#     4: 'videos/lane4.mp4',
# }

# # --- Global Data Structures & Threading Lock ---
# traffic_data = {}
# lock = threading.Lock()        # protects traffic_data
# yolo_lock = threading.Lock()   # protects first call to YOLO

# for lane_id in LANE_IDS:
#     traffic_data[lane_id] = {
#         'current_counts': defaultdict(int),
#         'total_count': 0,
#         'latest_frame': None,
#         'green_time': 10,
#         'density_history': [],
#         'timing_history': [],
#         'time_labels': []
#     }

# # --- Core Logic ---
# def calculate_green_time(vehicle_count):
#     base_time = 10
#     time_per_vehicle = 2
#     max_time = 60
#     return min(base_time + vehicle_count * time_per_vehicle, max_time)

# def process_video_for_lane(lane_id):
#     video_path = VIDEO_SOURCES[lane_id]
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"[ERROR] Could not open video for lane {lane_id}: {video_path}")
#         return

#     print(f"[INFO] Started processing for lane {lane_id}...")

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         # --- YOLO Inference (thread-safe) ---
#         with yolo_lock:
#             results = MODEL.track(frame, persist=True, verbose=False,
#                                   classes=VEHICLE_CLASSES)

#         current_counts = defaultdict(int)
#         if results and results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

#             for box, cid in zip(boxes, class_ids):
#                 label = CLASS_NAMES[cid]
#                 current_counts[label] += 1
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 8),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         total = sum(current_counts.values())
#         gtime = calculate_green_time(total)

#         info = f"Total: {total} | Green Time: {gtime}s"
#         cv2.putText(frame, f"Lane {lane_id}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         cv2.putText(frame, info, (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         ret, buf = cv2.imencode(".jpg", frame)
#         frame_bytes = buf.tobytes()

#         # --- Update global data safely ---
#         with lock:
#             lane = traffic_data[lane_id]
#             lane["current_counts"] = current_counts
#             lane["total_count"] = total
#             lane["green_time"] = gtime
#             lane["latest_frame"] = frame_bytes

#             now = time.time()
#             last = lane["time_labels"][-1] if lane["time_labels"] else 0
#             if now - last > 5:
#                 lane["density_history"].append(total)
#                 lane["timing_history"].append(gtime)
#                 lane["time_labels"].append(now)
#                 if len(lane["density_history"]) > 20:
#                     lane["density_history"].pop(0)
#                     lane["timing_history"].pop(0)
#                     lane["time_labels"].pop(0)

# # --- Flask Routes ---
# @app.route("/")
# def index():
#     return render_template("index.html", lanes=LANE_IDS)

# def generate_video_stream(lane_id):
#     while True:
#         time.sleep(0.03)
#         with lock:
#             frame = traffic_data[lane_id]["latest_frame"]
#         if frame is None:
#             continue
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
#                frame + b"\r\n")

# @app.route("/video_feed/<int:lane_id>")
# def video_feed(lane_id):
#     if lane_id not in LANE_IDS:
#         return "Invalid Lane ID", 404
#     return Response(generate_video_stream(lane_id),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/traffic_data")
# def get_traffic_data():
#     with lock:
#         payload = {}
#         for lid, data in traffic_data.items():
#             d = {k: v for k, v in data.items() if k != "latest_frame"}
#             d["time_labels"] = [time.strftime("%H:%M:%S", time.localtime(ts))
#                                 for ts in d["time_labels"]]
#             payload[lid] = d
#     return jsonify(payload)

# if __name__ == "__main__":
#     for lid in LANE_IDS:
#         t = threading.Thread(target=process_video_for_lane,
#                              args=(lid,), daemon=True)
#         t.start()
#     app.run(debug=False, host="0.0.0.0", port=3001, threaded=True)









# emergency vehicle prioritization
# from flask import Flask, render_template, Response, jsonify
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import defaultdict
# import time
# import threading
# import copy

# app = Flask(__name__)

# # --- Configuration ---
# MODEL = YOLO('yolov8n.pt')
# VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
# EMERGENCY_CLASSES = [0]  # person (can be extended to specific emergency vehicle classes)
# # Note: You might want to train a custom model or use different classes for emergency vehicles
# # Common emergency vehicle classes: ambulance, fire truck, police car
# CLASS_NAMES = MODEL.names
# LANE_IDS = [1, 2, 3, 4]
# VIDEO_SOURCES = {
#     1: 'videos/lane1.mp4',
#     2: 'videos/lane2.mp4',
#     3: 'videos/lane3.mp4',
#     4: 'videos/lane4.mp4',
# }

# # --- Emergency Vehicle Detection Configuration ---
# EMERGENCY_PRIORITY_THRESHOLD = 1  # Minimum emergency vehicles to trigger priority
# EMERGENCY_GREEN_TIME = 20  # Extended green time for emergency lanes
# PRIORITY_COOLDOWN = 30  # Seconds before another emergency can trigger priority

# # --- Global Data Structures & Threading Lock ---
# traffic_data = {}
# emergency_status = {
#     'active_emergency': False,
#     'emergency_lane': None,
#     'priority_end_time': 0,
#     'cooldown_end_time': 0
# }
# lock = threading.Lock()        # protects traffic_data and emergency_status
# yolo_lock = threading.Lock()   # protects first call to YOLO

# for lane_id in LANE_IDS:
#     traffic_data[lane_id] = {
#         'current_counts': defaultdict(int),
#         'emergency_count': 0,
#         'total_count': 0,
#         'latest_frame': None,
#         'green_time': 10,
#         'density_history': [],
#         'timing_history': [],
#         'time_labels': [],
#         'has_emergency': False
#     }

# # --- Core Logic ---
# def calculate_green_time(vehicle_count, has_emergency=False):
#     if has_emergency:
#         return EMERGENCY_GREEN_TIME
    
#     base_time = 10
#     time_per_vehicle = 2
#     max_time = 60
#     return min(base_time + vehicle_count * time_per_vehicle, max_time)

# def check_emergency_priority():
#     """Check if emergency priority should be activated or deactivated"""
#     global emergency_status
    
#     current_time = time.time()
    
#     # Check if we're in cooldown period
#     if current_time < emergency_status['cooldown_end_time']:
#         return
    
#     # Check if emergency priority is already active
#     if emergency_status['active_emergency']:
#         if current_time >= emergency_status['priority_end_time']:
#             # Emergency priority period ended
#             with lock:
#                 emergency_status['active_emergency'] = False
#                 emergency_status['emergency_lane'] = None
#                 emergency_status['cooldown_end_time'] = current_time + PRIORITY_COOLDOWN
#             print(f"[EMERGENCY] Priority mode ended. Cooldown until {time.strftime('%H:%M:%S', time.localtime(emergency_status['cooldown_end_time']))}")
#         return
    
#     # Check for new emergency vehicles
#     emergency_lane = None
#     max_emergency_count = 0
    
#     with lock:
#         for lane_id in LANE_IDS:
#             if traffic_data[lane_id]['emergency_count'] >= EMERGENCY_PRIORITY_THRESHOLD:
#                 if traffic_data[lane_id]['emergency_count'] > max_emergency_count:
#                     max_emergency_count = traffic_data[lane_id]['emergency_count']
#                     emergency_lane = lane_id
    
#     if emergency_lane is not None:
#         # Activate emergency priority
#         with lock:
#             emergency_status['active_emergency'] = True
#             emergency_status['emergency_lane'] = emergency_lane
#             emergency_status['priority_end_time'] = current_time + EMERGENCY_GREEN_TIME
#         print(f"[EMERGENCY] Priority activated for Lane {emergency_lane}! Green time: {EMERGENCY_GREEN_TIME}s")

# def process_video_for_lane(lane_id):
#     video_path = VIDEO_SOURCES[lane_id]
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"[ERROR] Could not open video for lane {lane_id}: {video_path}")
#         return

#     print(f"[INFO] Started processing for lane {lane_id}...")

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         # --- YOLO Inference (thread-safe) ---
#         with yolo_lock:
#             results = MODEL.track(frame, persist=True, verbose=False,
#                                   classes=VEHICLE_CLASSES + EMERGENCY_CLASSES)

#         current_counts = defaultdict(int)
#         emergency_count = 0

#         if results and results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
#             confidences = results[0].boxes.conf.cpu().numpy()

#             for box, cid, conf in zip(boxes, class_ids, confidences):
#                 label = CLASS_NAMES[cid]
#                 current_counts[label] += 1
                
#                 # Check for emergency vehicles (customize this logic based on your needs)
#                 if cid in EMERGENCY_CLASSES and conf > 0.6:
#                     emergency_count += 1
#                     # Draw emergency vehicle with red border
#                     color = (0, 0, 255)  # Red for emergency
#                     thickness = 3
#                 else:
#                     color = (0, 255, 0)  # Green for regular vehicles
#                     thickness = 2
                
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
#                 cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         total = sum(current_counts.values())
        
#         # Check emergency status
#         with lock:
#             has_emergency = emergency_status['active_emergency'] and emergency_status['emergency_lane'] == lane_id
        
#         gtime = calculate_green_time(total, has_emergency)

#         # Draw status information
#         status_color = (0, 0, 255) if has_emergency else (255, 255, 255)
#         status_text = "EMERGENCY PRIORITY" if has_emergency else "NORMAL"
        
#         cv2.putText(frame, f"Lane {lane_id} - {status_text}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
#         info = f"Total: {total} | Emergency: {emergency_count} | Green: {gtime}s"
#         cv2.putText(frame, info, (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Draw emergency indicator
#         if emergency_count > 0:
#             cv2.putText(frame, "🚑 EMERGENCY VEHICLE DETECTED", (10, 110),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         ret, buf = cv2.imencode(".jpg", frame)
#         frame_bytes = buf.tobytes()

#         # --- Update global data safely ---
#         with lock:
#             lane = traffic_data[lane_id]
#             lane["current_counts"] = current_counts
#             lane["emergency_count"] = emergency_count
#             lane["total_count"] = total
#             lane["green_time"] = gtime
#             lane["latest_frame"] = frame_bytes
#             lane["has_emergency"] = has_emergency

#             now = time.time()
#             last = lane["time_labels"][-1] if lane["time_labels"] else 0
#             if now - last > 5:
#                 lane["density_history"].append(total)
#                 lane["timing_history"].append(gtime)
#                 lane["time_labels"].append(now)
#                 if len(lane["density_history"]) > 20:
#                     lane["density_history"].pop(0)
#                     lane["timing_history"].pop(0)
#                     lane["time_labels"].pop(0)

# # --- Emergency Monitoring Thread ---
# def emergency_monitor():
#     """Continuously monitor for emergency vehicles and manage priority"""
#     while True:
#         check_emergency_priority()
#         time.sleep(1)  # Check every second

# # --- Flask Routes ---
# @app.route("/")
# def index():
#     return render_template("index.html", lanes=LANE_IDS)

# def generate_video_stream(lane_id):
#     while True:
#         time.sleep(0.03)
#         with lock:
#             frame = traffic_data[lane_id]["latest_frame"]
#         if frame is None:
#             continue
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
#                frame + b"\r\n")

# @app.route("/video_feed/<int:lane_id>")
# def video_feed(lane_id):
#     if lane_id not in LANE_IDS:
#         return "Invalid Lane ID", 404
#     return Response(generate_video_stream(lane_id),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/traffic_data")
# def get_traffic_data():
#     with lock:
#         payload = {
#             'lanes': {},
#             'emergency_status': emergency_status.copy()
#         }
        
#         # Convert timestamps to readable format
#         emergency_status_readable = emergency_status.copy()
#         emergency_status_readable['priority_end_time'] = time.strftime("%H:%M:%S", time.localtime(emergency_status['priority_end_time'])) if emergency_status['priority_end_time'] > 0 else "N/A"
#         emergency_status_readable['cooldown_end_time'] = time.strftime("%H:%M:%S", time.localtime(emergency_status['cooldown_end_time'])) if emergency_status['cooldown_end_time'] > 0 else "N/A"
#         payload['emergency_status'] = emergency_status_readable
        
#         for lid, data in traffic_data.items():
#             d = {k: v for k, v in data.items() if k != "latest_frame"}
#             d["time_labels"] = [time.strftime("%H:%M:%S", time.localtime(ts))
#                                 for ts in d["time_labels"]]
#             payload['lanes'][lid] = d
    
#     return jsonify(payload)

# @app.route("/emergency_override/<int:lane_id>", methods=["POST"])
# def emergency_override(lane_id):
#     """Manual emergency override endpoint"""
#     if lane_id not in LANE_IDS:
#         return jsonify({"error": "Invalid lane ID"}), 400
    
#     with lock:
#         emergency_status['active_emergency'] = True
#         emergency_status['emergency_lane'] = lane_id
#         emergency_status['priority_end_time'] = time.time() + EMERGENCY_GREEN_TIME
#         emergency_status['cooldown_end_time'] = 0
    
#     print(f"[MANUAL OVERRIDE] Emergency priority activated for Lane {lane_id}")
#     return jsonify({
#         "status": "success",
#         "message": f"Emergency priority activated for Lane {lane_id}",
#         "green_time": EMERGENCY_GREEN_TIME
#     })

# @app.route("/clear_emergency", methods=["POST"])
# def clear_emergency():
#     """Clear emergency priority manually"""
#     with lock:
#         emergency_status['active_emergency'] = False
#         emergency_status['emergency_lane'] = None
#         emergency_status['cooldown_end_time'] = time.time() + PRIORITY_COOLDOWN
    
#     print("[MANUAL OVERRIDE] Emergency priority cleared")
#     return jsonify({
#         "status": "success",
#         "message": "Emergency priority cleared"
#     })

# if __name__ == "__main__":
#     # Start video processing threads
#     for lid in LANE_IDS:
#         t = threading.Thread(target=process_video_for_lane,
#                              args=(lid,), daemon=True)
#         t.start()
    
#     # Start emergency monitoring thread
#     emergency_thread = threading.Thread(target=emergency_monitor, daemon=True)
#     emergency_thread.start()
    
#     print("[SYSTEM] Traffic monitoring system started with emergency vehicle prioritization")
#     print("[SYSTEM] Emergency classes being monitored:", [CLASS_NAMES[c] for c in EMERGENCY_CLASSES])
    
#     app.run(debug=False, host="0.0.0.0", port=3001, threaded=True)














# emergency -> 2
# from flask import Flask, render_template, Response, jsonify
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import defaultdict, deque
# import time
# import threading
# import copy

# app = Flask(__name__)

# # --- Configuration ---
# MODEL = YOLO('yolov8n.pt')
# VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
# EMERGENCY_CLASSES = [0]  # person (can be extended to specific emergency vehicle classes)
# CLASS_NAMES = MODEL.names
# LANE_IDS = [1, 2, 3, 4]
# VIDEO_SOURCES = {
#     1: 'videos/lane1.mp4',
#     2: 'videos/lane2.mp4',
#     3: 'videos/lane3.mp4',
#     4: 'videos/lane4.mp4',
# }

# # --- Emergency Vehicle Detection Configuration ---
# EMERGENCY_PRIORITY_THRESHOLD = 1
# EMERGENCY_GREEN_TIME = 20
# PRIORITY_COOLDOWN = 30

# # --- Green Time Configuration ---
# GREEN_TIME_AVERAGING_WINDOW = 10  # Number of samples to average for green time
# MIN_GREEN_TIME = 10  # Minimum green time in seconds
# MAX_GREEN_TIME = 60  # Maximum green time in seconds
# BASE_GREEN_TIME = 10  # Base green time without vehicles
# TIME_PER_VEHICLE = 2  # Additional seconds per vehicle
# UPDATE_INTERVAL = 5   # Seconds between green time updates

# # --- Global Data Structures & Threading Lock ---
# traffic_data = {}
# emergency_status = {
#     'active_emergency': False,
#     'emergency_lane': None,
#     'priority_end_time': 0,
#     'cooldown_end_time': 0
# }
# lock = threading.Lock()
# yolo_lock = threading.Lock()

# for lane_id in LANE_IDS:
#     traffic_data[lane_id] = {
#         'current_counts': defaultdict(int),
#         'emergency_count': 0,
#         'total_count': 0,
#         'latest_frame': None,
#         'green_time': BASE_GREEN_TIME,  # Start with base time
#         'target_green_time': BASE_GREEN_TIME,  # Target time we're moving toward
#         'current_green_time': BASE_GREEN_TIME,  # Currently displayed time
#         'density_history': deque(maxlen=20),
#         'timing_history': deque(maxlen=20),
#         'time_labels': deque(maxlen=20),
#         'vehicle_count_history': deque(maxlen=GREEN_TIME_AVERAGING_WINDOW),
#         'has_emergency': False,
#         'last_update_time': 0
#     }

# # --- Core Logic ---
# def calculate_target_green_time(vehicle_count, has_emergency=False):
#     """Calculate target green time based on vehicle count"""
#     if has_emergency:
#         return EMERGENCY_GREEN_TIME
    
#     calculated_time = BASE_GREEN_TIME + (vehicle_count * TIME_PER_VEHICLE)
#     return max(MIN_GREEN_TIME, min(calculated_time, MAX_GREEN_TIME))

# def smooth_green_time_transition(lane_id, current_time):
#     """Smoothly transition between green time values"""
#     with lock:
#         lane_data = traffic_data[lane_id]
        
#         # Only update at specified intervals
#         if current_time - lane_data['last_update_time'] < UPDATE_INTERVAL:
#             return lane_data['current_green_time']
        
#         # Calculate new target based on average vehicle count
#         if lane_data['vehicle_count_history']:
#             avg_vehicles = sum(lane_data['vehicle_count_history']) / len(lane_data['vehicle_count_history'])
#             new_target = calculate_target_green_time(avg_vehicles, lane_data['has_emergency'])
#         else:
#             new_target = calculate_target_green_time(lane_data['total_count'], lane_data['has_emergency'])
        
#         # Smooth transition (move 25% toward target each update)
#         current_gt = lane_data['current_green_time']
#         smoothed_gt = current_gt + 0.25 * (new_target - current_gt)
        
#         # Round to nearest integer and clamp
#         smoothed_gt = round(smoothed_gt)
#         smoothed_gt = max(MIN_GREEN_TIME, min(smoothed_gt, MAX_GREEN_TIME))
        
#         lane_data['target_green_time'] = new_target
#         lane_data['current_green_time'] = smoothed_gt
#         lane_data['last_update_time'] = current_time
        
#         return smoothed_gt

# def check_emergency_priority():
#     """Check if emergency priority should be activated or deactivated"""
#     global emergency_status
    
#     current_time = time.time()
    
#     # Check if we're in cooldown period
#     if current_time < emergency_status['cooldown_end_time']:
#         return
    
#     # Check if emergency priority is already active
#     if emergency_status['active_emergency']:
#         if current_time >= emergency_status['priority_end_time']:
#             # Emergency priority period ended
#             with lock:
#                 emergency_status['active_emergency'] = False
#                 emergency_status['emergency_lane'] = None
#                 emergency_status['cooldown_end_time'] = current_time + PRIORITY_COOLDOWN
#             print(f"[EMERGENCY] Priority mode ended. Cooldown until {time.strftime('%H:%M:%S', time.localtime(emergency_status['cooldown_end_time']))}")
#         return
    
#     # Check for new emergency vehicles
#     emergency_lane = None
#     max_emergency_count = 0
    
#     with lock:
#         for lane_id in LANE_IDS:
#             if traffic_data[lane_id]['emergency_count'] >= EMERGENCY_PRIORITY_THRESHOLD:
#                 if traffic_data[lane_id]['emergency_count'] > max_emergency_count:
#                     max_emergency_count = traffic_data[lane_id]['emergency_count']
#                     emergency_lane = lane_id
    
#     if emergency_lane is not None:
#         # Activate emergency priority
#         with lock:
#             emergency_status['active_emergency'] = True
#             emergency_status['emergency_lane'] = emergency_lane
#             emergency_status['priority_end_time'] = current_time + EMERGENCY_GREEN_TIME
#         print(f"[EMERGENCY] Priority activated for Lane {emergency_lane}! Green time: {EMERGENCY_GREEN_TIME}s")

# def process_video_for_lane(lane_id):
#     video_path = VIDEO_SOURCES[lane_id]
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"[ERROR] Could not open video for lane {lane_id}: {video_path}")
#         return

#     print(f"[INFO] Started processing for lane {lane_id}...")

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         # --- YOLO Inference (thread-safe) ---
#         with yolo_lock:
#             results = MODEL.track(frame, persist=True, verbose=False,
#                                   classes=VEHICLE_CLASSES + EMERGENCY_CLASSES)

#         current_counts = defaultdict(int)
#         emergency_count = 0
#         current_time = time.time()

#         if results and results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
#             confidences = results[0].boxes.conf.cpu().numpy()

#             for box, cid, conf in zip(boxes, class_ids, confidences):
#                 label = CLASS_NAMES[cid]
#                 current_counts[label] += 1
                
#                 # Check for emergency vehicles
#                 if cid in EMERGENCY_CLASSES and conf > 0.6:
#                     emergency_count += 1
#                     color = (0, 0, 255)  # Red for emergency
#                     thickness = 3
#                 else:
#                     color = (0, 255, 0)  # Green for regular vehicles
#                     thickness = 2
                
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
#                 cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         total = sum(current_counts.values())
        
#         # Check emergency status and calculate smoothed green time
#         with lock:
#             has_emergency = emergency_status['active_emergency'] and emergency_status['emergency_lane'] == lane_id
#             traffic_data[lane_id]['has_emergency'] = has_emergency
            
#             # Add to vehicle count history for averaging
#             traffic_data[lane_id]['vehicle_count_history'].append(total)
        
#         # Get smoothed green time
#         smoothed_gtime = smooth_green_time_transition(lane_id, current_time)

#         # Draw status information
#         status_color = (0, 0, 255) if has_emergency else (255, 255, 255)
#         status_text = "EMERGENCY PRIORITY" if has_emergency else "NORMAL"
        
#         cv2.putText(frame, f"Lane {lane_id} - {status_text}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
#         info = f"Total: {total} | Emergency: {emergency_count} | Green: {smoothed_gtime}s"
#         cv2.putText(frame, info, (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Draw emergency indicator
#         if emergency_count > 0:
#             cv2.putText(frame, "🚑 EMERGENCY VEHICLE DETECTED", (10, 110),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         ret, buf = cv2.imencode(".jpg", frame)
#         frame_bytes = buf.tobytes()

#         # --- Update global data safely ---
#         with lock:
#             lane = traffic_data[lane_id]
#             lane["current_counts"] = current_counts
#             lane["emergency_count"] = emergency_count
#             lane["total_count"] = total
#             lane["green_time"] = smoothed_gtime  # Use smoothed value

#             now = time.time()
#             last = lane["time_labels"][-1] if lane["time_labels"] else 0
#             if now - last > 5:
#                 lane["density_history"].append(total)
#                 lane["timing_history"].append(smoothed_gtime)
#                 lane["time_labels"].append(now)
            
#             lane["latest_frame"] = frame_bytes

# # --- Emergency Monitoring Thread ---
# def emergency_monitor():
#     """Continuously monitor for emergency vehicles and manage priority"""
#     while True:
#         check_emergency_priority()
#         time.sleep(1)

# # --- Flask Routes ---
# @app.route("/")
# def index():
#     return render_template("index.html", lanes=LANE_IDS)

# def generate_video_stream(lane_id):
#     while True:
#         time.sleep(0.03)
#         with lock:
#             frame = traffic_data[lane_id]["latest_frame"]
#         if frame is None:
#             continue
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
#                frame + b"\r\n")

# @app.route("/video_feed/<int:lane_id>")
# def video_feed(lane_id):
#     if lane_id not in LANE_IDS:
#         return "Invalid Lane ID", 404
#     return Response(generate_video_stream(lane_id),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/traffic_data")
# def get_traffic_data():
#     with lock:
#         payload = {
#             'lanes': {},
#             'emergency_status': emergency_status.copy()
#         }
        
#         # Convert timestamps to readable format
#         emergency_status_readable = emergency_status.copy()
#         emergency_status_readable['priority_end_time'] = time.strftime("%H:%M:%S", time.localtime(emergency_status['priority_end_time'])) if emergency_status['priority_end_time'] > 0 else "N/A"
#         emergency_status_readable['cooldown_end_time'] = time.strftime("%H:%M:%S", time.localtime(emergency_status['cooldown_end_time'])) if emergency_status['cooldown_end_time'] > 0 else "N/A"
#         payload['emergency_status'] = emergency_status_readable
        
#         for lid, data in traffic_data.items():
#             d = {k: v for k, v in data.items() if k != "latest_frame"}
#             # Convert deques to lists for JSON serialization
#             d["density_history"] = list(d["density_history"])
#             d["timing_history"] = list(d["timing_history"])
#             d["time_labels"] = [time.strftime("%H:%M:%S", time.localtime(ts)) for ts in d["time_labels"]]
#             d["vehicle_count_history"] = list(d["vehicle_count_history"])
#             payload['lanes'][lid] = d
    
#     return jsonify(payload)

# @app.route("/emergency_override/<int:lane_id>", methods=["POST"])
# def emergency_override(lane_id):
#     """Manual emergency override endpoint"""
#     if lane_id not in LANE_IDS:
#         return jsonify({"error": "Invalid lane ID"}), 400
    
#     with lock:
#         emergency_status['active_emergency'] = True
#         emergency_status['emergency_lane'] = lane_id
#         emergency_status['priority_end_time'] = time.time() + EMERGENCY_GREEN_TIME
#         emergency_status['cooldown_end_time'] = 0
        
#         # Immediately update the green time for the emergency lane
#         traffic_data[lane_id]['current_green_time'] = EMERGENCY_GREEN_TIME
#         traffic_data[lane_id]['target_green_time'] = EMERGENCY_GREEN_TIME
    
#     print(f"[MANUAL OVERRIDE] Emergency priority activated for Lane {lane_id}")
#     return jsonify({
#         "status": "success",
#         "message": f"Emergency priority activated for Lane {lane_id}",
#         "green_time": EMERGENCY_GREEN_TIME
#     })

# @app.route("/clear_emergency", methods=["POST"])
# def clear_emergency():
#     """Clear emergency priority manually"""
#     with lock:
#         emergency_status['active_emergency'] = False
#         emergency_status['emergency_lane'] = None
#         emergency_status['cooldown_end_time'] = time.time() + PRIORITY_COOLDOWN
    
#     print("[MANUAL OVERRIDE] Emergency priority cleared")
#     return jsonify({
#         "status": "success",
#         "message": "Emergency priority cleared"
#     })

# if __name__ == "__main__":
#     # Start video processing threads
#     for lid in LANE_IDS:
#         t = threading.Thread(target=process_video_for_lane,
#                              args=(lid,), daemon=True)
#         t.start()
    
#     # Start emergency monitoring thread
#     emergency_thread = threading.Thread(target=emergency_monitor, daemon=True)
#     emergency_thread.start()
    
#     print("[SYSTEM] Traffic monitoring system started with smoothed green time calculations")
#     print("[SYSTEM] Emergency classes being monitored:", [CLASS_NAMES[c] for c in EMERGENCY_CLASSES])
#     print(f"[SYSTEM] Green time parameters: Base={BASE_GREEN_TIME}s, Min={MIN_GREEN_TIME}s, Max={MAX_GREEN_TIME}s")
#     print(f"[SYSTEM] Averaging window: {GREEN_TIME_AVERAGING_WINDOW} samples, Update interval: {UPDATE_INTERVAL}s")
    
#     app.run(debug=False, host="0.0.0.0", port=3001, threaded=True)












# combined
# from flask import Flask, render_template, Response, jsonify, request
# from werkzeug.utils import secure_filename
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import defaultdict, deque
# import time
# import threading
# import copy
# import os

# app = Flask(__name__)

# # --- Configuration ---
# MODEL = YOLO('yolov8n.pt')
# VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
# EMERGENCY_CLASSES = [0]  # person (can be extended to specific emergency vehicle classes)
# CLASS_NAMES = MODEL.names
# LANE_IDS = [1, 2, 3, 4]
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # --- Emergency Vehicle Detection Configuration ---
# EMERGENCY_PRIORITY_THRESHOLD = 1
# EMERGENCY_GREEN_TIME = 20
# PRIORITY_COOLDOWN = 30

# # --- Green Time Configuration ---
# GREEN_TIME_AVERAGING_WINDOW = 10  # Number of samples to average for green time
# MIN_GREEN_TIME = 10  # Minimum green time in seconds
# MAX_GREEN_TIME = 60  # Maximum green time in seconds
# BASE_GREEN_TIME = 10  # Base green time without vehicles
# TIME_PER_VEHICLE = 2  # Additional seconds per vehicle
# UPDATE_INTERVAL = 5   # Seconds between green time updates

# # --- Global Data Structures & Threading Lock ---
# traffic_data = {}
# emergency_status = {
#     'active_emergency': False,
#     'emergency_lane': None,
#     'priority_end_time': 0,
#     'cooldown_end_time': 0
# }
# lock = threading.Lock()
# yolo_lock = threading.Lock()
# video_sources = {}

# for lane_id in LANE_IDS:
#     traffic_data[lane_id] = {
#         'current_counts': defaultdict(int),
#         'emergency_count': 0,
#         'total_count': 0,
#         'latest_frame': None,
#         'green_time': BASE_GREEN_TIME,  # Start with base time
#         'target_green_time': BASE_GREEN_TIME,  # Target time we're moving toward
#         'current_green_time': BASE_GREEN_TIME,  # Currently displayed time
#         'density_history': deque(maxlen=20),
#         'timing_history': deque(maxlen=20),
#         'time_labels': deque(maxlen=20),
#         'vehicle_count_history': deque(maxlen=GREEN_TIME_AVERAGING_WINDOW),
#         'has_emergency': False,
#         'last_update_time': 0
#     }

# # --- Core Logic ---
# def calculate_target_green_time(vehicle_count, has_emergency=False):
#     """Calculate target green time based on vehicle count"""
#     if has_emergency:
#         return EMERGENCY_GREEN_TIME
    
#     calculated_time = BASE_GREEN_TIME + (vehicle_count * TIME_PER_VEHICLE)
#     return max(MIN_GREEN_TIME, min(calculated_time, MAX_GREEN_TIME))

# def smooth_green_time_transition(lane_id, current_time):
#     """Smoothly transition between green time values"""
#     with lock:
#         lane_data = traffic_data[lane_id]
        
#         if current_time - lane_data['last_update_time'] < UPDATE_INTERVAL:
#             return lane_data['current_green_time']
        
#         if lane_data['vehicle_count_history']:
#             avg_vehicles = sum(lane_data['vehicle_count_history']) / len(lane_data['vehicle_count_history'])
#             new_target = calculate_target_green_time(avg_vehicles, lane_data['has_emergency'])
#         else:
#             new_target = calculate_target_green_time(lane_data['total_count'], lane_data['has_emergency'])
        
#         current_gt = lane_data['current_green_time']
#         smoothed_gt = current_gt + 0.25 * (new_target - current_gt)
        
#         smoothed_gt = round(smoothed_gt)
#         smoothed_gt = max(MIN_GREEN_TIME, min(smoothed_gt, MAX_GREEN_TIME))
        
#         lane_data['target_green_time'] = new_target
#         lane_data['current_green_time'] = smoothed_gt
#         lane_data['last_update_time'] = current_time
        
#         return smoothed_gt

# def check_emergency_priority():
#     """Check if emergency priority should be activated or deactivated"""
#     global emergency_status
    
#     current_time = time.time()
    
#     if current_time < emergency_status['cooldown_end_time']:
#         return
    
#     if emergency_status['active_emergency']:
#         if current_time >= emergency_status['priority_end_time']:
#             with lock:
#                 emergency_status['active_emergency'] = False
#                 emergency_status['emergency_lane'] = None
#                 emergency_status['cooldown_end_time'] = current_time + PRIORITY_COOLDOWN
#             print(f"[EMERGENCY] Priority mode ended. Cooldown until {time.strftime('%H:%M:%S', time.localtime(emergency_status['cooldown_end_time']))}")
#         return
    
#     emergency_lane = None
#     max_emergency_count = 0
    
#     with lock:
#         for lane_id in LANE_IDS:
#             if traffic_data[lane_id]['emergency_count'] >= EMERGENCY_PRIORITY_THRESHOLD:
#                 if traffic_data[lane_id]['emergency_count'] > max_emergency_count:
#                     max_emergency_count = traffic_data[lane_id]['emergency_count']
#                     emergency_lane = lane_id
    
#     if emergency_lane is not None:
#         with lock:
#             emergency_status['active_emergency'] = True
#             emergency_status['emergency_lane'] = emergency_lane
#             emergency_status['priority_end_time'] = current_time + EMERGENCY_GREEN_TIME
#         print(f"[EMERGENCY] Priority activated for Lane {emergency_lane}! Green time: {EMERGENCY_GREEN_TIME}s")

# def process_video_for_lane(lane_id, video_path):
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"[ERROR] Could not open video for lane {lane_id}: {video_path}")
#         return

#     print(f"[INFO] Started processing for lane {lane_id}...")

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         with yolo_lock:
#             results = MODEL.track(frame, persist=True, verbose=False,
#                                   classes=VEHICLE_CLASSES + EMERGENCY_CLASSES)

#         current_counts = defaultdict(int)
#         emergency_count = 0
#         current_time = time.time()

#         if results and results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
#             confidences = results[0].boxes.conf.cpu().numpy()

#             for box, cid, conf in zip(boxes, class_ids, confidences):
#                 label = CLASS_NAMES[cid]
#                 current_counts[label] += 1
                
#                 if cid in EMERGENCY_CLASSES and conf > 0.6:
#                     emergency_count += 1
#                     color = (0, 0, 255)
#                     thickness = 3
#                 else:
#                     color = (0, 255, 0)
#                     thickness = 2
                
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
#                 cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         total = sum(current_counts.values())
        
#         with lock:
#             has_emergency = emergency_status['active_emergency'] and emergency_status['emergency_lane'] == lane_id
#             traffic_data[lane_id]['has_emergency'] = has_emergency
            
#             traffic_data[lane_id]['vehicle_count_history'].append(total)
        
#         smoothed_gtime = smooth_green_time_transition(lane_id, current_time)

#         status_color = (0, 0, 255) if has_emergency else (255, 255, 255)
#         status_text = "EMERGENCY PRIORITY" if has_emergency else "NORMAL"
        
#         cv2.putText(frame, f"Lane {lane_id} - {status_text}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
#         info = f"Total: {total} | Emergency: {emergency_count} | Green: {smoothed_gtime}s"
#         cv2.putText(frame, info, (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         if emergency_count > 0:
#             cv2.putText(frame, "🚑 EMERGENCY VEHICLE DETECTED", (10, 110),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         ret, buf = cv2.imencode(".jpg", frame)
#         frame_bytes = buf.tobytes()

#         with lock:
#             lane = traffic_data[lane_id]
#             lane["current_counts"] = current_counts
#             lane["emergency_count"] = emergency_count
#             lane["total_count"] = total
#             lane["green_time"] = smoothed_gtime
#             now = time.time()
#             last = lane["time_labels"][-1] if lane["time_labels"] else 0
#             if now - last > 5:
#                 lane["density_history"].append(total)
#                 lane["timing_history"].append(smoothed_gtime)
#                 lane["time_labels"].append(now)
            
#             lane["latest_frame"] = frame_bytes

# # --- Emergency Monitoring Thread ---
# def emergency_monitor():
#     """Continuously monitor for emergency vehicles and manage priority"""
#     while True:
#         check_emergency_priority()
#         time.sleep(1)

# # --- Flask Routes ---
# @app.route("/")
# def index():
#     return render_template("index.html", lanes=LANE_IDS)

# @app.route("/upload_and_process", methods=["POST"])
# def upload_and_process():
#     global video_sources
#     uploaded_files = request.files.to_dict()
#     if not uploaded_files:
#         return jsonify({"error": "No files uploaded"}), 400

#     video_sources = {}
#     for lane_name, file_storage in uploaded_files.items():
#         if file_storage.filename != '':
#             lane_id = int(lane_name.replace('lane', ''))
#             filename = secure_filename(file_storage.filename)
#             filepath = os.path.join(UPLOAD_FOLDER, filename)
#             file_storage.save(filepath)
#             video_sources[lane_id] = filepath

#     if not video_sources:
#         return jsonify({"error": "No video files provided"}), 400

#     for lid in video_sources.keys():
#         t = threading.Thread(target=process_video_for_lane,
#                              args=(lid, video_sources[lid]), daemon=True)
#         t.start()

#     emergency_thread = threading.Thread(target=emergency_monitor, daemon=True)
#     emergency_thread.start()

#     print("[SYSTEM] Traffic monitoring system started with uploaded videos")
#     print("[SYSTEM] Emergency classes being monitored:", [CLASS_NAMES[c] for c in EMERGENCY_CLASSES])
    
#     return jsonify({
#         "status": "success",
#         "message": "Videos uploaded and processing started.",
#         "processed_lanes": list(video_sources.keys())
#     })

# def generate_video_stream(lane_id):
#     while True:
#         time.sleep(0.03)
#         with lock:
#             frame = traffic_data[lane_id]["latest_frame"]
#         if frame is None:
#             continue
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
#                frame + b"\r\n")

# @app.route("/video_feed/<int:lane_id>")
# def video_feed(lane_id):
#     if lane_id not in LANE_IDS:
#         return "Invalid Lane ID", 404
#     return Response(generate_video_stream(lane_id),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/traffic_data")
# def get_traffic_data():
#     with lock:
#         payload = {
#             'lanes': {},
#             'emergency_status': emergency_status.copy()
#         }
        
#         emergency_status_readable = emergency_status.copy()
#         emergency_status_readable['priority_end_time'] = time.strftime("%H:%M:%S", time.localtime(emergency_status['priority_end_time'])) if emergency_status['priority_end_time'] > 0 else "N/A"
#         emergency_status_readable['cooldown_end_time'] = time.strftime("%H:%M:%S", time.localtime(emergency_status['cooldown_end_time'])) if emergency_status['cooldown_end_time'] > 0 else "N/A"
#         payload['emergency_status'] = emergency_status_readable
        
#         for lid, data in traffic_data.items():
#             d = {k: v for k, v in data.items() if k != "latest_frame"}
#             d["density_history"] = list(d["density_history"])
#             d["timing_history"] = list(d["timing_history"])
#             d["time_labels"] = [time.strftime("%H:%M:%S", time.localtime(ts)) for ts in d["time_labels"]]
#             d["vehicle_count_history"] = list(d["vehicle_count_history"])
#             payload['lanes'][lid] = d
    
#     return jsonify(payload)

# @app.route("/emergency_override/<int:lane_id>", methods=["POST"])
# def emergency_override(lane_id):
#     """Manual emergency override endpoint"""
#     if lane_id not in LANE_IDS:
#         return jsonify({"error": "Invalid lane ID"}), 400
    
#     with lock:
#         emergency_status['active_emergency'] = True
#         emergency_status['emergency_lane'] = lane_id
#         emergency_status['priority_end_time'] = time.time() + EMERGENCY_GREEN_TIME
#         emergency_status['cooldown_end_time'] = 0
        
#         traffic_data[lane_id]['current_green_time'] = EMERGENCY_GREEN_TIME
#         traffic_data[lane_id]['target_green_time'] = EMERGENCY_GREEN_TIME
    
#     print(f"[MANUAL OVERRIDE] Emergency priority activated for Lane {lane_id}")
#     return jsonify({
#         "status": "success",
#         "message": f"Emergency priority activated for Lane {lane_id}",
#         "green_time": EMERGENCY_GREEN_TIME
#     })

# @app.route("/clear_emergency", methods=["POST"])
# def clear_emergency():
#     """Clear emergency priority manually"""
#     with lock:
#         emergency_status['active_emergency'] = False
#         emergency_status['emergency_lane'] = None
#         emergency_status['cooldown_end_time'] = time.time() + PRIORITY_COOLDOWN
    
#     print("[MANUAL OVERRIDE] Emergency priority cleared")
#     return jsonify({
#         "status": "success",
#         "message": "Emergency priority cleared"
#     })

# if __name__ == "__main__":
#     app.run(debug=False, host="0.0.0.0", port=3001, threaded=True)










#  accident prevention -> a little bit incorrect
# from flask import Flask, render_template, Response, jsonify, request
# from werkzeug.utils import secure_filename
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import defaultdict, deque
# import time
# import threading
# import copy
# import os

# app = Flask(__name__)

# # --- Configuration ---
# MODEL = YOLO('yolov8n.pt')
# VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
# EMERGENCY_CLASSES = [0]  # person (can be extended to specific emergency vehicle classes)
# CLASS_NAMES = MODEL.names
# LANE_IDS = [1, 2, 3, 4]
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # --- Emergency Vehicle Detection Configuration ---
# EMERGENCY_PRIORITY_THRESHOLD = 1
# EMERGENCY_GREEN_TIME = 20
# PRIORITY_COOLDOWN = 30

# # --- Green Time Configuration ---
# GREEN_TIME_AVERAGING_WINDOW = 10  # Number of samples to average for green time
# MIN_GREEN_TIME = 10  # Minimum green time in seconds
# MAX_GREEN_TIME = 60  # Maximum green time in seconds
# BASE_GREEN_TIME = 10  # Base green time without vehicles
# TIME_PER_VEHICLE = 2  # Additional seconds per vehicle
# UPDATE_INTERVAL = 5   # Seconds between green time updates

# # --- Global Data Structures & Threading Lock ---
# traffic_data = {}
# system_status = {
#     'priority_mode': 'normal', # 'normal', 'emergency', 'accident'
#     'active_lane': None,
#     'priority_end_time': 0,
#     'cooldown_end_time': 0
# }
# lock = threading.Lock()
# yolo_lock = threading.Lock()
# video_sources = {}

# for lane_id in LANE_IDS:
#     traffic_data[lane_id] = {
#         'current_counts': defaultdict(int),
#         'emergency_count': 0,
#         'total_count': 0,
#         'latest_frame': None,
#         'green_time': BASE_GREEN_TIME,
#         'target_green_time': BASE_GREEN_TIME,
#         'current_green_time': BASE_GREEN_TIME,
#         'density_history': deque(maxlen=20),
#         'timing_history': deque(maxlen=20),
#         'time_labels': deque(maxlen=20),
#         'vehicle_count_history': deque(maxlen=GREEN_TIME_AVERAGING_WINDOW),
#         'has_emergency': False,
#         'last_update_time': 0,
#         'tracked_objects': {},
#         'accident_detected': False,
#         'accident_location': None
#     }

# # --- Core Logic ---
# def calculate_target_green_time(vehicle_count, has_emergency=False):
#     """Calculate target green time based on vehicle count"""
#     if has_emergency:
#         return EMERGENCY_GREEN_TIME
    
#     calculated_time = BASE_GREEN_TIME + (vehicle_count * TIME_PER_VEHICLE)
#     return max(MIN_GREEN_TIME, min(calculated_time, MAX_GREEN_TIME))

# def smooth_green_time_transition(lane_id, current_time):
#     """Smoothly transition between green time values"""
#     with lock:
#         lane_data = traffic_data[lane_id]
        
#         if current_time - lane_data['last_update_time'] < UPDATE_INTERVAL:
#             return lane_data['current_green_time']
        
#         if lane_data['vehicle_count_history']:
#             avg_vehicles = sum(lane_data['vehicle_count_history']) / len(lane_data['vehicle_count_history'])
#             new_target = calculate_target_green_time(avg_vehicles, lane_data['has_emergency'])
#         else:
#             new_target = calculate_target_green_time(lane_data['total_count'], lane_data['has_emergency'])
        
#         current_gt = lane_data['current_green_time']
#         smoothed_gt = current_gt + 0.25 * (new_target - current_gt)
        
#         smoothed_gt = round(smoothed_gt)
#         smoothed_gt = max(MIN_GREEN_TIME, min(smoothed_gt, MAX_GREEN_TIME))
        
#         lane_data['target_green_time'] = new_target
#         lane_data['current_green_time'] = smoothed_gt
#         lane_data['last_update_time'] = current_time
        
#         return smoothed_gt

# def check_overlap(box1, box2):
#     """Simple check for bounding box overlap."""
#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2
    
#     return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

# def manage_priorities():
#     """Check for emergency/accident and manage priority status"""
#     global system_status
    
#     current_time = time.time()
    
#     if current_time < system_status['cooldown_end_time']:
#         return
    
#     if system_status['priority_mode'] == 'accident':
#         return # Accident mode requires manual reset
    
#     # Check if a previously active emergency has ended
#     if system_status['priority_mode'] == 'emergency' and current_time >= system_status['priority_end_time']:
#         with lock:
#             system_status['priority_mode'] = 'normal'
#             system_status['active_lane'] = None
#             system_status['cooldown_end_time'] = current_time + PRIORITY_COOLDOWN
#         print(f"[PRIORITY] Emergency mode ended. Cooldown until {time.strftime('%H:%M:%S', time.localtime(system_status['cooldown_end_time']))}")
#         return

#     # Check for new emergency
#     emergency_lane = None
#     max_emergency_count = 0
#     with lock:
#         for lane_id in LANE_IDS:
#             if traffic_data[lane_id]['emergency_count'] >= EMERGENCY_PRIORITY_THRESHOLD:
#                 if traffic_data[lane_id]['emergency_count'] > max_emergency_count:
#                     max_emergency_count = traffic_data[lane_id]['emergency_count']
#                     emergency_lane = lane_id
    
#     if emergency_lane is not None:
#         with lock:
#             system_status['priority_mode'] = 'emergency'
#             system_status['active_lane'] = emergency_lane
#             system_status['priority_end_time'] = current_time + EMERGENCY_GREEN_TIME
#         print(f"[PRIORITY] Emergency activated for Lane {emergency_lane}! Green time: {EMERGENCY_GREEN_TIME}s")

# def process_video_for_lane(lane_id, video_path):
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"[ERROR] Could not open video for lane {lane_id}: {video_path}")
#         return

#     print(f"[INFO] Started processing for lane {lane_id}...")
    
#     prev_positions = defaultdict(lambda: deque(maxlen=5)) # For sudden stop detection

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         current_time = time.time()
        
#         with yolo_lock:
#             results = MODEL.track(frame, persist=True, verbose=False,
#                                   classes=VEHICLE_CLASSES + EMERGENCY_CLASSES)

#         current_counts = defaultdict(int)
#         emergency_count = 0
#         accident_detected = False
#         accident_location = None
#         current_tracked_ids = set()

#         if results and results[0].boxes.id is not None:
#             track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
#             confidences = results[0].boxes.conf.cpu().numpy()

#             for box, track_id, cid, conf in zip(boxes, track_ids, class_ids, confidences):
#                 label = CLASS_NAMES[cid]
#                 current_counts[label] += 1
#                 current_tracked_ids.add(track_id)
                
#                 # Update tracked object history
#                 x1, y1, x2, y2 = map(int, box)
#                 center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
#                 # Check for emergency vehicle
#                 if cid in EMERGENCY_CLASSES and conf > 0.6:
#                     emergency_count += 1
#                     color = (0, 0, 255)
#                     thickness = 3
#                 else:
#                     color = (0, 255, 0)
#                     thickness = 2
                
#                 # Check for sudden stop
#                 prev_positions[track_id].append({'pos': (center_x, center_y), 'time': current_time})
#                 if len(prev_positions[track_id]) >= 3:
#                     p1 = prev_positions[track_id][-3]
#                     p2 = prev_positions[track_id][-2]
#                     p3 = prev_positions[track_id][-1]
#                     dist_1_2 = np.sqrt((p2['pos'][0] - p1['pos'][0])**2 + (p2['pos'][1] - p1['pos'][1])**2)
#                     dist_2_3 = np.sqrt((p3['pos'][0] - p2['pos'][0])**2 + (p3['pos'][1] - p2['pos'][1])**2)
                    
#                     if dist_1_2 > 20 and dist_2_3 < 5: # Was moving, now stopped
#                         accident_detected = True
#                         accident_location = (center_x, center_y)
#                         break

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
#                 cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
#             # Check for bounding box overlap (collision)
#             if not accident_detected:
#                 for i in range(len(boxes)):
#                     for j in range(i + 1, len(boxes)):
#                         if check_overlap(boxes[i], boxes[j]) and confidences[i] > 0.5 and confidences[j] > 0.5:
#                             accident_detected = True
#                             accident_location = ((boxes[i][0]+boxes[i][2])/2, (boxes[i][1]+boxes[i][3])/2)
#                             break
#                     if accident_detected:
#                         break

#         total = sum(current_counts.values())
        
#         with lock:
#             has_emergency = system_status['priority_mode'] == 'emergency' and system_status['active_lane'] == lane_id
            
#             traffic_data[lane_id]['has_emergency'] = has_emergency
#             traffic_data[lane_id]['vehicle_count_history'].append(total)
#             traffic_data[lane_id]['accident_detected'] = accident_detected
#             traffic_data[lane_id]['accident_location'] = accident_location
            
#             if accident_detected:
#                 system_status['priority_mode'] = 'accident'
#                 system_status['active_lane'] = lane_id

#         smoothed_gtime = smooth_green_time_transition(lane_id, current_time)
        
#         if system_status['priority_mode'] == 'accident':
#             status_text = "ACCIDENT DETECTED"
#             status_color = (0, 0, 255)
#             cv2.putText(frame, "❗ ACCIDENT DETECTED ❗", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
#             info = f"Lane {lane_id} - Accident Location: ({int(accident_location[0])},{int(accident_location[1])})" if accident_location else "N/A"
#             cv2.putText(frame, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
#         else:
#             status_color = (0, 0, 255) if has_emergency else (255, 255, 255)
#             status_text = "EMERGENCY PRIORITY" if has_emergency else "NORMAL"
#             cv2.putText(frame, f"Lane {lane_id} - {status_text}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
#             info = f"Total: {total} | Emergency: {emergency_count} | Green: {smoothed_gtime}s"
#             cv2.putText(frame, info, (10, 70),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         if emergency_count > 0:
#             cv2.putText(frame, "🚑 EMERGENCY VEHICLE DETECTED", (10, 110),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         ret, buf = cv2.imencode(".jpg", frame)
#         frame_bytes = buf.tobytes()

#         with lock:
#             lane = traffic_data[lane_id]
#             lane["current_counts"] = current_counts
#             lane["emergency_count"] = emergency_count
#             lane["total_count"] = total
#             lane["green_time"] = smoothed_gtime
#             now = time.time()
#             last = lane["time_labels"][-1] if lane["time_labels"] else 0
#             if now - last > 5:
#                 lane["density_history"].append(total)
#                 lane["timing_history"].append(smoothed_gtime)
#                 lane["time_labels"].append(now)
            
#             lane["latest_frame"] = frame_bytes
        
# # --- Emergency/Accident Monitoring Thread ---
# def priority_monitor():
#     """Continuously monitor for emergency vehicles and accidents and manage priority"""
#     while True:
#         manage_priorities()
#         time.sleep(1)

# # --- Flask Routes ---
# @app.route("/")
# def index():
#     return render_template("index.html", lanes=LANE_IDS)

# @app.route("/upload_and_process", methods=["POST"])
# def upload_and_process():
#     global video_sources
#     uploaded_files = request.files.to_dict()
#     if not uploaded_files:
#         return jsonify({"error": "No files uploaded"}), 400

#     video_sources = {}
#     for lane_name, file_storage in uploaded_files.items():
#         if file_storage.filename != '':
#             lane_id = int(lane_name.replace('lane', ''))
#             filename = secure_filename(file_storage.filename)
#             filepath = os.path.join(UPLOAD_FOLDER, filename)
#             file_storage.save(filepath)
#             video_sources[lane_id] = filepath

#     if not video_sources:
#         return jsonify({"error": "No video files provided"}), 400

#     for lid in video_sources.keys():
#         t = threading.Thread(target=process_video_for_lane,
#                              args=(lid, video_sources[lid]), daemon=True)
#         t.start()

#     priority_thread = threading.Thread(target=priority_monitor, daemon=True)
#     priority_thread.start()

#     print("[SYSTEM] Traffic monitoring system started with uploaded videos")
#     print("[SYSTEM] Emergency classes being monitored:", [CLASS_NAMES[c] for c in EMERGENCY_CLASSES])
    
#     return jsonify({
#         "status": "success",
#         "message": "Videos uploaded and processing started.",
#         "processed_lanes": list(video_sources.keys())
#     })

# def generate_video_stream(lane_id):
#     while True:
#         time.sleep(0.03)
#         with lock:
#             frame = traffic_data[lane_id]["latest_frame"]
#         if frame is None:
#             continue
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
#                frame + b"\r\n")

# @app.route("/video_feed/<int:lane_id>")
# def video_feed(lane_id):
#     if lane_id not in LANE_IDS:
#         return "Invalid Lane ID", 404
#     return Response(generate_video_stream(lane_id),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/traffic_data")
# def get_traffic_data():
#     with lock:
#         payload = {
#             'lanes': {},
#             'system_status': system_status.copy()
#         }
        
#         system_status_readable = system_status.copy()
#         system_status_readable['priority_end_time'] = time.strftime("%H:%M:%S", time.localtime(system_status['priority_end_time'])) if system_status['priority_end_time'] > 0 else "N/A"
#         system_status_readable['cooldown_end_time'] = time.strftime("%H:%M:%S", time.localtime(system_status['cooldown_end_time'])) if system_status['cooldown_end_time'] > 0 else "N/A"
#         payload['system_status'] = system_status_readable
        
#         for lid, data in traffic_data.items():
#             d = {k: v for k, v in data.items() if k not in ["latest_frame", "tracked_objects"]}
#             d["density_history"] = list(d["density_history"])
#             d["timing_history"] = list(d["timing_history"])
#             d["time_labels"] = [time.strftime("%H:%M:%S", time.localtime(ts)) for ts in d["time_labels"]]
#             d["vehicle_count_history"] = list(d["vehicle_count_history"])
#             payload['lanes'][lid] = d
    
#     return jsonify(payload)

# @app.route("/emergency_override/<int:lane_id>", methods=["POST"])
# def emergency_override(lane_id):
#     """Manual emergency override endpoint"""
#     if lane_id not in LANE_IDS:
#         return jsonify({"error": "Invalid lane ID"}), 400
    
#     with lock:
#         system_status['priority_mode'] = 'emergency'
#         system_status['active_lane'] = lane_id
#         system_status['priority_end_time'] = time.time() + EMERGENCY_GREEN_TIME
#         system_status['cooldown_end_time'] = 0
        
#         traffic_data[lane_id]['current_green_time'] = EMERGENCY_GREEN_TIME
#         traffic_data[lane_id]['target_green_time'] = EMERGENCY_GREEN_TIME
    
#     print(f"[MANUAL OVERRIDE] Emergency priority activated for Lane {lane_id}")
#     return jsonify({
#         "status": "success",
#         "message": f"Emergency priority activated for Lane {lane_id}",
#         "green_time": EMERGENCY_GREEN_TIME
#     })

# @app.route("/clear_emergency", methods=["POST"])
# def clear_emergency():
#     """Clear emergency priority manually"""
#     with lock:
#         system_status['priority_mode'] = 'normal'
#         system_status['active_lane'] = None
#         system_status['cooldown_end_time'] = time.time() + PRIORITY_COOLDOWN
    
#     print("[MANUAL OVERRIDE] Emergency priority cleared")
#     return jsonify({
#         "status": "success",
#         "message": "Emergency priority cleared"
#     })

# @app.route("/clear_accident", methods=["POST"])
# def clear_accident():
#     """Manually clear an accident alert"""
#     with lock:
#         for lane_id in LANE_IDS:
#             traffic_data[lane_id]['accident_detected'] = False
#             traffic_data[lane_id]['accident_location'] = None
#         system_status['priority_mode'] = 'normal'
#         system_status['active_lane'] = None
#         system_status['cooldown_end_time'] = time.time() + PRIORITY_COOLDOWN
    
#     print("[MANUAL OVERRIDE] Accident alert cleared")
#     return jsonify({
#         "status": "success",
#         "message": "Accident alert cleared and system reset."
#     })

# if __name__ == "__main__":
#     app.run(debug=False, host="0.0.0.0", port=3001, threaded=True)












# accident prevention -> imporved version
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import time
import threading
import copy
import os

app = Flask(__name__)

# --- Configuration ---
MODEL = YOLO('yolov8n.pt')
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
EMERGENCY_CLASSES = [0]  # person (can be extended to specific emergency vehicle classes)
CLASS_NAMES = MODEL.names
LANE_IDS = [1, 2, 3, 4]
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Emergency Vehicle Detection Configuration ---
EMERGENCY_PRIORITY_THRESHOLD = 1
EMERGENCY_GREEN_TIME = 20
PRIORITY_COOLDOWN = 30

# --- Green Time Configuration ---
GREEN_TIME_AVERAGING_WINDOW = 10  # Number of samples to average for green time
MIN_GREEN_TIME = 10  # Minimum green time in seconds
MAX_GREEN_TIME = 60  # Maximum green time in seconds
BASE_GREEN_TIME = 10  # Base green time without vehicles
TIME_PER_VEHICLE = 2  # Additional seconds per vehicle
UPDATE_INTERVAL = 5   # Seconds between green time updates

# --- Global Data Structures & Threading Lock ---
traffic_data = {}
system_status = {
    'priority_mode': 'normal', # 'normal', 'emergency', 'accident'
    'active_lane': None,
    'priority_end_time': 0,
    'cooldown_end_time': 0
}
lock = threading.Lock()
yolo_lock = threading.Lock()
video_sources = {}

for lane_id in LANE_IDS:
    traffic_data[lane_id] = {
        'current_counts': defaultdict(int),
        'emergency_count': 0,
        'total_count': 0,
        'latest_frame': None,
        'green_time': BASE_GREEN_TIME,
        'target_green_time': BASE_GREEN_TIME,
        'current_green_time': BASE_GREEN_TIME,
        'density_history': deque(maxlen=20),
        'timing_history': deque(maxlen=20),
        'time_labels': deque(maxlen=20),
        'vehicle_count_history': deque(maxlen=GREEN_TIME_AVERAGING_WINDOW),
        'has_emergency': False,
        'last_update_time': 0,
        'tracked_objects': {},
        'accident_detected': False,
        'accident_location': None
    }

# --- Core Logic ---
def calculate_target_green_time(vehicle_count, has_emergency=False):
    """Calculate target green time based on vehicle count"""
    if has_emergency:
        return EMERGENCY_GREEN_TIME
    
    calculated_time = BASE_GREEN_TIME + (vehicle_count * TIME_PER_VEHICLE)
    return max(MIN_GREEN_TIME, min(calculated_time, MAX_GREEN_TIME))

def smooth_green_time_transition(lane_id, current_time):
    """Smoothly transition between green time values"""
    with lock:
        lane_data = traffic_data[lane_id]
        
        if current_time - lane_data['last_update_time'] < UPDATE_INTERVAL:
            return lane_data['current_green_time']
        
        if lane_data['vehicle_count_history']:
            avg_vehicles = sum(lane_data['vehicle_count_history']) / len(lane_data['vehicle_count_history'])
            new_target = calculate_target_green_time(avg_vehicles, lane_data['has_emergency'])
        else:
            new_target = calculate_target_green_time(lane_data['total_count'], lane_data['has_emergency'])
        
        current_gt = lane_data['current_green_time']
        smoothed_gt = current_gt + 0.25 * (new_target - current_gt)
        
        smoothed_gt = round(smoothed_gt)
        smoothed_gt = max(MIN_GREEN_TIME, min(smoothed_gt, MAX_GREEN_TIME))
        
        lane_data['target_green_time'] = new_target
        lane_data['current_green_time'] = smoothed_gt
        lane_data['last_update_time'] = current_time
        
        return smoothed_gt

def check_overlap(box1, box2):
    """Simple check for bounding box overlap."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

def manage_priorities():
    """Check for emergency/accident and manage priority status"""
    global system_status
    
    current_time = time.time()
    
    if current_time < system_status['cooldown_end_time']:
        return
    
    if system_status['priority_mode'] == 'accident':
        return # Accident mode requires manual reset
    
    # Check if a previously active emergency has ended
    if system_status['priority_mode'] == 'emergency' and current_time >= system_status['priority_end_time']:
        with lock:
            system_status['priority_mode'] = 'normal'
            system_status['active_lane'] = None
            system_status['cooldown_end_time'] = current_time + PRIORITY_COOLDOWN
        print(f"[PRIORITY] Emergency mode ended. Cooldown until {time.strftime('%H:%M:%S', time.localtime(system_status['cooldown_end_time']))}")
        return

    # Check for new emergency
    emergency_lane = None
    max_emergency_count = 0
    with lock:
        for lane_id in LANE_IDS:
            if traffic_data[lane_id]['emergency_count'] >= EMERGENCY_PRIORITY_THRESHOLD:
                if traffic_data[lane_id]['emergency_count'] > max_emergency_count:
                    max_emergency_count = traffic_data[lane_id]['emergency_count']
                    emergency_lane = lane_id
    
    if emergency_lane is not None:
        with lock:
            system_status['priority_mode'] = 'emergency'
            system_status['active_lane'] = emergency_lane
            system_status['priority_end_time'] = current_time + EMERGENCY_GREEN_TIME
        print(f"[PRIORITY] Emergency activated for Lane {emergency_lane}! Green time: {EMERGENCY_GREEN_TIME}s")

def process_video_for_lane(lane_id, video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video for lane {lane_id}: {video_path}")
        return

    print(f"[INFO] Started processing for lane {lane_id}...")
    
    prev_positions = defaultdict(lambda: deque(maxlen=5)) # For sudden stop detection

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        current_time = time.time()
        
        with yolo_lock:
            results = MODEL.track(frame, persist=True, verbose=False,
                                  classes=VEHICLE_CLASSES + EMERGENCY_CLASSES)

        current_counts = defaultdict(int)
        emergency_count = 0
        accident_detected = False
        accident_location = None
        current_tracked_ids = set()

        if results and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cid, conf in zip(boxes, track_ids, class_ids, confidences):
                label = CLASS_NAMES[cid]
                current_counts[label] += 1
                current_tracked_ids.add(track_id)
                
                # Update tracked object history
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Check for emergency vehicle
                if cid in EMERGENCY_CLASSES and conf > 0.6:
                    emergency_count += 1
                    color = (0, 0, 255)
                    thickness = 3
                else:
                    color = (0, 255, 0)
                    thickness = 2
                
                # Check for sudden stop
                prev_positions[track_id].append({'pos': (center_x, center_y), 'time': current_time})
                if len(prev_positions[track_id]) >= 3:
                    p1 = prev_positions[track_id][-3]
                    p2 = prev_positions[track_id][-2]
                    p3 = prev_positions[track_id][-1]
                    dist_1_2 = np.sqrt((p2['pos'][0] - p1['pos'][0])**2 + (p2['pos'][1] - p1['pos'][1])**2)
                    dist_2_3 = np.sqrt((p3['pos'][0] - p2['pos'][0])**2 + (p3['pos'][1] - p2['pos'][1])**2)
                    
                    if dist_1_2 > 20 and dist_2_3 < 5: # Was moving, now stopped
                        accident_detected = True
                        accident_location = (float(center_x), float(center_y)) # FIX APPLIED HERE
                        break

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Check for bounding box overlap (collision)
            if not accident_detected:
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        if check_overlap(boxes[i], boxes[j]) and confidences[i] > 0.5 and confidences[j] > 0.5:
                            accident_detected = True
                            # FIX: Convert numpy.float32 values to standard floats
                            accident_location = (float((boxes[i][0]+boxes[i][2])/2), float((boxes[i][1]+boxes[i][3])/2))
                            break
                    if accident_detected:
                        break

        total = sum(current_counts.values())
        
        with lock:
            has_emergency = system_status['priority_mode'] == 'emergency' and system_status['active_lane'] == lane_id
            
            traffic_data[lane_id]['has_emergency'] = has_emergency
            traffic_data[lane_id]['vehicle_count_history'].append(total)
            traffic_data[lane_id]['accident_detected'] = accident_detected
            traffic_data[lane_id]['accident_location'] = accident_location
            
            if accident_detected:
                system_status['priority_mode'] = 'accident'
                system_status['active_lane'] = lane_id

        smoothed_gtime = smooth_green_time_transition(lane_id, current_time)
        
        if system_status['priority_mode'] == 'accident':
            status_text = "ACCIDENT DETECTED"
            status_color = (0, 0, 255)
            cv2.putText(frame, "❗ ACCIDENT DETECTED ❗", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            info = f"Lane {lane_id} - Accident Location: ({int(accident_location[0])},{int(accident_location[1])})" if accident_location else "N/A"
            cv2.putText(frame, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            status_color = (0, 0, 255) if has_emergency else (255, 255, 255)
            status_text = "EMERGENCY PRIORITY" if has_emergency else "NORMAL"
            cv2.putText(frame, f"Lane {lane_id} - {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            info = f"Total: {total} | Emergency: {emergency_count} | Green: {smoothed_gtime}s"
            cv2.putText(frame, info, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if emergency_count > 0:
            cv2.putText(frame, "🚑 EMERGENCY VEHICLE DETECTED", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buf = cv2.imencode(".jpg", frame)
        frame_bytes = buf.tobytes()

        with lock:
            lane = traffic_data[lane_id]
            lane["current_counts"] = current_counts
            lane["emergency_count"] = emergency_count
            lane["total_count"] = total
            lane["green_time"] = smoothed_gtime
            now = time.time()
            last = lane["time_labels"][-1] if lane["time_labels"] else 0
            if now - last > 5:
                lane["density_history"].append(total)
                lane["timing_history"].append(smoothed_gtime)
                lane["time_labels"].append(now)
            
            lane["latest_frame"] = frame_bytes
        
# --- Emergency/Accident Monitoring Thread ---
def priority_monitor():
    """Continuously monitor for emergency vehicles and accidents and manage priority"""
    while True:
        manage_priorities()
        time.sleep(1)

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html", lanes=LANE_IDS)

@app.route("/upload_and_process", methods=["POST"])
def upload_and_process():
    global video_sources
    uploaded_files = request.files.to_dict()
    if not uploaded_files:
        return jsonify({"error": "No files uploaded"}), 400

    video_sources = {}
    for lane_name, file_storage in uploaded_files.items():
        if file_storage.filename != '':
            lane_id = int(lane_name.replace('lane', ''))
            filename = secure_filename(file_storage.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file_storage.save(filepath)
            video_sources[lane_id] = filepath

    if not video_sources:
        return jsonify({"error": "No video files provided"}), 400

    for lid in video_sources.keys():
        t = threading.Thread(target=process_video_for_lane,
                             args=(lid, video_sources[lid]), daemon=True)
        t.start()

    priority_thread = threading.Thread(target=priority_monitor, daemon=True)
    priority_thread.start()

    print("[SYSTEM] Traffic monitoring system started with uploaded videos")
    print("[SYSTEM] Emergency classes being monitored:", [CLASS_NAMES[c] for c in EMERGENCY_CLASSES])
    
    return jsonify({
        "status": "success",
        "message": "Videos uploaded and processing started.",
        "processed_lanes": list(video_sources.keys())
    })

def generate_video_stream(lane_id):
    while True:
        time.sleep(0.03)
        with lock:
            frame = traffic_data[lane_id]["latest_frame"]
        if frame is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               frame + b"\r\n")

@app.route("/video_feed/<int:lane_id>")
def video_feed(lane_id):
    if lane_id not in LANE_IDS:
        return "Invalid Lane ID", 404
    return Response(generate_video_stream(lane_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/traffic_data")
def get_traffic_data():
    with lock:
        payload = {
            'lanes': {},
            'system_status': system_status.copy()
        }
        
        system_status_readable = system_status.copy()
        system_status_readable['priority_end_time'] = time.strftime("%H:%M:%S", time.localtime(system_status['priority_end_time'])) if system_status['priority_end_time'] > 0 else "N/A"
        system_status_readable['cooldown_end_time'] = time.strftime("%H:%M:%S", time.localtime(system_status['cooldown_end_time'])) if system_status['cooldown_end_time'] > 0 else "N/A"
        payload['system_status'] = system_status_readable
        
        for lid, data in traffic_data.items():
            d = {k: v for k, v in data.items() if k not in ["latest_frame", "tracked_objects"]}
            d["density_history"] = list(d["density_history"])
            d["timing_history"] = list(d["timing_history"])
            d["time_labels"] = [time.strftime("%H:%M:%S", time.localtime(ts)) for ts in d["time_labels"]]
            d["vehicle_count_history"] = list(d["vehicle_count_history"])
            payload['lanes'][lid] = d
    
    return jsonify(payload)

@app.route("/emergency_override/<int:lane_id>", methods=["POST"])
def emergency_override(lane_id):
    """Manual emergency override endpoint"""
    if lane_id not in LANE_IDS:
        return jsonify({"error": "Invalid lane ID"}), 400
    
    with lock:
        system_status['priority_mode'] = 'emergency'
        system_status['active_lane'] = lane_id
        system_status['priority_end_time'] = time.time() + EMERGENCY_GREEN_TIME
        system_status['cooldown_end_time'] = 0
        
        traffic_data[lane_id]['current_green_time'] = EMERGENCY_GREEN_TIME
        traffic_data[lane_id]['target_green_time'] = EMERGENCY_GREEN_TIME
    
    print(f"[MANUAL OVERRIDE] Emergency priority activated for Lane {lane_id}")
    return jsonify({
        "status": "success",
        "message": f"Emergency priority activated for Lane {lane_id}",
        "green_time": EMERGENCY_GREEN_TIME
    })

@app.route("/clear_emergency", methods=["POST"])
def clear_emergency():
    """Clear emergency priority manually"""
    with lock:
        system_status['priority_mode'] = 'normal'
        system_status['active_lane'] = None
        system_status['cooldown_end_time'] = time.time() + PRIORITY_COOLDOWN
    
    print("[MANUAL OVERRIDE] Emergency priority cleared")
    return jsonify({
        "status": "success",
        "message": "Emergency priority cleared"
    })

@app.route("/clear_accident", methods=["POST"])
def clear_accident():
    """Manually clear an accident alert"""
    with lock:
        for lane_id in LANE_IDS:
            traffic_data[lane_id]['accident_detected'] = False
            traffic_data[lane_id]['accident_location'] = None
        system_status['priority_mode'] = 'normal'
        system_status['active_lane'] = None
        system_status['cooldown_end_time'] = time.time() + PRIORITY_COOLDOWN
    
    print("[MANUAL OVERRIDE] Accident alert cleared")
    return jsonify({
        "status": "success",
        "message": "Accident alert cleared and system reset."
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3001, threaded=True)