import threading
import time
import random
from datetime import datetime, timedelta
from app import db
from app.models.traffic_model import Intersection, TrafficData, Alert

def simulate_traffic(socketio):
    """Simulate traffic data changes"""
    # Create some initial intersections
    if Intersection.query.count() == 0:
        intersections = [
            Intersection(name="Main St & 1st Ave", lat=40.7128, lng=-74.0060),
            Intersection(name="Oak St & Pine Ave", lat=40.7138, lng=-74.0070),
            Intersection(name="Central Square", lat=40.7148, lng=-74.0080),
            Intersection(name="North Bridge", lat=40.7158, lng=-74.0090),
        ]
        for intersection in intersections:
            db.session.add(intersection)
        db.session.commit()
    
    while True:
        intersections = Intersection.query.all()
        for intersection in intersections:
            # Simulate traffic fluctuations
            time_factor = 1.0
            hour = datetime.now().hour
            if (7 <= hour <= 9) or (16 <= hour <= 19):  # Rush hours
                time_factor = 2.5
            
            # Update vehicle count
            new_count = max(5, int(intersection.vehicle_count * 0.9 + random.randint(0, 20) * time_factor))
            intersection.vehicle_count = new_count
            
            # Update average speed
            speed_change = random.randint(-5, 5)
            new_speed = max(10, min(100, intersection.avg_speed + speed_change))
            intersection.avg_speed = new_speed
            
            # Update congestion level
            if new_count > 40:
                intersection.congestion_level = "high"
            elif new_count > 20:
                intersection.congestion_level = "medium"
            else:
                intersection.congestion_level = "low"
            
            intersection.last_updated = datetime.utcnow()
            
            # Store historical data
            traffic_data = TrafficData(
                intersection_id=intersection.id,
                vehicle_count=new_count,
                avg_speed=new_speed,
                weather_condition=random.choice(["clear", "rainy", "cloudy"]),
                road_condition=random.choice(["dry", "wet", "icy"])
            )
            db.session.add(traffic_data)
            
            # Occasionally generate alerts for demonstration
            if random.random() < 0.05:  # 5% chance each interval
                alert = Alert(
                    message=f"Potential risk detected at {intersection.name}",
                    severity=random.choice(["warning", "danger"]),
                    location=intersection.name
                )
                db.session.add(alert)
                socketio.emit('new_alert', {
                    'message': alert.message,
                    'severity': alert.severity,
                    'location': alert.location,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        db.session.commit()
        
        # Emit update to all connected clients
        socketio.emit('traffic_update', {
            'timestamp': datetime.utcnow().isoformat(),
            'message': 'Traffic data updated'
        })
        
        time.sleep(5)  # Update every 5 seconds

def start_simulation(socketio):
    """Start the simulation in a background thread"""
    thread = threading.Thread(target=simulate_traffic, args=(socketio,))
    thread.daemon = True
    thread.start()