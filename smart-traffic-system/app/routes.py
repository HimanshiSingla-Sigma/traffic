from flask import render_template, jsonify, request
from app import create_app, db, socketio
from .models.traffic_model import TrafficData, Intersection, Alert
from .utils.predictors import predict_accident_risk
import json
from datetime import datetime

app = create_app()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/map')
def traffic_map():
    return render_template('map.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/api/traffic-data')
def get_traffic_data():
    # Get real-time traffic data
    intersections = Intersection.query.all()
    data = []
    for intersection in intersections:
        data.append({
            'id': intersection.id,
            'name': intersection.name,
            'location': [intersection.lat, intersection.lng],
            'vehicle_count': intersection.vehicle_count,
            'avg_speed': intersection.avg_speed,
            'congestion_level': intersection.congestion_level,
            'risk_score': predict_accident_risk(intersection)
        })
    return jsonify(data)

@app.route('/api/alerts')
def get_alerts():
    alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(10).all()
    result = []
    for alert in alerts:
        result.append({
            'id': alert.id,
            'message': alert.message,
            'severity': alert.severity,
            'location': alert.location,
            'timestamp': alert.timestamp.isoformat()
        })
    return jsonify(result)

@app.route('/api/intervene', methods=['POST'])
def trigger_intervention():
    data = request.json
    intersection_id = data.get('intersection_id')
    intervention_type = data.get('type')
    
    # Logic to trigger intervention (simulated)
    # In a real system, this would communicate with traffic signals
    socketio.emit('intervention', {
        'intersection_id': intersection_id,
        'type': intervention_type,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    # Log the intervention
    alert = Alert(
        message=f"Intervention triggered: {intervention_type}",
        severity="info",
        location=intersection_id
    )
    db.session.add(alert)
    db.session.commit()
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    socketio.run(app)