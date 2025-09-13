from app import db
from datetime import datetime

class Intersection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    lat = db.Column(db.Float, nullable=False)
    lng = db.Column(db.Float, nullable=False)
    vehicle_count = db.Column(db.Integer, default=0)
    avg_speed = db.Column(db.Float, default=0)
    congestion_level = db.Column(db.String(20), default='low')
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class TrafficData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    intersection_id = db.Column(db.Integer, db.ForeignKey('intersection.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    vehicle_count = db.Column(db.Integer)
    avg_speed = db.Column(db.Float)
    weather_condition = db.Column(db.String(50))
    road_condition = db.Column(db.String(50))

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(500), nullable=False)
    severity = db.Column(db.String(20), nullable=False)
    location = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    acknowledged = db.Column(db.Boolean, default=False)