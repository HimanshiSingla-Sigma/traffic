import random
from datetime import datetime

def predict_accident_risk(intersection):
    """Predict accident risk based on various factors"""
    base_risk = 0.1
    
    # Factor 1: Vehicle count
    if intersection.vehicle_count > 50:
        base_risk += 0.3
    elif intersection.vehicle_count > 30:
        base_risk += 0.2
    elif intersection.vehicle_count > 15:
        base_risk += 0.1
    
    # Factor 2: Average speed
    if intersection.avg_speed > 80:  # km/h
        base_risk += 0.3
    elif intersection.avg_speed > 60:
        base_risk += 0.2
    elif intersection.avg_speed > 40:
        base_risk += 0.1
    
    # Factor 3: Time of day (rush hour)
    hour = datetime.now().hour
    if (7 <= hour <= 9) or (16 <= hour <= 19):
        base_risk += 0.2
    
    # Factor 4: Weather (simulated)
    # In a real system, this would come from weather API
    weather_factor = random.choice([0, 0.1, 0.3])  # 0: clear, 0.1: rain, 0.3: snow
    base_risk += weather_factor
    
    # Ensure risk is between 0 and 1
    return min(1.0, max(0.0, base_risk))

def should_trigger_intervention(risk_score, intersection_data):
    """Determine if an intervention should be triggered"""
    if risk_score > 0.7:
        return "emergency"
    elif risk_score > 0.5:
        return "moderate"
    elif risk_score > 0.3:
        return "advisory"
    return None