from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO

db = SQLAlchemy()
socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions with app
    db.init_app(app)
    socketio.init_app(app)
    
    # Import and register blueprints, models, etc.
    with app.app_context():
        from . import routes
        from .models import traffic_model
        
        # Create database tables
        db.create_all()
        
        # Start background simulation
        from .utils.data_processor import start_simulation
        start_simulation(socketio)
        
    return app