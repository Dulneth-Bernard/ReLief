"""
ReLief Dermoscopic Analysis Platform - Backend API
Main Flask Application Entry Point
"""
from flask import Flask, send_from_directory
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STATIC_DIR, UPLOADS_DIR, DEBUG, PORT
from routes import api


def create_app():
    """Application factory"""
    app = Flask(__name__, static_folder=STATIC_DIR)
    
    # Enable CORS for frontend
    CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])
    
    # Register blueprints
    app.register_blueprint(api)
    

    # Serve static files (heatmaps, etc.)
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        return send_from_directory(STATIC_DIR, filename)

    # Serve uploaded images
    @app.route('/uploads/<path:filename>')
    def serve_uploads(filename):
        return send_from_directory(UPLOADS_DIR, filename)
    
    # Root endpoint
    @app.route('/')
    def index():
        return {
            "name": "ReLief Dermoscopic Analysis API",
            "version": "1.0.0",
            "endpoints": {
                "classification_models": "/api/classification-models",
                "diffusion_models": "/api/diffusion-models",
                "images": "/api/images",
                "upload": "/api/images/upload",
                "predict": "/api/predict",
                "explain": "/api/explain",
                "classes": "/api/classes",
                "health": "/api/health"
            }
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == '__main__':
    print(f"""
+-----------------------------------------------------------+
|          ReLief Dermoscopic Analysis Platform             |
|                     Backend API                           |
+-----------------------------------------------------------+
|  Running on: http://localhost:{PORT}                        |
|  API Docs:   http://localhost:{PORT}/api/health             |
|  Debug Mode: {DEBUG}                                      |
+-----------------------------------------------------------+
    """)
    app.run(debug=DEBUG, port=PORT, host='0.0.0.0')
