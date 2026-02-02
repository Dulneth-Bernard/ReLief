"""
API Routes for the ReLief Platform
"""
from flask import Blueprint, request, jsonify, current_app
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services import inference_service, explainability_service, image_service
from config import CLASS_NAMES, CLASS_FULL_NAMES


# Create blueprint
api = Blueprint('api', __name__, url_prefix='/api')


# ============== Classification Models ==============

@api.route('/classification-models', methods=['GET'])
def get_classification_models():
    """List available classification models with summaries"""
    try:
        models = inference_service.get_available_models()
        return jsonify({
            "success": True,
            "data": models,
            "count": len(models)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api.route('/classification-models/<model_id>', methods=['GET'])
def get_classification_model(model_id):
    """Get specific classification model details"""
    try:
        models = inference_service.get_available_models()
        model = next((m for m in models if m['id'] == model_id), None)
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        return jsonify({"success": True, "data": model})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Diffusion Models ==============

@api.route('/diffusion-models', methods=['GET'])
def get_diffusion_models():
    """List available diffusion models"""
    try:
        models = image_service.get_diffusion_models()
        return jsonify({
            "success": True,
            "data": models,
            "count": len(models)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api.route('/diffusion-models/<model_id>', methods=['GET'])
def get_diffusion_model(model_id):
    """Get specific diffusion model details"""
    try:
        model = image_service.get_diffusion_model(model_id)
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        return jsonify({"success": True, "data": model.to_dict()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Image Upload & Browse ==============

@api.route('/images', methods=['GET'])
def get_images():
    """List uploaded images"""
    try:
        images = image_service.get_uploaded_images()
        return jsonify({
            "success": True,
            "data": [img.to_dict() for img in images],
            "count": len(images)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api.route('/images/upload', methods=['POST'])
def upload_image():
    """Upload a dermoscopic image"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400
        
        lesion_image = image_service.upload_image(file)
        return jsonify({
            "success": True,
            "data": lesion_image.to_dict(),
            "message": "Image uploaded successfully"
        }), 201
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Prediction ==============

@api.route('/predict', methods=['POST'])
def predict():
    """Run classification prediction on an image"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        image_id = data.get('image_id')
        model_id = data.get('model_id')
        
        if not image_id or not model_id:
            return jsonify({
                "success": False, 
                "error": "Both image_id and model_id are required"
            }), 400
        
        # Get image path
        image_path = image_service.get_image_path(image_id)
        if not image_path:
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        # Run prediction
        result = inference_service.predict(image_path, model_id)
        
        # Add class full names to response
        response_data = result.to_dict()
        response_data['predicted_class_name'] = CLASS_FULL_NAMES.get(
            result.predicted_class, result.predicted_class
        )
        
        return jsonify({
            "success": True,
            "data": response_data
        })
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Explainability ==============

@api.route('/explain', methods=['POST'])
def explain():
    """Generate Grad-CAM explanation for a prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        image_id = data.get('image_id')
        model_id = data.get('model_id')
        target_class = data.get('target_class')  # Optional
        
        if not image_id or not model_id:
            return jsonify({
                "success": False, 
                "error": "Both image_id and model_id are required"
            }), 400
        
        # Get image path
        image_path = image_service.get_image_path(image_id)
        if not image_path:
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        # Generate Grad-CAM
        result = explainability_service.generate_gradcam(
            image_path, model_id, target_class
        )
        
        return jsonify({
            "success": True,
            "data": result.to_dict()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Classes Info ==============

@api.route('/classes', methods=['GET'])
def get_classes():
    """Get list of lesion classes"""
    return jsonify({
        "success": True,
        "data": [
            {"code": code, "name": CLASS_FULL_NAMES[code], "index": i}
            for i, code in enumerate(CLASS_NAMES)
        ]
    })


# ============== Health Check ==============

@api.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        "success": True,
        "message": "ReLief API is running",
        "device": str(inference_service.device)
    })
