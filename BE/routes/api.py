"""
API Routes for the ReLief Platform
"""
from flask import Blueprint, request, jsonify, current_app
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services import inference_service, explainability_service, image_service, db_service
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


@api.route('/sample-images', methods=['GET'])
def get_sample_images():
    """List sample test images available on the server"""
    try:
        from config import STATIC_DIR
        sample_dir = os.path.join(STATIC_DIR, 'sample-images')
        os.makedirs(sample_dir, exist_ok=True)
        
        images = []
        for filename in os.listdir(sample_dir):
            if filename.split('.')[-1].lower() in {'png', 'jpg', 'jpeg'}:
                images.append({
                    "id": filename.split('.')[0], 
                    "filename": filename,
                    "url": f"/static/sample-images/{filename}",
                    "source": "sample"
                })
                
        return jsonify({
            "success": True,
            "data": images,
            "count": len(images)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api.route('/synthetic-images', methods=['GET'])
def get_synthetic_images():
    """Fetch paginated synthetic images from Cloudinary"""
    import cloudinary
    import cloudinary.api
    import cloudinary.search
    
    # Optional manual config if url is not present
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    api_key = os.getenv('CLOUDINARY_API_KEY')
    api_secret = os.getenv('CLOUDINARY_API_SECRET')
    
    if cloud_name and api_key and api_secret:
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret
        )
    
    try:
        folder = request.args.get('folder', 'synthetic') # The folder name to filter by
        next_cursor = request.args.get('next_cursor', None)
        max_results = int(request.args.get('max_results', 20))
        
        # Build search expression
        expression = ""
        if folder and folder.lower() != 'all':
            expression = f'folder="{folder}"'
            
        search = cloudinary.search.Search()\
            .expression(expression)\
            .sort_by('created_at', 'desc')\
            .max_results(max_results)
            
        if next_cursor:
            search = search.next_cursor(next_cursor)
            
        result = search.execute()
        
        images = []
        for resource in result.get('resources', []):
            images.append({
                'id': resource.get('asset_id'),
                'public_id': resource.get('public_id'),
                'url': resource.get('secure_url'),
                'format': resource.get('format'),
                'created_at': resource.get('created_at'),
                'folder': resource.get('folder')
            })
            
        return jsonify({
            "success": True,
            "data": images,
            "next_cursor": result.get('next_cursor'),
            "total_count": result.get('total_count', 0)
        })
    except Exception as e:
        # Check if error is related to missing auth
        if "Must supply api_key" in str(e):
            return jsonify({
                "success": False,
                "error": "Cloudinary credentials missing. Please configure CLOUDINARY_URL in backend .env file."
            }), 401
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
        
        # Save to database
        db_service.save_image_upload(
            image_id=lesion_image.id,
            filename=lesion_image.filename,
            image_url=f"/uploads/{lesion_image.filename}"
        )
        
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
        
        # Save prediction results to DB
        db_service.save_prediction(
            image_id=image_id,
            model_id=model_id,
            prediction=response_data['predicted_class_name'],
            confidence=result.confidence,
            probabilities=result.class_probabilities
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
        method = data.get('method', 'gradcam') # Optional, defaults to gradcam
        
        if not image_id or not model_id:
            return jsonify({
                "success": False, 
                "error": "Both image_id and model_id are required"
            }), 400
        
        # Get image path
        image_path = image_service.get_image_path(image_id)
        if not image_path:
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        # Generate Explanation (Grad-CAM or Grad-CAM++)
        result = explainability_service.generate_gradcam(
            image_path, model_id, target_class, method
        )
        
        result_dict = result.to_dict()
        
        # Save map URL to DB
        db_service.update_gradcam(
            image_id=image_id,
            gradcam_url=result_dict.get('overlay_path')
        )
        
        return jsonify({
            "success": True,
            "data": result_dict
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== History / DB ==============

@api.route('/history', methods=['GET'])
def get_prediction_history():
    """Get history of uploaded images and predictions"""
    try:
        history = db_service.get_history()
        return jsonify({
            "success": True,
            "data": history,
            "count": len(history)
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
