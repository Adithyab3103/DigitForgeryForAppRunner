import json
import base64
import io
import os
import tempfile
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variable to store the model (loaded once per container)
model = None

def load_model():
    """Load the ML model (cached globally)"""
    global model
    if model is None:
        try:
            # Try to load the model from the deployment package
            model_path = os.path.join(os.path.dirname(__file__), 'enhanced_mnist_forgery_final.keras')
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                logger.info("Model loaded successfully from local file")
            else:
                # If model not found locally, try to load from S3 or other sources
                logger.error(f"Model file not found at {model_path}")
                raise FileNotFoundError("Model file not found")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    return model

def preprocess_image(image_bytes, target_size=(28, 28)):
    """Preprocess the uploaded image for prediction"""
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Invert colors (MNIST has white digits on black background)
        image = cv2.bitwise_not(image)
        
        # Resize and normalize
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        image = image.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        return image.reshape(1, *target_size, 1)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise e

def generate_heatmap(model, img_array, last_conv_layer_name='conv2'):
    """Generate a heatmap showing which parts influenced the prediction"""
    try:
        # Create a model that maps the input to the activations of the last conv layer
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]]
        )
        
        # Compute gradient of the top predicted class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, tf.argmax(predictions[0])]
        
        # Get gradients and compute importance
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        logger.error(f"Error generating heatmap: {str(e)}")
        return None

def lambda_handler(event, context):
    """
    AWS Lambda handler for digit forgery recognition
    """
    try:
        # Set CORS headers
        headers = {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS'
        }
        
        # Handle preflight requests
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({'message': 'CORS preflight'})
            }
        
        # Handle different event types
        if 'httpMethod' in event:
            # API Gateway event
            return handle_api_gateway(event, context, headers)
        else:
            # Direct Lambda invocation
            return handle_direct_invocation(event, context, headers)
            
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def handle_api_gateway(event, context, headers):
    """Handle API Gateway requests"""
    if event['httpMethod'] == 'GET':
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html',
                'Access-Control-Allow-Origin': '*'
            },
            'body': get_simple_html()
        }
    elif event['httpMethod'] == 'POST':
        try:
            # Parse the request body
            if event.get('isBase64Encoded', False):
                body = base64.b64decode(event['body']).decode('utf-8')
            else:
                body = event['body']
            
            request_data = json.loads(body)
            
            # Extract image data
            if 'image' in request_data:
                # Image is base64 encoded
                image_data = base64.b64decode(request_data['image'])
            else:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({'error': 'No image data provided'})
                }
            
            # Process the image
            result = process_image(image_data)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(result)
            }
            
        except Exception as e:
            logger.error(f"API Gateway error: {str(e)}")
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': str(e)})
            }
    else:
        return {
            'statusCode': 405,
            'headers': headers,
            'body': json.dumps({'error': 'Method not allowed'})
        }

def handle_direct_invocation(event, context, headers):
    """Handle direct Lambda invocation"""
    try:
        # Check if image data is provided
        if 'image' in event:
            image_data = base64.b64decode(event['image'])
            result = process_image(image_data)
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(result)
            }
        else:
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'message': 'Digit Forgery Recognition API',
                    'usage': 'Send POST request with base64 encoded image in "image" field'
                })
            }
    except Exception as e:
        logger.error(f"Direct invocation error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        }

def process_image(image_data):
    """Process image and return predictions"""
    try:
        # Load model
        model = load_model()
        
        # Preprocess image
        img_array = preprocess_image(image_data)
        
        # Make predictions
        pred_digit_probs, pred_forgery_probs = model.predict(img_array, verbose=0)
        
        # Process predictions
        pred_digit = int(np.argmax(pred_digit_probs[0]))
        digit_confidence = float(np.max(pred_digit_probs[0])) * 100
        forgery_confidence = float(pred_forgery_probs[0][0])
        is_forged = forgery_confidence > 0.5
        
        # Generate heatmap (optional, as it's computationally expensive)
        heatmap = None
        try:
            heatmap = generate_heatmap(model, img_array)
            # Convert heatmap to base64 for transmission
            if heatmap is not None:
                heatmap_bytes = (heatmap * 255).astype(np.uint8)
                heatmap_pil = Image.fromarray(heatmap_bytes)
                heatmap_buffer = io.BytesIO()
                heatmap_pil.save(heatmap_buffer, format='PNG')
                heatmap_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
            else:
                heatmap_base64 = None
        except Exception as e:
            logger.warning(f"Could not generate heatmap: {str(e)}")
            heatmap_base64 = None
        
        # Prepare response
        result = {
            'success': True,
            'predictions': {
                'digit': pred_digit,
                'digit_confidence': round(digit_confidence, 2),
                'is_forged': is_forged,
                'forgery_confidence': round(forgery_confidence * 100, 2),
                'digit_probabilities': [round(float(prob) * 100, 2) for prob in pred_digit_probs[0]]
            },
            'heatmap': heatmap_base64,
            'analysis': {
                'forgery_detected': is_forged,
                'confidence_level': 'high' if digit_confidence > 80 else 'medium' if digit_confidence > 60 else 'low',
                'recommendations': get_recommendations(is_forged, digit_confidence, forgery_confidence)
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to process image'
        }

def get_recommendations(is_forged, digit_confidence, forgery_confidence):
    """Generate recommendations based on analysis"""
    recommendations = []
    
    if is_forged:
        recommendations.append("⚠️ Potential forgery detected - consider additional verification")
        if forgery_confidence > 0.8:
            recommendations.append("High confidence of forgery - manual review recommended")
    else:
        recommendations.append("✓ No signs of forgery detected")
    
    if digit_confidence < 60:
        recommendations.append("Low digit recognition confidence - image quality may be poor")
    elif digit_confidence > 90:
        recommendations.append("High confidence in digit recognition")
    
    return recommendations

def get_simple_html():
    """Return a simple HTML interface for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Digit Forgery Recognition API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .upload-area { 
                border: 2px dashed #007bff; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0;
                border-radius: 10px;
                background: #f8f9fa;
            }
            .btn { 
                background: #007bff; 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
            }
            .btn:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 20px; border-radius: 5px; }
            .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .loading { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Digit Forgery Recognition API</h1>
            <p>This API analyzes handwritten digits and detects potential forgeries using machine learning.</p>
            
            <div class="upload-area">
                <h3>Upload Image</h3>
                <p>Choose an image of a handwritten digit (PNG, JPG, JPEG)</p>
                <input type="file" id="imageInput" accept="image/*">
                <br><br>
                <button class="btn" onclick="analyzeImage()">Analyze Image</button>
            </div>
            
            <div id="result"></div>
            
            <script>
                function analyzeImage() {
                    const fileInput = document.getElementById('imageInput');
                    const file = fileInput.files[0];
                    
                    if (!file) {
                        alert('Please select an image first!');
                        return;
                    }
                    
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = '<div class="result loading">🔄 Analyzing image...</div>';
                    
                    // Convert image to base64
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const base64 = e.target.result.split(',')[1];
                        
                        // Send to Lambda function
                        fetch(window.location.href, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ image: base64 })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                const pred = data.predictions;
                                resultDiv.innerHTML = `
                                    <div class="result success">
                                        <h3>Analysis Results</h3>
                                        <p><strong>Predicted Digit:</strong> ${pred.digit}</p>
                                        <p><strong>Confidence:</strong> ${pred.digit_confidence}%</p>
                                        <p><strong>Forgery Detection:</strong> ${pred.is_forged ? '⚠️ Potential forgery detected' : '✓ No forgery detected'}</p>
                                        <p><strong>Forgery Confidence:</strong> ${pred.forgery_confidence}%</p>
                                        <h4>Recommendations:</h4>
                                        <ul>
                                            ${data.analysis.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                                        </ul>
                                    </div>
                                `;
                            } else {
                                resultDiv.innerHTML = `<div class="result error">Error: ${data.error || data.message}</div>`;
                            }
                        })
                        .catch(error => {
                            resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
                        });
                    };
                    reader.readAsDataURL(file);
                }
            </script>
        </div>
    </body>
    </html>
    """