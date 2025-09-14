import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras

def preprocess_image(img_path, target_size=(28, 28)):
    """
    Load and preprocess an image for prediction
    
    Args:
        img_path: Path to the image file
        target_size: Target size for resizing (height, width)
        
    Returns:
        Tuple of (preprocessed image array, original image, processed image for display)
    """
    try:
        # Read image in grayscale
        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise ValueError("Could not read the image. Please check the file path.")
        
        # Store a copy for display
        display_img = original_img.copy()
        
        # Invert colors (MNIST has white digits on black background)
        img = cv2.bitwise_not(original_img)
        
        # Resize and normalize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        img = img.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        img_array = img.reshape(1, *target_size, 1)
        
        return img_array, original_img, img
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None, None

def visualize_results(original_img, processed_img, pred_digit, digit_confidence, 
                    is_forged, forgery_confidence, heatmap=None):
    """
    Visualize the original image, processed image, and predictions
    
    Args:
        original_img: The original input image
        processed_img: The preprocessed image
        pred_digit: Predicted digit (0-9)
        digit_confidence: Confidence score for digit prediction
        is_forged: Boolean indicating if forgery was detected
        forgery_confidence: Confidence score for forgery detection
        heatmap: Optional heatmap for visualization
    """
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot processed image with prediction
    plt.subplot(1, 3, 2)
    plt.imshow(processed_img, cmap='gray')
    title = f"Predicted: {pred_digit} ({digit_confidence:.1f}%)"
    title += f"\nForgery: {'YES' if is_forged else 'NO'} ({forgery_confidence*100:.1f}%)"
    plt.title(title, fontsize=10)
    plt.axis('off')
    
    # Plot heatmap if available
    if heatmap is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.colorbar()
        plt.title("Model Attention Heatmap")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_heatmap(model, img_array, last_conv_layer_name='conv2'):
    """
    Generate a heatmap showing which parts of the image influenced the prediction
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed input image
        last_conv_layer_name: Name of the last convolutional layer
        
    Returns:
        Heatmap array
    """
    # Create a model that maps the input to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output[0]]
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

def main():
    # Load the trained model
    try:
        model = keras.models.load_model('enhanced_mnist_forgery.keras')
        print("Enhanced forgery detection model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please run 'simple_train.py' first to train the model.")
        return
    
    print("\nEnhanced Handwritten Digit Recognition with Forgery Detection")
    print("-" * 60)
    print("Instructions:")
    print("1. Save your handwritten digit image")
    print("2. Enter the full path to your image")
    print("3. Type 'q' to quit\n")
    
    while True:
        img_path = input("\nEnter image path (or 'q' to quit): ").strip('"')
        
        if img_path.lower() == 'q':
            break
            
        try:
            # Preprocess the image
            img_array, original_img, processed_img = preprocess_image(img_path)
            if img_array is None:
                continue
            
            # Make predictions
            pred_digit_probs, pred_forgery_probs = model.predict(img_array, verbose=0)
            
            # Process predictions
            pred_digit = np.argmax(pred_digit_probs[0])
            digit_confidence = np.max(pred_digit_probs[0]) * 100
            forgery_confidence = pred_forgery_probs[0][0]
            is_forged = forgery_confidence > 0.5
            
            # Generate heatmap for visualization
            heatmap = generate_heatmap(model, img_array)
            
            # Resize heatmap to match original image for better visualization
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            
            # Visualize results
            visualize_results(
                original_img, 
                processed_img, 
                pred_digit, 
                digit_confidence, 
                is_forged,  
                forgery_confidence,
                heatmap=heatmap_resized
            )
            
            # Print detailed results
            print("\n" + "="*50)
            print(f"Predicted Digit: {pred_digit}")
            print(f"Confidence: {digit_confidence:.1f}%")
            print("\nForgery Analysis:")
            print(f"- Detected: {'YES' if is_forged else 'NO'}")
            print(f"- Confidence: {forgery_confidence*100:.1f}%")
            print("="*50)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please check the image path and try again.")

if __name__ == "__main__":
    import tensorflow as tf
    main()
