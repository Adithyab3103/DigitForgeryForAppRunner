import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import io
import os
print("Streamlit will listen on PORT:", os.environ.get("PORT"))


def preprocess_image(image, target_size=(28, 28)):
    """Preprocess the uploaded image for prediction"""
    try:
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
        return image.reshape(1, *target_size, 1), image
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def generate_heatmap(model, img_array, last_conv_layer_name='conv2'):
    """Generate a heatmap showing which parts influenced the prediction"""
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

def plot_heatmap(original_img, heatmap, alpha=0.4):
    """Plot heatmap on top of the original image"""
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert to uint8 and apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to RGB if grayscale
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

def main():
    # Page configuration
    st.set_page_config(
        page_title="Enhanced Handwriting Analysis",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        color: #42A5F5;
        margin-top: 1.5rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .forgery-detected {
        color: #E53935;
        font-weight: bold;
    }
    .forgery-not-detected {
        color: #43A047;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.markdown('<h1 class="main-title">Enhanced Handwriting Analysis</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application uses deep learning to recognize handwritten digits and detect potential forgeries.
    Upload an image of a handwritten digit, and the model will predict the digit and analyze it for signs of forgery.
    """)
    
    # Sidebar with model info and settings
    st.sidebar.title("Settings")
    st.sidebar.markdown("### Model Information")
    st.sidebar.info("""
    - **Model**: Custom CNN with dual output
    - **Training Data**: MNIST + Synthetic Forgeries
    - **Features**: 
      - Digit Recognition (0-9)
      - Forgery Detection
      - Attention Heatmap
    """)
    
    # File uploader
    st.markdown("## Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a handwritten digit (PNG, JPG, JPEG)", 
        type=["png", "jpg", "jpeg"]
    )
    
    # Load model with caching
    @st.cache_resource
    def load_model():
        try:
            model = keras.models.load_model('enhanced_mnist_forgery.keras')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.error("Please make sure 'enhanced_mnist_forgery.keras' exists in the directory.")
            return None
    
    model = load_model()
    
    if uploaded_file is not None and model is not None:
        try:
            # Read and preprocess the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Preprocess for model
            img_array, processed_img = preprocess_image(original_img)
            
            if img_array is not None:
                # Make predictions
                with st.spinner('Analyzing the image...'):
                    pred_digit_probs, pred_forgery_probs = model.predict(img_array, verbose=0)
                    
                    # Process predictions
                    pred_digit = np.argmax(pred_digit_probs[0])
                    digit_confidence = np.max(pred_digit_probs[0]) * 100
                    forgery_confidence = float(pred_forgery_probs[0][0])
                    is_forged = forgery_confidence > 0.5
                    
                    # Generate heatmap
                    heatmap = generate_heatmap(model, img_array)
                    
                    # Create visualization
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    ax1.imshow(original_img)
                    ax1.set_title("Original Image")
                    ax1.axis('off')
                    
                    # Processed image with prediction
                    ax2.imshow(processed_img, cmap='gray')
                    ax2.set_title(f"Predicted: {pred_digit} ({digit_confidence:.1f}%)")
                    ax2.axis('off')
                    
                    # Heatmap
                    superimposed_img = plot_heatmap(original_img, heatmap)
                    ax3.imshow(superimposed_img)
                    ax3.set_title("Attention Heatmap")
                    ax3.axis('off')
                    
                    plt.tight_layout()
                    
                # Display results
                st.markdown("## Analysis Results")
                st.pyplot(fig)
                
                # Detailed predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Digit Recognition")
                    st.markdown(f"**Predicted Digit:** {pred_digit}")
                    st.markdown(f"**Confidence:** {digit_confidence:.1f}%")
                    
                    # Display confidence scores for all digits
                    st.markdown("#### All Digit Probabilities")
                    digit_probs = pred_digit_probs[0]
                    for i, prob in enumerate(digit_probs):
                        st.progress(
                            float(prob), 
                            text=f"{i}: {prob*100:.1f}%"
                        )
                
                with col2:
                    st.markdown("### Forgery Detection")
                    if is_forged:
                        st.markdown(
                            f"<p class='forgery-detected'>⚠️ Potential forgery detected! "
                            f"({forgery_confidence*100:.1f}% confidence)</p>", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<p class='forgery-not-detected'>✓ No signs of forgery detected "
                            f"({(1-forgery_confidence)*100:.1f}% confidence)</p>", 
                            unsafe_allow_html=True
                        )
                    
                    # Forgery analysis
                    st.markdown("#### Analysis")
                    if is_forged:
                        st.warning("""
                        The model detected potential signs of forgery, which could be due to:
                        - Inconsistent stroke patterns
                        - Unnatural variations in the digit
                        - Signs of digital manipulation
                        - Unusual writing characteristics
                        """)
                    else:
                        st.success("""
                        The digit appears to be naturally written with consistent 
                        stroke patterns and no obvious signs of manipulation.
                        """)
                
                # Add some space
                st.markdown("---")
                st.markdown("### How does it work?")
                st.markdown("""
                - **Digit Recognition**: The model analyzes the input image to identify the written digit (0-9).
                - **Forgery Detection**: The model looks for inconsistencies and anomalies that might indicate manipulation.
                - **Attention Heatmap**: Shows which parts of the image were most important for the model's prediction.
                
                The model was trained on both genuine digits and various types of synthetic forgeries 
                to learn the differences between authentic and manipulated writing.
                """)
                
        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
            st.error("Please try with a different image or check the file format.")
    
    # Add some instructions when no file is uploaded
    else:
        st.markdown("### How to use:")
        st.markdown("""
        1. Click 'Browse files' to upload an image of a handwritten digit
        2. The model will analyze the image and provide predictions
        3. Review the digit recognition and forgery detection results
        4. Examine the attention heatmap to understand the model's focus
        
        For best results, use clear images of single digits on a light background.
        """)
        
        # Add example images
        st.markdown("### Example Images")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://github.com/plotly/datasets/raw/master/digit-embedding/0.png", 
                   caption="Example: 0", width=100)
        with col2:
            st.image("https://github.com/plotly/datasets/raw/master/digit-embedding/1.png", 
                   caption="Example: 1", width=100)
        with col3:
            st.image("https://github.com/plotly/datasets/raw/master/digit-embedding/2.png", 
                   caption="Example: 2", width=100)

if __name__ == "__main__":
    main()
