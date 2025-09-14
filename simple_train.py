import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2
import random

def create_model():
    """Create a multi-output model for digit classification and forgery detection"""
    inputs = keras.Input(shape=(28, 28, 1))
    
    # Shared convolutional base
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    
    # Dual output: digit classification and forgery detection
    digit_output = layers.Dense(10, activation='softmax', name='digit_output')(x)
    forgery_output = layers.Dense(1, activation='sigmoid', name='forgery_output')(x)
    
    return keras.Model(
        inputs=inputs, 
        outputs=[digit_output, forgery_output], 
        name='enhanced_forgery_detection_model'
    )

def elastic_transform(image, alpha=34, sigma=4, random_state=None):
    """Elastic deformation of images as described in [Simard 2003]."""
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), 
        sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), 
        sigma, mode="constant", cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)

def add_stroke_variation(image, intensity=0.5):
    """Add natural-looking stroke variations"""
    if len(image.shape) == 3:
        image = image[:, :, 0]  # Convert to 2D if needed
    
    rows, cols = image.shape
    result = image.copy()
    
    # Add subtle waviness to strokes
    for i in range(rows):
        offset = int(intensity * np.sin(i/3.0))
        if 0 <= i+offset < rows:
            result[i] = np.roll(image[i], offset)
    
    return result.reshape(*image.shape, 1) if len(image.shape) == 3 else result

def cut_and_paste(image, other_image):
    """Cut a portion from another digit and paste it"""
    h, w = image.shape[:2]
    y1, y2 = np.random.randint(0, h//2), np.random.randint(h//2, h)
    x1, x2 = np.random.randint(0, w//2), np.random.randint(w//2, w)
    
    result = image.copy()
    result[y1:y2, x1:x2] = other_image[y1:y2, x1:x2]
    return result

def add_realistic_forgeries(x, y, num_forgeries=3000):
    """Add realistic forgeries to the dataset"""
    print(f"Generating {num_forgeries} realistic forgeries...")
    
    x_forged = []
    y_digit = []
    y_forgery = []
    
    # Create a mapping from digit to list of indices for faster lookup
    digit_to_indices = {}
    for i in range(10):
        digit_to_indices[i] = [j for j in range(len(x)) if np.argmax(y[j]) == i]
    
    for i in range(num_forgeries):
        try:
            # Choose a random digit to forge
            original_digit = np.random.randint(0, 10)
            
            # Get a random image of the original digit
            if not digit_to_indices[original_digit]:
                print(f"No samples found for digit {original_digit}, skipping...")
                continue
                
            original_idx = np.random.choice(digit_to_indices[original_digit])
            original_img = x[original_idx].copy()
            
            # Choose a random forgery type
            forgery_type = np.random.choice([
                'elastic', 'stroke', 'cut_paste', 'morph', 'noise'
            ])
            
            try:
                if forgery_type == 'elastic':
                    # Elastic deformation
                    alpha = np.random.uniform(30, 50)
                    sigma = np.random.uniform(4, 6)
                    img = elastic_transform(original_img, alpha=alpha, sigma=sigma)
                    
                elif forgery_type == 'stroke':
                    # Vary stroke width
                    img = add_stroke_variation(original_img)
                    
                elif forgery_type == 'cut_paste':
                    # Cut and paste from another digit
                    other_digit = np.random.choice([d for d in range(10) if d != original_digit])
                    if not digit_to_indices[other_digit]:
                        continue
                    other_idx = np.random.choice(digit_to_indices[other_digit])
                    other_img = x[other_idx].copy()
                    img = cut_and_paste(original_img, other_img)
                    
                elif forgery_type == 'morph':
                    # Morph with another digit
                    other_digit = np.random.choice([d for d in range(10) if d != original_digit])
                    if not digit_to_indices[other_digit]:
                        continue
                    other_idx = np.random.choice(digit_to_indices[other_digit])
                    other_img = x[other_idx].copy()
                    alpha = np.random.uniform(0.3, 0.7)
                    img = (1 - alpha) * original_img + alpha * other_img
                    
                else:  # noise
                    # Add realistic noise
                    noise = np.random.normal(0, 0.05, original_img.shape)
                    img = np.clip(original_img + noise, 0, 1)
                
                # Ensure the image is in the correct range and shape
                img = np.clip(img, 0, 1)
                if len(img.shape) == 2:
                    img = img[..., np.newaxis]  # Add channel dimension if missing
                
                x_forged.append(img)
                y_digit.append(y[original_idx])  # Original digit label
                y_forgery.append(1)  # Mark as forgery
                
                if (i + 1) % 100 == 0:
                    print(f"Generated {i+1}/{num_forgeries} forgeries")
                    
            except Exception as e:
                print(f"Error creating {forgery_type} forgery: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Error in forgery generation loop: {str(e)}")
            continue
    
    if not x_forged:  # If no forgeries were created successfully
        print("Warning: No forgeries were created successfully")
        return x, y, np.zeros(len(x))
    
    # Convert lists to numpy arrays
    x_forged = np.array(x_forged)
    y_digit = np.array(y_digit)
    y_forgery = np.array(y_forgery)
    
    # Combine with original data
    x_combined = np.concatenate([x, x_forged], axis=0)
    y_digit_combined = np.concatenate([y, y_digit], axis=0)
    y_forgery_combined = np.concatenate([np.zeros(len(x)), y_forgery], axis=0)
    
    # Shuffle the combined dataset
    indices = np.random.permutation(len(x_combined))
    return x_combined[indices], y_digit_combined[indices], y_forgery_combined[indices]

def main():
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    
    # Convert labels to one-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    # Add realistic forgeries to training and test sets
    print("Generating training forgeries...")
    x_train_aug, y_digit_train, y_forgery_train = add_realistic_forgeries(
        x_train, y_train_cat, num_forgeries=3000
    )
    
    print("Generating test forgeries...")
    x_test_aug, y_digit_test, y_forgery_test = add_realistic_forgeries(
        x_test, y_test_cat, num_forgeries=500
    )

    # Create and compile the model
    model = create_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'digit_output': 'categorical_crossentropy', 
            'forgery_output': 'binary_crossentropy'
        },
        metrics={
            'digit_output': 'accuracy', 
            'forgery_output': 'accuracy'
        },
        loss_weights={
            'digit_output': 1.0,
            'forgery_output': 0.5  # Adjust based on which task is more important
        }
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'enhanced_mnist_forgery.keras',
            save_best_only=True,
            monitor='val_digit_output_accuracy',
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_digit_output_accuracy',
            mode='max',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        x_train_aug,
        {
            'digit_output': y_digit_train,
            'forgery_output': y_forgery_train
        },
        batch_size=128,
        epochs=30,
        validation_data=(
            x_test_aug,
            {
                'digit_output': y_digit_test,
                'forgery_output': y_forgery_test
            }
        ),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save('enhanced_mnist_forgery_final.keras')
    print("Model saved as 'enhanced_mnist_forgery_final.keras'")

if __name__ == "__main__":
    print("Starting enhanced MNIST forgery detection training...")
    main()
