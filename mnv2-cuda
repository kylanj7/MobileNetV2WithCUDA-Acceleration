import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Check CUDA availability
print("CUDA available:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

# Model creation and loading (same as before)
def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Assume model is trained and saved
gpu_classes = ['GTX 1080', 'RTX 2080', 'RTX 3080', 'RTX 3090']  # Add your GPU classes here
model = create_model(num_classes=len(gpu_classes))
model.load_weights('path_to_your_trained_weights.h5')

# CUDA-accelerated image processing
def preprocess_image_cuda(image_path):
    # Read image
    img = cv2.imread(image_path)
    
    # Upload to GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    
    # Resize
    gpu_resized = cv2.cuda.resize(gpu_img, (224, 224))
    
    # Convert to RGB (OpenCV uses BGR by default)
    gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
    
    # Download to CPU for further processing
    rgb_img = gpu_rgb.download()
    
    # Normalize and expand dimensions
    x = np.expand_dims(rgb_img, axis=0)
    x = preprocess_input(x)
    
    return x, img

# Inference function
def detect_gpu(image_path, model, gpu_classes):
    x, original_img = preprocess_image_cuda(image_path)
    
    predictions = model.predict(x)
    gpu_class = gpu_classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    return gpu_class, confidence, original_img

# Example usage
image_path = 'path_to_your_gpu_image.jpg'
detected_gpu, confidence, img = detect_gpu(image_path, model, gpu_classes)

print(f"Detected GPU: {detected_gpu}")
print(f"Confidence: {confidence:.2f}")

# Display the image with detection result (using CUDA)
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(img)

# Add text to image (on CPU as CUDA doesn't support drawing operations)
img = gpu_img.download()
cv2.putText(img, f"{detected_gpu}: {confidence:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display
cv2.imshow('Detected GPU', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
