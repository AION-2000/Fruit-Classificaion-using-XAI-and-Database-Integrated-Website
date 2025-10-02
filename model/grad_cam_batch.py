import os
import numpy as np
import tensorflow as tf
import cv2
import json
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grad_cam_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load model and class labels
def load_model_and_labels(model_path, labels_path):
    """Load the model and class labels"""
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Load class labels
        with open(labels_path, "r") as f:
            labels = json.load(f)
        
        # If labels is a dictionary, convert to list of values
        if isinstance(labels, dict):
            labels = list(labels.values())
        elif not isinstance(labels, list):
            logger.warning(f"Labels is not a list or dict. Got {type(labels)}")
            return None, None
            
        logger.info(f"Class labels loaded: {len(labels)} classes")
        return model, labels
    except Exception as e:
        logger.error(f"Error loading model or labels: {e}")
        return None, None

def preprocess_image(img_path, target_size=(96, 96)):
    """Preprocess image for model input"""
    try:
        # Read image
        img = Image.open(img_path)
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(img)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        
        # Convert back to PIL Image
        img = Image.fromarray(denoised)
        
        # Resize to target size
        img = img.resize(target_size)
        
        # Convert to array and add batch dimension
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply MobileNetV2 preprocessing
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array, img
    except Exception as e:
        logger.error(f"Error preprocessing image {img_path}: {e}")
        return None, None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """Generate Grad-CAM heatmap"""
    try:
        # Find a suitable conv layer if not specified
        if last_conv_layer_name is None:
            # Instead of the very last conv layer, try to find a more suitable one
            conv_layers = []
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    conv_layers.append(layer.name)
            
            # If we have multiple conv layers, pick one that's not too small
            if len(conv_layers) > 1:
                # Try to pick a middle layer, not the first or last
                last_conv_layer_name = conv_layers[len(conv_layers)//2]
            elif len(conv_layers) == 1:
                last_conv_layer_name = conv_layers[0]
            else:
                logger.error("No convolutional layer found in the model")
                return None
                
            logger.info(f"Using conv layer: {last_conv_layer_name}")
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Handle different output formats
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
                pred_index = int(pred_index)
            
            # Get the class channel
            if len(predictions.shape) == 1:
                class_channel = predictions[pred_index]
            else:
                class_channel = predictions[0, pred_index]
            
            # Watch the tensor
            tape.watch(conv_outputs)
            
            # Calculate gradients
            grads = tape.gradient(class_channel, conv_outputs)
            
            if grads is None:
                logger.error("Gradients are None")
                return None
            
            # Pool the gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Get the convolutional outputs
            conv_outputs = conv_outputs[0]
            
            # Calculate the heatmap
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize the heatmap
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
            
            return heatmap.numpy()
    
    except Exception as e:
        logger.error(f"Error in make_gradcam_heatmap: {e}")
        return None

def generate_gradcam_visualizations(img_path, model, class_labels, output_dir):
    """Generate and save Grad-CAM visualizations for a single image"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filename without extension
        filename = Path(img_path).stem
        
        # Preprocess image
        img_array, original_img = preprocess_image(img_path)
        if img_array is None:
            return False
        
        # Get prediction
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        
        # Check if the prediction index is valid
        if pred_index >= len(class_labels):
            logger.error(f"Prediction index {pred_index} is out of bounds for class_labels of length {len(class_labels)}")
            return False
        
        pred_label = class_labels[pred_index]
        confidence = float(preds[0][pred_index])
        
        logger.info(f"Image: {img_path} - Predicted: {pred_label} ({confidence:.2f})")
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, pred_index=pred_index)
        if heatmap is None:
            return False
        
        # Debug: Log heatmap shape and values
        logger.info(f"Heatmap shape: {heatmap.shape}, min: {np.min(heatmap)}, max: {np.max(heatmap)}")
        
        # Convert original image to numpy array
        original_img_array = np.array(original_img)
        logger.info(f"Original image shape: {original_img_array.shape}")
        
        # Check if heatmap has valid dimensions
        if heatmap.size == 0:
            logger.error("Heatmap has zero size")
            return False
        
        # Check if heatmap dimensions are valid
        if len(heatmap.shape) != 2:
            logger.error(f"Heatmap has invalid shape: {heatmap.shape}")
            return False
        
        # Try to resize heatmap to match original image
        try:
            # Ensure heatmap is float32 and in range [0, 1]
            heatmap = np.float32(heatmap)
            
            # Use dsize parameter format (width, height)
            dsize = (original_img_array.shape[1], original_img_array.shape[0])
            logger.info(f"Resizing heatmap from {heatmap.shape} to {dsize}")
            
            # Try different interpolation methods
            try:
                # First try with INTER_NEAREST for small heatmaps
                heatmap_resized = cv2.resize(heatmap, dsize, interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                logger.error(f"INTER_NEAREST failed: {e}")
                try:
                    # Fallback to INTER_LINEAR
                    heatmap_resized = cv2.resize(heatmap, dsize, interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    logger.error(f"INTER_LINEAR failed: {e}")
                    # Last resort: use PIL resize
                    try:
                        # Convert heatmap to PIL Image, resize, then convert back
                        heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
                        heatmap_resized = heatmap_img.resize(dsize, Image.BILINEAR)
                        heatmap_resized = np.array(heatmap_resized) / 255.0
                    except Exception as e:
                        logger.error(f"PIL resize also failed: {e}")
                        return False
            
            # Ensure heatmap is still in valid range
            heatmap_resized = np.clip(heatmap_resized, 0, 1)
            
        except Exception as e:
            logger.error(f"Error resizing heatmap: {e}")
            return False
        
        # Convert to uint8 and apply colormap
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Ensure both arrays are uint8
        if original_img_array.dtype != np.uint8:
            logger.info(f"Converting original_img_array from {original_img_array.dtype} to uint8")
            original_img_array = np.uint8(original_img_array)
        
        # Create overlay
        try:
            superimposed_img = cv2.addWeighted(
                original_img_array, 0.6, 
                heatmap_colored, 0.4, 0, 
                dtype=cv2.CV_8U
            )
        except Exception as e:
            logger.error(f"Error in addWeighted: {e}")
            try:
                superimposed_img = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
            except Exception as e2:
                logger.error(f"Fallback addWeighted also failed: {e2}")
                # Last resort: just use the heatmap
                superimposed_img = heatmap_colored.copy()
        
        # Save visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original image
        original_path = os.path.join(output_dir, f"{filename}_original_{timestamp}.jpg")
        original_img.save(original_path)
        
        # Save heatmap
        heatmap_path = os.path.join(output_dir, f"{filename}_heatmap_{timestamp}.jpg")
        Image.fromarray(heatmap_colored).save(heatmap_path)
        
        # Save overlay
        overlay_path = os.path.join(output_dir, f"{filename}_overlay_{timestamp}.jpg")
        Image.fromarray(superimposed_img).save(overlay_path)
        
        # Save prediction info
        info_path = os.path.join(output_dir, f"{filename}_info_{timestamp}.txt")
        with open(info_path, "w") as f:
            f.write(f"Image: {img_path}\n")
            f.write(f"Predicted class: {pred_label}\n")
            f.write(f"Confidence: {confidence:.4f}\n")
            f.write(f"Class index: {pred_index}\n")
        
        logger.info(f"Saved Grad-CAM visualizations for {filename} to {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating Grad-CAM for {img_path}: {e}")
        return False

def process_directory(input_dir, model, class_labels, output_dir):
    """Process all images in a directory"""
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Get all image files in the directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} image files to process")
    
    # Process each image
    success_count = 0
    for img_path in image_files:
        if generate_gradcam_visualizations(str(img_path), model, class_labels, output_dir):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(image_files)} images")

def main():
    parser = argparse.ArgumentParser(description="Batch generate Grad-CAM visualizations")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output visualizations")
    parser.add_argument("--model_path", type=str, default="optimized_fruit_model.keras", help="Path to the model file")
    parser.add_argument("--labels_path", type=str, default="enhanced_class_labels.json", help="Path to the class labels file")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Load model and labels
    model, class_labels = load_model_and_labels(args.model_path, args.labels_path)
    if model is None or class_labels is None:
        logger.error("Failed to load model or labels")
        return
    
    # Process the directory
    process_directory(args.input_dir, model, class_labels, args.output_dir)

if __name__ == "__main__":
    main()