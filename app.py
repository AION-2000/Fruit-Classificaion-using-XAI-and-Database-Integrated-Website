from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import psutil
import uuid
import shutil
# Database imports
from database import add_fruit, add_xai_explanation, get_fruit_by_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app with explicit template and static folders
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Get the absolute path to the directory containing this script
basedir = os.path.abspath(os.path.dirname(__file__))

# Define paths for model and class labels
MODEL_PATH = os.path.join(basedir, "model", "optimized_fruit_model.keras")
CLASS_LABELS_PATH = os.path.join(basedir, "model", "enhanced_class_labels.json")

# Configuration
CONFIDENCE_THRESHOLD = 0.7  # Increased to 70% confidence threshold for fruit classification
NON_FRUIT_CLASSES = ["person", "human", "man", "woman", "child", "animal", "object", "car", "building", "tree", "flower"]  # Common non-fruit classes

# Updated load_class_labels function with better priority handling
def load_class_labels():
    """Load class labels from multiple possible locations, prioritizing the training output"""
    possible_paths = [
        os.path.join(basedir, "model", "enhanced_class_labels.json"),  # First priority
        os.path.join(basedir, "model", "optimized_class_labels.json"),  # Second priority
        os.path.join(basedir, "model", "class_labels.json"),
        os.path.join(basedir, "enhanced_class_labels.json"),
        os.path.join(basedir, "class_labels.json")
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    labels = json.load(f)
                # If labels is a dictionary, convert to list of values
                if isinstance(labels, dict):
                    labels = list(labels.values())
                elif not isinstance(labels, list):
                    logger.warning(f"Labels in {path} is not a list or dict. Got {type(labels)}")
                    continue
                logger.info(f"Class labels loaded from {path}: {len(labels)} classes")
                return labels
        except Exception as e:
            logger.warning(f"Failed to load class labels from {path}: {e}")
    
    # If no labels found, return empty list
    logger.warning("No class labels file found. Using empty list.")
    return []

# Load model and class labels
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    # Print model summary to debug layer names
    model.summary(print_fn=logger.info)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Load class labels
class_labels = load_class_labels()

# Check model and labels consistency
if model and class_labels:
    output_size = model.output_shape[1]
    if len(class_labels) != output_size:
        logger.error(f"CRITICAL: Class labels count ({len(class_labels)}) doesn't match model output size ({output_size})")
        logger.error("This will cause incorrect predictions. Please check your labels file.")
        # Instead of padding, we'll set class_labels to None to prevent incorrect predictions
        class_labels = None
    else:
        logger.info(f"Model and labels are consistent: {output_size} classes")

# Helper functions
def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting temporary file {file_path}: {e}")

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image with better error handling"""
    try:
        if not base64_string:
            return None
            
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
            
        img_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(img_data))
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        return None

def save_uploaded_file(file):
    """Save uploaded file to static/uploads and return the path"""
    try:
        # Use your existing uploads directory
        upload_dir = os.path.join(basedir, 'static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        upload_path = os.path.join(upload_dir, filename)
        file.save(upload_path)
        return upload_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise

def save_gradcam_images(original_img, heatmap_img, overlay_img):
    """Save Grad-CAM images to model/grad_cam_output and return their paths"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use your existing grad_cam_output directory in model folder
        gradcam_dir = os.path.join(basedir, 'model', 'grad_cam_output')
        os.makedirs(gradcam_dir, exist_ok=True)
        
        # Save images
        original_path = os.path.join(gradcam_dir, f"original_{timestamp}.jpg")
        heatmap_path = os.path.join(gradcam_dir, f"heatmap_{timestamp}.jpg")
        overlay_path = os.path.join(gradcam_dir, f"overlay_{timestamp}.jpg")
        
        original_img.save(original_path)
        heatmap_img.save(heatmap_path)
        overlay_img.save(overlay_path)
        
        return original_path, heatmap_path, overlay_path
    except Exception as e:
        logger.error(f"Error saving Grad-CAM images: {e}")
        raise

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
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
                logger.info(f"Using middle conv layer: {last_conv_layer_name}")
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

def preprocess_image(img_bytes, target_size=(96, 96)):  # Changed from (128, 128) to (96, 96)
    """Enhanced preprocessing with noise reduction - exactly matching training"""
    try:
        # Check if img_bytes is empty
        if not img_bytes or len(img_bytes) == 0:
            raise ValueError("Empty image data")
        
        # Convert bytes to image
        img = Image.open(BytesIO(img_bytes))
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(img)
        
        # Apply noise reduction (same as training)
        denoised = cv2.fastNlMeansDenoisingColored(img_array.astype(np.uint8), None, 10, 10, 7, 21)
        
        # Convert back to PIL Image
        img = Image.fromarray(denoised)
        
        # Resize to target size (96x96 to match model input)
        img = img.resize(target_size)
        
        # Convert to array and add batch dimension
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply MobileNetV2 preprocessing (same as training)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array, img
    except Exception as e:
        logger.error(f"Error in preprocess_image: {e}")
        raise

def create_pdf_report(original_image_base64, heatmap_image_base64, overlay_image_base64, prediction, confidence, top_predictions):
    """Generate a PDF report with classification results and visualizations"""
    temp_files = []  # Track temporary files for cleanup
    
    try:
        # Create a temporary file for the PDF
        temp_pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_files.append(temp_pdf_file.name)
        
        # Create the PDF document
        doc = SimpleDocTemplate(temp_pdf_file.name, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00f0ff'),
            alignment=1  # Center alignment
        )
        
        heading_style = ParagraphStyle(
            name='CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#00ff88')
        )
        
        normal_style = styles['Normal']
        
        # Build the story
        story = []
        
        # Title
        story.append(Paragraph("Neural Fruit Classifier Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Date and time
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Classification results
        story.append(Paragraph("Classification Results", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Create a table for results
        data = [
            ['Predicted Class', prediction],
            ['Confidence', f"{confidence:.2f}%"]
        ]
        
        t = Table(data, colWidths=[2*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Top predictions
        if top_predictions:
            story.append(Paragraph("Top Predictions", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            top_data = [['Class', 'Confidence']]
            for pred in top_predictions:
                top_data.append([pred['class'], f"{pred['confidence']*100:.2f}%"])
            
            top_table = Table(top_data, colWidths=[2*inch, 3*inch])
            top_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(top_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Visualizations
        story.append(Paragraph("Visualizations", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Function to add an image to the story
        def add_image_to_story(image_base64, title, width=4*inch, height=3*inch):
            if image_base64:
                story.append(Paragraph(title, normal_style))
                story.append(Spacer(1, 0.1*inch))
                
                # Convert base64 to PIL Image
                img = base64_to_image(image_base64)
                if img is None:
                    story.append(Paragraph("Image could not be loaded", normal_style))
                    return
                
                # Save to a temporary file
                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_files.append(temp_img.name)
                
                try:
                    img.save(temp_img.name, format="JPEG", quality=95)
                    logger.info(f"Temporary image saved to: {temp_img.name}")
                    
                    # Verify file size
                    file_size = os.path.getsize(temp_img.name)
                    logger.info(f"Image file size: {file_size} bytes")
                    
                    if file_size == 0:
                        logger.error("Image file is empty")
                        story.append(Paragraph("Image file is empty", normal_style))
                        return
                    
                    # Add to story
                    rl_image = RLImage(temp_img.name, width=width, height=height)
                    story.append(rl_image)
                    story.append(Spacer(1, 0.2*inch))
                    
                except Exception as e:
                    logger.error(f"Error adding image to PDF: {e}")
                    story.append(Paragraph(f"Error adding image: {str(e)}", normal_style))
        
        # Add images
        add_image_to_story(original_image_base64, "Original Image")
        add_image_to_story(heatmap_image_base64, "Grad-CAM Heatmap")
        add_image_to_story(overlay_image_base64, "Overlay Visualization")
        
        # Explanation
        story.append(Paragraph("About Grad-CAM", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        explanation_text = """
        Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique for making Convolutional Neural Network (CNN) 
        models more transparent by visualizing the regions of input that are important for predictions. The heatmap shows which 
        parts of the image the model focused on when making its classification decision. Red areas indicate high influence, 
        while blue areas indicate low influence.
        """
        
        story.append(Paragraph(explanation_text, normal_style))
        
        # Build PDF
        doc.build(story)
        
        # Verify PDF file size
        pdf_size = os.path.getsize(temp_pdf_file.name)
        logger.info(f"Generated PDF size: {pdf_size} bytes")
        
        if pdf_size == 0:
            logger.error("Generated PDF is empty")
            raise ValueError("Generated PDF is empty")
        
        return temp_pdf_file.name
    
    except Exception as e:
        logger.error(f"Error creating PDF report: {e}", exc_info=True)
        # Clean up temporary files before raising
        cleanup_temp_files(temp_files)
        raise

def generate_gradcam_for_report(fruit_id):
    """Generate Grad-CAM images for a fruit record and return base64 strings"""
    if not model:
        logger.error("Model not loaded")
        return None, None, None, None, None
    
    if not class_labels:
        logger.error("Class labels not loaded")
        return None, None, None, None, None
    
    # Get fruit record from database
    fruit_record = get_fruit_by_id(fruit_id)
    if not fruit_record:
        logger.error(f"Fruit record not found for id: {fruit_id}")
        return None, None, None, None, None
    
    # Try to get the image path from the record
    image_path = fruit_record[2] if len(fruit_record) > 2 else None
    
    # Check if the path exists
    if image_path and os.path.exists(image_path):
        logger.info(f"Found valid image path at index 2: {image_path}")
    else:
        # If no valid path found, try to find the image in the uploads directory
        filename = None
        for item in fruit_record:
            if isinstance(item, str) and '.' in item and not os.path.isabs(item):
                filename = item
                break
        
        if filename:
            # Construct the full path
            upload_dir = os.path.join(basedir, 'static', 'uploads')
            image_path = os.path.join(upload_dir, filename)
            logger.info(f"Constructed image path: {image_path}")
            
            if not os.path.exists(image_path):
                logger.error(f"Constructed image path does not exist: {image_path}")
                return None, None, None, None, None
        else:
            logger.error("Could not find image path in database record")
            return None, None, None, None, None
    
    try:
        # Read the image from the saved path
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        # Preprocess image with correct size (96x96)
        img_array, original_img = preprocess_image(img_bytes)
        
        # Get prediction
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        
        # Check if the prediction index is valid
        if pred_index >= len(class_labels):
            logger.error(f"Prediction index {pred_index} is out of bounds for class_labels of length {len(class_labels)}")
            return None, None, None, None, None
        
        pred_label = class_labels[pred_index]
        confidence = float(preds[0][pred_index])  # Convert numpy float32 to Python float
        
        logger.info(f"Predicted class: {pred_label}, confidence: {confidence}")
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=pred_index)
        
        if heatmap is None:
            logger.error("Heatmap generation failed")
            return None, None, None, None, None
        
        logger.info(f"Heatmap generated successfully. Shape: {heatmap.shape}")
        
        # Convert original image to numpy array
        original_img_array = np.array(original_img)
        logger.info(f"Original image array shape: {original_img_array.shape}, dtype: {original_img_array.dtype}")
        
        # Enhanced heatmap resizing with multiple fallback methods
        try:
            # Check if heatmap is valid
            if heatmap.size == 0 or len(heatmap.shape) != 2:
                logger.error(f"Invalid heatmap shape: {heatmap.shape}")
                return None, None, None, None, None
            
            # Ensure heatmap is float32
            heatmap = np.float32(heatmap)
            
            # Target size for resizing
            target_size = (original_img_array.shape[1], original_img_array.shape[0])
            logger.info(f"Resizing heatmap from {heatmap.shape} to {target_size}")
            
            # Try different interpolation methods
            try:
                # First try with INTER_NEAREST for small heatmaps
                heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_NEAREST)
                logger.info("Successfully resized with INTER_NEAREST")
            except Exception as e:
                logger.warning(f"INTER_NEAREST failed: {e}")
                try:
                    # Fallback to INTER_LINEAR
                    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
                    logger.info("Successfully resized with INTER_LINEAR")
                except Exception as e2:
                    logger.warning(f"INTER_LINEAR failed: {e2}")
                    try:
                        # Try INTER_CUBIC
                        heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)
                        logger.info("Successfully resized with INTER_CUBIC")
                    except Exception as e3:
                        logger.warning(f"INTER_CUBIC failed: {e3}")
                        # Last resort: use PIL resize
                        try:
                            # Convert heatmap to PIL Image, resize, then convert back
                            heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
                            heatmap_resized = heatmap_img.resize(target_size, Image.BILINEAR)
                            heatmap_resized = np.array(heatmap_resized) / 255.0
                            logger.info("Successfully resized with PIL")
                        except Exception as e4:
                            logger.error(f"All resize methods failed: {e4}")
                            return None, None, None, None, None
            
            # Ensure heatmap is still in valid range
            heatmap_resized = np.clip(heatmap_resized, 0, 1)
            
        except Exception as e:
            logger.error(f"Error resizing heatmap: {e}")
            return None, None, None, None, None
        
        # Convert to uint8 and apply colormap
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        logger.info(f"Heatmap colored shape: {heatmap_colored.shape}, dtype: {heatmap_colored.dtype}")
        
        # Ensure both arrays are uint8
        if original_img_array.dtype != np.uint8:
            logger.info(f"Converting original_img_array from {original_img_array.dtype} to uint8")
            original_img_array = np.uint8(original_img_array)
        
        # Create overlay with explicit output type
        try:
            superimposed_img = cv2.addWeighted(
                original_img_array, 0.6, 
                heatmap_colored, 0.4, 0, 
                dtype=cv2.CV_8U
            )
            logger.info(f"Superimposed image shape: {superimposed_img.shape}, dtype: {superimposed_img.dtype}")
        except Exception as e:
            logger.error(f"Error in addWeighted: {e}")
            # Fallback: try without explicit dtype
            try:
                superimposed_img = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
                logger.info("Superimposed image created without explicit dtype")
            except Exception as e2:
                logger.error(f"Fallback addWeighted also failed: {e2}")
                # Last resort: just use the heatmap
                superimposed_img = heatmap_colored.copy()
        
        # Convert images to base64
        def img_to_base64(img_array):
            try:
                img = Image.fromarray(img_array.astype('uint8'))
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{img_str}"
            except Exception as e:
                logger.error(f"Error converting image to base64: {e}")
                return None
        
        original_base64 = img_to_base64(original_img_array)
        heatmap_base64 = img_to_base64(heatmap_colored)
        overlay_base64 = img_to_base64(superimposed_img)
        
        # Check if any conversion failed
        if not original_base64 or not heatmap_base64 or not overlay_base64:
            return None, None, None, None, None
        
        return original_base64, heatmap_base64, overlay_base64, pred_label, confidence
    
    except Exception as e:
        logger.error(f"Error generating Grad-CAM for report: {e}", exc_info=True)
        return None, None, None, None, None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model/grad_cam_output/<filename>')
def serve_gradcam_image(filename):
    """Serve Grad-CAM images from model folder"""
    return send_from_directory(os.path.join(basedir, 'model', 'grad_cam_output'), filename)

@app.route('/classify', methods=['POST'])
def classify():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if not class_labels:
        return jsonify({'error': 'Class labels not loaded or inconsistent with model'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Check file type
    if not image_file.content_type.startswith('image/'):
        return jsonify({'error': 'File must be an image'}), 400
    
    # Check file size (limit to 10MB)
    image_file.seek(0, os.SEEK_END)
    file_length = image_file.tell()
    image_file.seek(0)  # Reset file pointer to beginning
    
    if file_length > 10 * 1024 * 1024:
        return jsonify({'error': 'File too large (max 10MB)'}), 400
    
    try:
        # Save the uploaded image
        image_path = save_uploaded_file(image_file)
        logger.info(f"Image saved to: {image_path}")
        
        # Reset file pointer again to read for processing
        image_file.seek(0)
        
        # Get image bytes
        img_bytes = image_file.read()
        
        # Preprocess image with correct size (96x96)
        img_array, original_img = preprocess_image(img_bytes)
        
        # Get prediction
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        
        # Check if the prediction index is valid
        if pred_index >= len(class_labels):
            logger.error(f"Prediction index {pred_index} is out of bounds for class_labels of length {len(class_labels)}")
            return jsonify({'error': f'Invalid prediction index: {pred_index}. The model might have more classes than the labels file.'}), 500
        
        pred_label = class_labels[pred_index]
        confidence = float(preds[0][pred_index])  # Convert numpy float32 to Python float
        
        # Get top predictions
        top_indices = np.argsort(preds[0])[-5:][::-1]
        top_predictions = []
        for i in top_indices:
            if i < len(class_labels):
                top_predictions.append({
                    'class': class_labels[i], 
                    'confidence': float(preds[0][i])
                })
            else:
                # Skip if index is out of bounds
                logger.warning(f"Top prediction index {i} is out of bounds for class_labels of length {len(class_labels)}")
        
        # Check if prediction is a known non-fruit class
        if pred_label.lower() in [non_fruit.lower() for non_fruit in NON_FRUIT_CLASSES]:
            logger.warning(f"Image classified as non-fruit class: {pred_label} with confidence: {confidence:.2f}")
            return jsonify({
                'error': f'This image is classified as "{pred_label}", which is not a fruit. Please upload an image of a fruit.',
                'confidence': confidence,
                'top_predictions': top_predictions,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }), 400
        
        # Check confidence threshold for fruit predictions
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"Low confidence prediction ({confidence:.2f} < {CONFIDENCE_THRESHOLD}): Image not classified as a fruit")
            return jsonify({
                'error': 'The image is not classified as a fruit with sufficient confidence. Please upload an image of a fruit.',
                'confidence': confidence,
                'top_predictions': top_predictions,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }), 400
        
        # Get appropriate icon
        fruit_icons = {
            'apple': 'fa-apple-alt',
            'banana': 'fa-seedling',
            'orange': 'fa-circle',
            'strawberry': 'fa-heart',
            'grape': 'fa-circle',
            'lemon': 'fa-lemon',
            'mango': 'fa-egg',
            'peach': 'fa-apple-alt',
            'pear': 'fa-lightbulb',
            'pineapple': 'fa-crown'
        }
        
        # Default icon if not found
        icon_class = fruit_icons.get(pred_label.lower(), 'fa-question')
        
        # Save to database - Only save if confidence is above threshold and is a fruit
        fruit_id = add_fruit(
            name=pred_label,
            image_path=image_path,
            predicted_class=pred_label,
            confidence_score=confidence,
            model_version="1.0"
        )
        
        # Log prediction details
        logger.info(f"Prediction: {pred_label}, Confidence: {confidence:.2f}, Fruit ID: {fruit_id}")
        logger.info(f"Model output shape: {model.output_shape}")
        logger.info(f"Number of class labels: {len(class_labels)}")
        logger.info(f"Prediction index: {pred_index}")
        logger.info(f"Top 5 predictions: {top_predictions}")
        
        # If confidence is low, log a warning
        if confidence < 0.5:
            logger.warning(f"Low confidence prediction: {pred_label} ({confidence:.2f})")
        
        return jsonify({
            'class': pred_label,
            'confidence': confidence,
            'icon': icon_class,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'top_predictions': top_predictions,
            'fruit_id': fruit_id  # Add fruit_id to response
        })
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/gradcam', methods=['POST'])
def gradcam():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if not class_labels:
        return jsonify({'error': 'Class labels not loaded'}), 500
    
    # Get fruit_id from form data
    fruit_id = request.form.get('fruit_id')
    if not fruit_id:
        return jsonify({'error': 'No fruit ID provided'}), 400
    
    # Get fruit record from database
    fruit_record = get_fruit_by_id(fruit_id)
    if not fruit_record:
        return jsonify({'error': 'Fruit record not found'}), 404
    
    # DEBUG: Print the entire fruit record with indices
    logger.info(f"Fruit record with indices:")
    for i, value in enumerate(fruit_record):
        logger.info(f"Index {i}: {value}")
    
    # Try to get the image path from the record
    image_path = fruit_record[2] if len(fruit_record) > 2 else None
    
    # Check if the path exists
    if image_path and os.path.exists(image_path):
        logger.info(f"Found valid image path at index 2: {image_path}")
    else:
        # If no valid path found, try to find the image in the uploads directory
        filename = None
        for item in fruit_record:
            if isinstance(item, str) and '.' in item and not os.path.isabs(item):
                filename = item
                break
        
        if filename:
            # Construct the full path
            upload_dir = os.path.join(basedir, 'static', 'uploads')
            image_path = os.path.join(upload_dir, filename)
            logger.info(f"Constructed image path: {image_path}")
            
            if not os.path.exists(image_path):
                logger.error(f"Constructed image path does not exist: {image_path}")
                return jsonify({'error': f'Image file not found: {filename}'}), 404
        else:
            logger.error("Could not find image path in database record")
            return jsonify({'error': 'Image path not found in database record'}), 404
    
    try:
        # Read the image from the saved path
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        # Preprocess image with correct size (96x96)
        img_array, original_img = preprocess_image(img_bytes)
        
        # Get prediction
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        
        # Check if the prediction index is valid
        if pred_index >= len(class_labels):
            logger.error(f"Prediction index {pred_index} is out of bounds for class_labels of length {len(class_labels)}")
            return jsonify({'error': f'Invalid prediction index: {pred_index}. The model might have more classes than the labels file.'}), 500
        
        pred_label = class_labels[pred_index]
        confidence = float(preds[0][pred_index])  # Convert numpy float32 to Python float
        
        logger.info(f"Predicted class: {pred_label}, confidence: {confidence}")
        
        # Check if prediction is a known non-fruit class
        if pred_label.lower() in [non_fruit.lower() for non_fruit in NON_FRUIT_CLASSES]:
            logger.warning(f"Image classified as non-fruit class: {pred_label} with confidence: {confidence:.2f}")
            return jsonify({
                'error': f'This image is classified as "{pred_label}", which is not a fruit. Grad-CAM visualization is not available.',
                'confidence': confidence
            }), 400
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"Low confidence prediction ({confidence:.2f} < {CONFIDENCE_THRESHOLD}) for Grad-CAM: Image not classified as a fruit")
            return jsonify({
                'error': 'The image is not classified as a fruit with sufficient confidence. Grad-CAM visualization is not available.',
                'confidence': confidence
            }), 400
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=pred_index)
        
        if heatmap is None:
            logger.error("Heatmap generation failed")
            return jsonify({'error': 'Heatmap generation failed'}), 500
        
        logger.info(f"Heatmap generated successfully. Shape: {heatmap.shape}")
        
        # Convert original image to numpy array
        original_img_array = np.array(original_img)
        logger.info(f"Original image array shape: {original_img_array.shape}, dtype: {original_img_array.dtype}")
        
        # Enhanced heatmap resizing with multiple fallback methods
        try:
            # Check if heatmap is valid
            if heatmap.size == 0 or len(heatmap.shape) != 2:
                logger.error(f"Invalid heatmap shape: {heatmap.shape}")
                return jsonify({'error': 'Invalid heatmap generated'}), 500
            
            # Ensure heatmap is float32
            heatmap = np.float32(heatmap)
            
            # Target size for resizing
            target_size = (original_img_array.shape[1], original_img_array.shape[0])
            logger.info(f"Resizing heatmap from {heatmap.shape} to {target_size}")
            
            # Try different interpolation methods
            try:
                # First try with INTER_NEAREST for small heatmaps
                heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_NEAREST)
                logger.info("Successfully resized with INTER_NEAREST")
            except Exception as e:
                logger.warning(f"INTER_NEAREST failed: {e}")
                try:
                    # Fallback to INTER_LINEAR
                    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
                    logger.info("Successfully resized with INTER_LINEAR")
                except Exception as e2:
                    logger.warning(f"INTER_LINEAR failed: {e2}")
                    try:
                        # Try INTER_CUBIC
                        heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)
                        logger.info("Successfully resized with INTER_CUBIC")
                    except Exception as e3:
                        logger.warning(f"INTER_CUBIC failed: {e3}")
                        # Last resort: use PIL resize
                        try:
                            # Convert heatmap to PIL Image, resize, then convert back
                            heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
                            heatmap_resized = heatmap_img.resize(target_size, Image.BILINEAR)
                            heatmap_resized = np.array(heatmap_resized) / 255.0
                            logger.info("Successfully resized with PIL")
                        except Exception as e4:
                            logger.error(f"All resize methods failed: {e4}")
                            return jsonify({'error': 'Failed to resize heatmap'}), 500
            
            # Ensure heatmap is still in valid range
            heatmap_resized = np.clip(heatmap_resized, 0, 1)
            
        except Exception as e:
            logger.error(f"Error resizing heatmap: {e}")
            return jsonify({'error': f'Error resizing heatmap: {str(e)}'}), 500
        
        # Convert to uint8 and apply colormap
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        logger.info(f"Heatmap colored shape: {heatmap_colored.shape}, dtype: {heatmap_colored.dtype}")
        
        # Ensure both arrays are uint8
        if original_img_array.dtype != np.uint8:
            logger.info(f"Converting original_img_array from {original_img_array.dtype} to uint8")
            original_img_array = np.uint8(original_img_array)
        
        # Create overlay with explicit output type
        try:
            superimposed_img = cv2.addWeighted(
                original_img_array, 0.6, 
                heatmap_colored, 0.4, 0, 
                dtype=cv2.CV_8U
            )
            logger.info(f"Superimposed image shape: {superimposed_img.shape}, dtype: {superimposed_img.dtype}")
        except Exception as e:
            logger.error(f"Error in addWeighted: {e}")
            # Fallback: try without explicit dtype
            try:
                superimposed_img = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
                logger.info("Superimposed image created without explicit dtype")
            except Exception as e2:
                logger.error(f"Fallback addWeighted also failed: {e2}")
                # Last resort: just use the heatmap
                superimposed_img = heatmap_colored.copy()
        
        # Save Grad-CAM images
        original_path, heatmap_path, overlay_path = save_gradcam_images(
            original_img, 
            Image.fromarray(heatmap_colored), 
            Image.fromarray(superimposed_img)
        )
        
        # Convert images to base64
        def img_to_base64(img_array):
            try:
                img = Image.fromarray(img_array.astype('uint8'))
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{img_str}"
            except Exception as e:
                logger.error(f"Error converting image to base64: {e}")
                return None
        
        original_base64 = img_to_base64(original_img_array)
        heatmap_base64 = img_to_base64(heatmap_colored)
        overlay_base64 = img_to_base64(superimposed_img)
        
        # Check if any conversion failed
        if not original_base64 or not heatmap_base64 or not overlay_base64:
            return jsonify({'error': 'Image conversion failed'}), 500
        
        # Save explanation to database
        explanation_data = {
            'original_path': original_path,
            'heatmap_path': heatmap_path,
            'overlay_path': overlay_path,
            'prediction': pred_label,
            'confidence': confidence
        }
        add_xai_explanation(fruit_id, "Grad-CAM", explanation_data)
        
        return jsonify({
            'original': original_base64,
            'heatmap': heatmap_base64,
            'overlay': overlay_base64,
            'prediction': pred_label,
            'confidence': confidence
        })
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    """Flask route for generating and downloading PDF reports"""
    temp_files = []  # Track temporary files for cleanup
    
    try:
        # Get data from request
        data = request.json
        
        # Extract required fields
        original_image_base64 = data.get('original')
        heatmap_image_base64 = data.get('heatmap')
        overlay_image_base64 = data.get('overlay')
        prediction = data.get('prediction', 'Unknown')
        confidence = data.get('confidence', 0) * 100  # Convert to percentage
        top_predictions = data.get('top_predictions', [])
        fruit_id = data.get('fruit_id')
        
        # If heatmap or overlay images are not provided, try to generate them using fruit_id
        if (not heatmap_image_base64 or not overlay_image_base64) and fruit_id:
            logger.info(f"Heatmap or overlay image not provided. Generating them for fruit_id: {fruit_id}")
            gradcam_result = generate_gradcam_for_report(fruit_id)
            if gradcam_result:
                gen_original, gen_heatmap, gen_overlay, gen_prediction, gen_confidence = gradcam_result
                # Use the generated images if they were not provided
                if not heatmap_image_base64:
                    heatmap_image_base64 = gen_heatmap
                if not overlay_image_base64:
                    overlay_image_base64 = gen_overlay
                # Update prediction and confidence if they were not provided or if we want to use the generated ones
                if prediction == 'Unknown':
                    prediction = gen_prediction
                if confidence == 0:
                    confidence = gen_confidence * 100
            else:
                logger.warning(f"Failed to generate Grad-CAM images for fruit_id: {fruit_id}")
        
        # Log the received data
        logger.info(f"Generating PDF report for: {prediction}")
        logger.info(f"Confidence: {confidence}%")
        logger.info(f"Number of top predictions: {len(top_predictions)}")
        logger.info(f"Original image provided: {original_image_base64 is not None}")
        logger.info(f"Heatmap image provided: {heatmap_image_base64 is not None}")
        logger.info(f"Overlay image provided: {overlay_image_base64 is not None}")
        
        # Generate the PDF report
        report_path = create_pdf_report(
            original_image_base64, 
            heatmap_image_base64, 
            overlay_image_base64, 
            prediction, 
            confidence, 
            top_predictions
        )
        
        temp_files.append(report_path)  # Add to cleanup list
        
        # Check if the PDF file was created successfully
        if not os.path.exists(report_path):
            logger.error("PDF file was not created successfully")
            return jsonify({'error': 'Failed to generate PDF report'}), 500
        
        # Get the file size
        file_size = os.path.getsize(report_path)
        logger.info(f"PDF report generated successfully. Size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("Generated PDF is empty")
            return jsonify({'error': 'Generated PDF is empty'}), 500
        
        # Return the PDF file
        response = send_file(
            report_path,
            as_attachment=True,
            download_name=f"fruit_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
        
        # Schedule cleanup after response is sent
        @response.call_on_close
        def cleanup():
            cleanup_temp_files(temp_files)
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        cleanup_temp_files(temp_files)  # Clean up even if there's an error
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Enhanced health check with system metrics"""
    try:
        # Get system resource usage
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check directory sizes
        uploads_dir = os.path.join(basedir, 'static', 'uploads')
        gradcam_dir = os.path.join(basedir, 'model', 'grad_cam_output')
        
        uploads_size = sum(os.path.getsize(os.path.join(uploads_dir, f)) 
                          for f in os.listdir(uploads_dir) 
                          if os.path.isfile(os.path.join(uploads_dir, f))) if os.path.exists(uploads_dir) else 0
        
        gradcam_size = sum(os.path.getsize(os.path.join(gradcam_dir, f)) 
                          for f in os.listdir(gradcam_dir) 
                          if os.path.isfile(os.path.join(gradcam_dir, f))) if os.path.exists(gradcam_dir) else 0
        
        uploads_count = len(os.listdir(uploads_dir)) if os.path.exists(uploads_dir) else 0
        gradcam_count = len(os.listdir(gradcam_dir)) if os.path.exists(gradcam_dir) else 0
        
        # Determine overall health status
        model_healthy = model is not None
        labels_healthy = len(class_labels) > 0
        overall_healthy = model_healthy and labels_healthy
        
        return jsonify({
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'model_loaded': model_healthy,
            'labels_loaded': labels_healthy,
            'classes_loaded': len(class_labels),
            'model_input_shape': model.input_shape if model else None,
            'model_output_shape': model.output_shape if model else None,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'uploads_size_mb': round(uploads_size / (1024 * 1024), 2),
            'gradcam_size_mb': round(gradcam_size / (1024 * 1024), 2),
            'uploads_file_count': uploads_count,
            'gradcam_file_count': gradcam_count,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Verification endpoint
@app.route('/verify_model', methods=['GET'])
def verify_model():
    """Verify model and labels consistency"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if not class_labels:
        return jsonify({'error': 'Class labels not loaded'}), 500
    
    output_size = model.output_shape[1]
    labels_count = len(class_labels)
    
    # Check if labels match model output
    labels_match = labels_count == output_size
    
    # Get sample predictions
    try:
        # Create a dummy image for testing
        dummy_img = np.random.randint(0, 255, (1, 96, 96, 3), dtype=np.uint8)
        dummy_img = tf.keras.applications.mobilenet_v2.preprocess_input(dummy_img.astype(np.float32))
        
        preds = model.predict(dummy_img)
        top_indices = np.argsort(preds[0])[-5:][::-1]
        
        sample_predictions = []
        for i in top_indices:
            if i < labels_count:
                sample_predictions.append({
                    'index': int(i),
                    'class': class_labels[i],
                    'confidence': float(preds[0][i])
                })
            else:
                sample_predictions.append({
                    'index': int(i),
                    'class': f"UNKNOWN_CLASS_{i}",
                    'confidence': float(preds[0][i])
                })
        
        return jsonify({
            'model_input_shape': model.input_shape,
            'model_output_shape': model.output_shape,
            'labels_count': labels_count,
            'output_size': output_size,
            'labels_match': labels_match,
            'sample_predictions': sample_predictions,
            'warning': None if labels_match else "CRITICAL: Labels count doesn't match model output size!"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)