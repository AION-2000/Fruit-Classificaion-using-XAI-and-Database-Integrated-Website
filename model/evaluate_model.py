import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

if gpus:
    try:
        # Configure GPU memory growth to prevent TensorFlow from allocating all memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs detected")
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU devices available. Using CPU instead.")
    print("To use GPU, ensure you have:")
    print("1. Installed CUDA and cuDNN compatible with your TensorFlow version")
    print("2. The correct GPU drivers installed")
    print("3. The GPU version of TensorFlow installed")

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the model path - using the correct filename
model_path = os.path.join(script_dir, 'optimized_fruit_model.keras')
print(f"\nLooking for model at: {model_path}")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure the model file exists in the same directory as the script.")
    exit(1)

# Load the trained model
try:
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define the path for class labels - using the correct filename
labels_path = os.path.join(script_dir, 'enhanced_class_labels.json')
print(f"Looking for class labels at: {labels_path}")

# Fallback to class_labels.json if enhanced version doesn't exist
if not os.path.exists(labels_path):
    labels_path = os.path.join(script_dir, 'class_labels.json')
    print(f"Enhanced class labels not found. Using fallback at: {labels_path}")

if not os.path.exists(labels_path):
    print(f"Error: Class labels file not found at {labels_path}")
    print("Please ensure the class labels file exists in the same directory as the script.")
    exit(1)

# Load class labels
try:
    with open(labels_path, 'r') as f:
        class_labels = json.load(f)
    print("Class labels loaded successfully.")
    print(f"Number of classes: {len(class_labels)}")
except Exception as e:
    print(f"Error loading class labels: {e}")
    exit(1)

# Create test data generator
base_dir = r"H:\Project\fruit_classifier_project\data\fruits-360_100x100\fruits-360"
test_dir = os.path.join(base_dir, 'Test')

# Check if test directory exists
if not os.path.exists(test_dir):
    print(f"Error: Test directory not found at: {test_dir}")
    exit(1)

# Note: Use simple rescaling for evaluation (no preprocessing needed for evaluation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(96, 96),  # Must match training size
    batch_size=64,
    class_mode='categorical',
    shuffle=False,  # Important: keep shuffle=False for evaluation
    classes=class_labels
)

# Generate predictions
print("\nGenerating predictions for test set...")
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Calculate evaluation metrics
print("\nCalculating evaluation metrics...")
# 1. Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 2. Precision, Recall, F1-score (weighted average for imbalanced dataset)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

# 3. Classification report (per-class metrics)
print("\nPer-class Classification Report:")
class_report = classification_report(
    y_true, 
    y_pred, 
    target_names=class_labels,
    digits=4
)
print(class_report)

# 4. Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix using matplotlib only
plt.figure(figsize=(20, 16))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix', fontsize=16)
plt.colorbar()

# Add class labels to axes (show only every 5th label to avoid crowding)
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks[::5], class_labels[::5], rotation=45, ha='right')
plt.yticks(tick_marks[::5], class_labels[::5])
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Top misclassifications
print("\nTop Misclassifications:")
# Get indices of misclassified samples
misclassified_indices = np.where(y_true != y_pred)[0]
misclassified_pairs = [(y_true[i], y_pred[i]) for i in misclassified_indices]

# Count most common misclassifications
from collections import defaultdict
misclass_counts = defaultdict(int)
for true_class, pred_class in misclassified_pairs:
    misclass_counts[(true_class, pred_class)] += 1

# Get top 10 misclassifications
top_misclass = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print("True Class -> Predicted Label (Count)")
for (true_idx, pred_idx), count in top_misclass:
    true_label = class_labels[true_idx]
    pred_label = class_labels[pred_idx]
    print(f"{true_label} -> {pred_label}: {count} times")

# 6. Save evaluation results
evaluation_results = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'confusion_matrix': cm.tolist(),
    'class_labels': class_labels
}

results_path = os.path.join(script_dir, 'evaluation_results.json')
with open(results_path, 'w') as f:
    json.dump(evaluation_results, f, indent=4)

print(f"\nEvaluation results saved to '{results_path}'")
print("Confusion matrix saved to 'confusion_matrix.png'")
print("="*50)
print("EVALUATION COMPLETE")
print("="*50)