import os
import json
import tensorflow as tf

# Load the model to get the expected number of classes
model = tf.keras.models.load_model("optimized_fruit_model.keras")
num_classes = model.output_shape[1]

# Path to your training directory
train_dir = r"H:\Project\fruit_classifier_project\data\fruits-360_100x100\fruits-360\Training"

# Get all class names
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# Remove "Caju seed 1" if it exists (as in your training script)
if "Caju seed 1" in class_names:
    class_names.remove("Caju seed 1")

# Ensure we have the right number of classes
if len(class_names) != num_classes:
    print(f"Warning: Found {len(class_names)} classes but model expects {num_classes}")
    
    # If we have fewer classes than expected, pad with generic names
    while len(class_names) < num_classes:
        class_names.append(f"Class_{len(class_names)}")
    
    # If we have more classes than expected, truncate
    if len(class_names) > num_classes:
        class_names = class_names[:num_classes]
        print(f"Truncated to {num_classes} classes")

# Save to JSON file
with open("enhanced_class_labels.json", "w") as f:
    json.dump(class_names, f)

print(f"Created enhanced_class_labels.json with {len(class_names)} classes")
print("First 10 classes:", class_names[:10])