import tensorflow as tf
import json

# Load the model
model = tf.keras.models.load_model("optimized_fruit_model.keras")

# Get the number of output classes
num_classes = model.output_shape[1]

# Create generic class names
class_names = [f"Class_{i}" for i in range(num_classes)]

# Save to JSON file
with open("enhanced_class_labels.json", "w") as f:
    json.dump(class_names, f)

print(f"Created enhanced_class_labels.json with {num_classes} generic classes")