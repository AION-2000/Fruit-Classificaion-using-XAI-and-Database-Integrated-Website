import json
import os

# Get the current directory (model folder) and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Your class labels (update with your actual labels)
class_labels = [
    "Apple", "Banana", "Orange", "Strawberry", "Grape", 
    "Lemon", "Mango", "Peach", "Pear", "Pineapple"
]

# Define save paths relative to project root
save_paths = [
    os.path.join(project_root, "enhanced_class_labels.json"),
    os.path.join(current_dir, "enhanced_class_labels.json"),
    os.path.join(current_dir, "class_labels.json")
]

# Create directories if they don't exist
for path in save_paths:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Save the class labels
for path in save_paths:
    with open(path, 'w') as f:
        json.dump(class_labels, f, indent=2)
    print(f"✅ Class labels saved to: {path}")