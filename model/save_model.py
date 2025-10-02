import tensorflow as tf
import os

# Get the current directory (model folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

# Try to find your model file
possible_model_paths = [
    os.path.join(current_dir, "optimized_fruit_model.keras"),
    os.path.join(current_dir, "best_model.h5"),
    os.path.join(current_dir, "best_model.keras"),
    os.path.join(project_dir, "model", "optimized_fruit_model.keras"),
    os.path.join(project_dir, "model", "best_model.h5"),
    os.path.join(project_dir, "model", "best_model.keras")
]

model_path = None
for path in possible_model_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("❌ Model file not found! Looking for one of these files:")
    for path in possible_model_paths:
        print(f"  - {path}")
    exit(1)

print(f"✅ Found model at: {model_path}")

# Load your trained model
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Save it with a clear name
final_model_path = os.path.join(current_dir, "fruit_classifier_final.keras")
model.save(final_model_path)
print(f"✅ Model saved to: {final_model_path}")

# Also save in .h5 format for compatibility
h5_path = os.path.join(current_dir, "fruit_classifier_final.h5")
model.save(h5_path)
print(f"✅ Model also saved as: {h5_path}")

# Print model summary
print("\n=== Model Summary ===")
model.summary()