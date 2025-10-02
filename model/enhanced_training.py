import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
import json
from PIL import Image
from tensorflow.keras import mixed_precision

# Enable mixed precision for faster training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Configure GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")

# Paths
base_dir = r"H:\Project\fruit_classifier_project\data\fruits-360_100x100\fruits-360"
train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Test')

# Enhanced preprocessing function with noise reduction
def enhanced_preprocess(img):
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img
    
    denoised = cv2.fastNlMeansDenoisingColored(img_array.astype(np.uint8), None, 10, 10, 7, 21)
    return tf.keras.applications.mobilenet_v2.preprocess_input(denoised)

# Optimized parameters
img_size = (96, 96)  # Reduced from 128x128
batch_size = 64  # Increased from 32

# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=enhanced_preprocess,
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=30.0,
    fill_mode='nearest',
    validation_split=0.1
)

test_datagen = ImageDataGenerator(
    preprocessing_function=enhanced_preprocess
)

# Get consistent classes
train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
test_classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
consistent_classes = sorted(set(train_classes) & set(test_classes))
if "Caju seed 1" in consistent_classes:
    consistent_classes.remove("Caju seed 1")
print(f"Using {len(consistent_classes)} consistent classes")

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=consistent_classes
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=consistent_classes
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    classes=consistent_classes
)

num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Calculate class weights (FIX: Added this section)
class_counts = Counter(train_generator.classes)
total_samples = sum(class_counts.values())
class_weights = {}
for class_idx, count in class_counts.items():
    class_weights[class_idx] = total_samples / (num_classes * count)
print("Class weights calculated successfully")
print(f"Number of classes with weights: {len(class_weights)}")

# Simplified model architecture
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*img_size, 3),
    include_top=False,
    weights='imagenet'
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # Reduced from 512
x = Dropout(0.3)(x)  # Reduced from 0.5
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Optimized callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Reduced from 10
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)  # Reduced from 4
checkpoint = ModelCheckpoint('optimized_fruit_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
callbacks = [early_stop, reduce_lr, checkpoint]

# Optimized training
print("\nStarting optimized training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,  # Reduced from 60
    callbacks=callbacks,
    class_weight=class_weights  # This should now work
)

# Fine-tuning
print("\nStarting fine-tuning...")
for layer in base_model.layers[-40:]:  # Reduced from 80
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,  # Reduced from 30
    callbacks=callbacks,
    class_weight=class_weights  # This should now work
)

# Evaluate
print("\nEvaluating model...")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model
model.save('optimized_fruit_classifier_model.keras')
class_labels = list(train_generator.class_indices.keys())
with open('optimized_class_labels.json', 'w') as f:
    json.dump(class_labels, f)
    
print("Optimized model and class labels saved successfully.")