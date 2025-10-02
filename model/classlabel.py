# Create a script to regenerate class labels
import os
import json

train_dir = r"E:\Project\fruit_classifier_project\fruit_classifier_project\data\fruits-360_100x100\fruits-360\Training"
class_names = sorted(os.listdir(train_dir))

with open("class_labels.json", "w") as f:
    json.dump(class_names, f)

print(f"Generated class_labels.json with {len(class_names)} classes")