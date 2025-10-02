import os

test_dir = r"E:\Project\fruit_classifier_project\fruit_classifier_project\data\fruits-360_100x100\Test"

empty_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)) and len(os.listdir(os.path.join(test_dir, d))) == 0]

print("Empty class folders in Test:", empty_classes)
