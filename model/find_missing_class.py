import os

train_dir = r"E:\Project\fruit_classifier_project\fruit_classifier_project\data\fruits-360_100x100\Training"
test_dir = r"E:\Project\fruit_classifier_project\fruit_classifier_project\data\fruits-360_100x100\Test"

print("Checking training directory:", train_dir)
print("Checking test directory:", test_dir)

try:
    train_classes = set(os.listdir(train_dir))
    print(f"Found {len(train_classes)} classes in Training.")
except Exception as e:
    print("Error reading training directory:", e)
    train_classes = set()

try:
    test_classes = set(os.listdir(test_dir))
    print(f"Found {len(test_classes)} classes in Test.")
except Exception as e:
    print("Error reading test directory:", e)
    test_classes = set()

missing_in_test = train_classes - test_classes

if missing_in_test:
    print("Missing class(es) in Test:", missing_in_test)
else:
    print("✅ No missing classes in Test.")
