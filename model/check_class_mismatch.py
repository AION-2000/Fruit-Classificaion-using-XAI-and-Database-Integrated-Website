import os

print("🔍 Starting class mismatch check...")

train_dir = 'data/fruits-360_split/train'
val_dir = 'data/fruits-360_split/val'
test_dir = 'data/fruits-360_split/test'

if not os.path.exists(train_dir):
    print("❌ Train directory not found!")
else:
    train_classes = sorted(os.listdir(train_dir))
    print(f"✅ Found {len(train_classes)} training classes.")

if not os.path.exists(val_dir):
    print("❌ Validation directory not found!")
else:
    val_classes = sorted(os.listdir(val_dir))
    print(f"✅ Found {len(val_classes)} validation classes.")

if not os.path.exists(test_dir):
    print("❌ Test directory not found!")
else:
    test_classes = sorted(os.listdir(test_dir))
    print(f"✅ Found {len(test_classes)} test classes.")

# If all exist, compare them
if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
    missing_in_val = set(train_classes) - set(val_classes)
    missing_in_test = set(train_classes) - set(test_classes)

    if missing_in_val:
        print("⚠️ Classes missing in validation set:")
        for cls in missing_in_val:
            print(f" - {cls}")
    else:
        print("✅ Validation set matches train set.")

    if missing_in_test:
        print("⚠️ Classes missing in test set:")
        for cls in missing_in_test:
            print(f" - {cls}")
    else:
        print("✅ Test set matches train set.")
