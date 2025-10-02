import os

save_path = os.path.join(os.getcwd(), 'fruit_model.h5')
model.save(save_path)
print(f"Model saved at: {save_path}")
# This script saves the trained model to the current working directory.