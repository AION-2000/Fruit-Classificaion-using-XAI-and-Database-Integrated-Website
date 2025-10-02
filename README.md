# Fruit-Classificaion-using-XAI-and-Database-Integrated-Website

A web application that classifies fruits from images using a deep learning model and provides visual explanations for its predictions using Grad-CAM.



## Overview

This project demonstrates the power of Explainable AI (XAI) in a practical web application. Users can upload an image of a fruit, and the application will not only classify it (e.g., Apple, Banana, Orange) but also generate a Grad-CAM heatmap. This heatmap highlights the specific regions of the image the model focused on, making the decision-making process transparent and trustworthy.

All user uploads and their corresponding results are stored in a database, allowing for a history of predictions.

## Key Features

-   **Intuitive Web Interface**: A clean and simple UI for uploading fruit images.
-   **Accurate Classification**: Powered by a pre-trained `[e.g., ResNet50, Custom CNN]` model.
-   **Visual Explanations with Grad-CAM**: Generates heatmaps to show *why* the model made a certain prediction.
-   **Result History**: View past uploads and their classifications/heatmaps.
-   **Database Integration**: Stores all predictions and metadata for persistence.

## Tech Stack

-   **Backend**: Python, Flask
-   **Machine Learning**: TensorFlow / Keras
-   **Frontend**: HTML5, CSS3, JavaScript
-   **Database**: SQLite
-   **Libraries**: NumPy, Pillow, OpenCV, Matplotlib

## Project Structure
├── app.py # Main Flask application and routes
├── database.py # Handles all database interactions
├── init_db.py # Script to initialize the database and tables
├── view_database.py # Utility to inspect the database contents
├── requirements.txt # List of Python dependencies
├── .gitignore # Specifies files for Git to ignore
├── README.md # This file
│
├── static/ # Static assets
│ ├── css/ # Stylesheets
│ ├── js/ # Client-side scripts
│ ├── uploads/ # User-uploaded images (git-ignored)
│ └── gradcam_output/ # Generated Grad-CAM heatmaps (git-ignored)
│
├── templates/ # HTML templates for web pages
│ └── ...
│
├── model/ # Trained ML model file (e.g., model.h5) (git-ignored)
│
├── data/ # Dataset or training data (git-ignored)
├── database/ # SQLite database file (e.g., predictions.db) (git-ignored)
│
└── venv/ # Python virtual environment (git-ignored)


# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# Initialize the Database
python init_db.py

# Place the Trained Model
Download or place your trained model file (e.g., fruit_classifier.h5) inside the model/ directory.
Ensure the model path in app.py correctly points to your model file.

# Run the Application
python app.py


#Usage
Open your web browser and navigate to http://127.0.0.1:5000.
Click the "Upload Image" button and select a fruit image from your computer.
Submit the form to see the prediction.
The results page will display the predicted fruit, confidence score, and the Grad-CAM heatmap overlay.
Use the navigation links to view a history of all past predictions.


#Model & XAI Details
Model Architecture: The project uses a [e.g., ResNet50 model pre-trained on ImageNet] that was fine-tuned on a fruit dataset.
Dataset: The model was trained on the [e.g., Fruits-360 dataset].
Explainability (Grad-CAM): Grad-CAM (Gradient-weighted Class Activation Mapping) uses the gradients of the target class flowing into the final convolutional layer to produce a coarse localization map, highlighting important regions in the image for prediction.


# Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

# Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
# License
Distributed under the MIT License. See LICENSE.txt for more information.
