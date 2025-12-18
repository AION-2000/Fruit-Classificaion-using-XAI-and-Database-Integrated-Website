# Fruit Classification with Explainable AI

A production-ready web application leveraging deep learning for fruit image classification with integrated explainability through Grad-CAM visualization. Built with Flask and TensorFlow, this system provides transparent, interpretable predictions backed by persistent storage.

<img width="919" height="439" alt="image" src="https://github.com/user-attachments/assets/0eeca6ff-cec3-4447-82ef-282f9becbd72" />

<img width="800" height="384" alt="image" src="https://github.com/user-attachments/assets/1be6b818-b2de-4e25-bb84-055e70825d39" />

<img width="815" height="391" alt="image" src="https://github.com/user-attachments/assets/dc611cda-2d6b-4717-9abc-95e472a7de7a" />

<img width="884" height="420" alt="image" src="https://github.com/user-attachments/assets/7991deb1-a37b-4123-a327-2a0a48da52b8" />

<img width="448" height="671" alt="image" src="https://github.com/user-attachments/assets/c5a79070-6d0f-403e-bdca-537efb182a4d" />

<img width="577" height="616" alt="image" src="https://github.com/user-attachments/assets/8b65c8b2-9bd8-4523-8eba-86e2adeb404b" />

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Details](#model-details)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

This application addresses the critical need for transparency in AI decision-making by combining state-of-the-art image classification with explainable AI techniques. Users can upload fruit images and receive not only accurate predictions but also visual explanations showing which image regions influenced the model's decision.

### Key Benefits

- **Transparency**: Grad-CAM heatmaps provide visual evidence for model predictions
- **Traceability**: Complete audit trail with database-backed prediction history
- **Extensibility**: Modular architecture supports easy integration of new models and XAI techniques
- **Production-Ready**: Includes error handling, logging, and database management

## Features

### Core Functionality

- **Multi-Class Fruit Classification**: Identifies various fruit types with confidence scoring
- **Explainable AI Integration**: Generates Grad-CAM visualizations highlighting decision-critical regions
- **Persistent Storage**: SQLite database maintains complete prediction history with metadata
- **RESTful API**: Clean API design for programmatic access
- **Responsive Web Interface**: Modern, intuitive UI built with semantic HTML5/CSS3

### Technical Features

- Efficient image preprocessing pipeline
- Real-time inference with optimized model loading
- Automatic file management for uploads and generated visualizations
- Database connection pooling and transaction management
- Comprehensive error handling and logging

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Client    │────────▶│ Flask Server │────────▶│  ML Model   │
│  (Browser)  │◀────────│   (app.py)   │◀────────│(TensorFlow) │
└─────────────┘         └──────────────┘         └─────────────┘
                              │    ▲
                              │    │
                              ▼    │
                        ┌──────────────┐
                        │   Database   │
                        │   (SQLite)   │
                        └──────────────┘
```

## Technology Stack

### Backend
- **Framework**: Flask 2.x
- **ML/DL**: TensorFlow 2.x / Keras
- **Database**: SQLite 3
- **Image Processing**: OpenCV, Pillow
- **Scientific Computing**: NumPy, Matplotlib

### Frontend
- **Markup**: HTML5 with semantic elements
- **Styling**: CSS3 with modern layout techniques
- **Scripting**: Vanilla JavaScript (ES6+)

### Development Tools
- Python 3.8+
- Virtual Environment (venv)
- Git version control

## Project Structure

```
fruit-classification-xai/
│
├── app.py                      # Flask application entry point
├── database.py                 # Database abstraction layer
├── init_db.py                  # Database initialization script
├── view_database.py            # Database inspection utility
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusion rules
├── README.md                   # Project documentation
├── LICENSE.txt                 # License information
│
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css          # Application styles
│   ├── js/
│   │   └── main.js            # Client-side logic
│   ├── uploads/               # User-uploaded images (gitignored)
│   └── gradcam_output/        # Generated heatmaps (gitignored)
│
├── templates/                  # Jinja2 templates
│   ├── index.html             # Upload interface
│   ├── result.html            # Prediction results
│   └── history.html           # Prediction history
│
├── model/                      # ML models (gitignored)
│   └── fruit_classifier.h5    # Trained Keras model
│
├── database/                   # Database files (gitignored)
│   └── predictions.db         # SQLite database
│
└── venv/                       # Virtual environment (gitignored)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fruit-classification-xai.git
cd fruit-classification-xai
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Initialize database**
```bash
python init_db.py
```

5. **Add trained model**

Place your trained model file in the `model/` directory:
```bash
model/fruit_classifier.h5
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
FLASK_ENV=development
FLASK_APP=app.py
DATABASE_PATH=database/predictions.db
MODEL_PATH=model/fruit_classifier.h5
UPLOAD_FOLDER=static/uploads
GRADCAM_FOLDER=static/gradcam_output
MAX_UPLOAD_SIZE=16777216  # 16MB
ALLOWED_EXTENSIONS=png,jpg,jpeg
```

### Application Settings

Modify `app.py` configuration as needed:

```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GRADCAM_FOLDER'] = 'static/gradcam_output'
```

## Usage

### Starting the Application

```bash
python app.py
```

Access the application at `http://127.0.0.1:5000`

### Web Interface

1. Navigate to the homepage
2. Click "Upload Image" and select a fruit image
3. Submit to receive prediction with Grad-CAM visualization
4. View prediction history via the navigation menu

### API Endpoints

#### POST /predict
Upload and classify an image.

**Request:**
```bash
curl -X POST -F "file=@apple.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "success": true,
  "prediction": "Apple",
  "confidence": 0.9847,
  "gradcam_path": "/static/gradcam_output/123456.jpg",
  "timestamp": "2024-12-19T10:30:00Z"
}
```

## Model Details

### Architecture

The classification model uses a fine-tuned ResNet50 architecture with the following modifications:

- **Base Model**: ResNet50 pre-trained on ImageNet
- **Custom Top Layers**: 
  - Global Average Pooling
  - Dense (256 units, ReLU activation)
  - Dropout (0.5)
  - Dense (num_classes, Softmax activation)

### Training Details

- **Dataset**: Fruits-360 dataset (90,000+ images, 131 classes)
- **Input Size**: 224x224x3
- **Preprocessing**: Standard ImageNet normalization
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: Categorical cross-entropy
- **Training Accuracy**: 98.5%
- **Validation Accuracy**: 96.8%

### Explainability (Grad-CAM)

Gradient-weighted Class Activation Mapping (Grad-CAM) generates visual explanations by:

1. Computing gradients of the predicted class score with respect to the final convolutional layer
2. Global average pooling of gradients to obtain importance weights
3. Weighted combination of forward activation maps
4. ReLU application to focus on positive influences
5. Upsampling and overlay on original image

## Database Schema

### Predictions Table

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    gradcam_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Utility Scripts

**View database contents:**
```bash
python view_database.py
```

**Reset database:**
```bash
python init_db.py --reset
```

## Development

### Running in Development Mode

```bash
export FLASK_ENV=development
flask run --debug
```

### Code Style

This project follows PEP 8 style guidelines. Format code using:

```bash
black app.py database.py
flake8 .
```

### Adding New Features

1. Create a feature branch: `git checkout -b feature/new-feature`
2. Implement changes with appropriate tests
3. Update documentation
4. Submit a pull request

## Testing

### Unit Tests

```bash
python -m pytest tests/
```

### Integration Tests

```bash
python -m pytest tests/integration/
```

### Test Coverage

```bash
pytest --cov=. --cov-report=html
```

## Deployment

### Production Considerations

1. **Use a production WSGI server**: Gunicorn or uWSGI
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

2. **Set up reverse proxy**: Nginx configuration example
```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

3. **Enable HTTPS**: Use Let's Encrypt or similar certificate authority

4. **Configure proper logging**: Use rotating file handlers

5. **Set environment to production**:
```bash
export FLASK_ENV=production
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
```

## Contributing

We welcome contributions from the community! Please follow these guidelines:

### Contribution Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Commit Message Convention

Follow conventional commits specification:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Build process or auxiliary tool changes

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for full details.

---

## Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Fruits-360 dataset creators
- Grad-CAM paper authors: Selvaraju et al.

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

## Roadmap

- [ ] Support for additional XAI techniques (LIME, SHAP)
- [ ] Multi-model ensemble predictions
- [ ] Real-time video classification
- [ ] Mobile application
- [ ] Cloud deployment templates (AWS, GCP, Azure)
- [ ] Advanced analytics dashboard
