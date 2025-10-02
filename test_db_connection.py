import sqlite3
import os
from datetime import datetime

# Fix the database path to ensure it's created in the correct location
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'fruit_database.db')

# Initialize the database first
def initialize_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing tables if they exist to start fresh
    cursor.execute('DROP TABLE IF EXISTS xai_explanations')
    cursor.execute('DROP TABLE IF EXISTS fruits')
    
    cursor.execute('''
    CREATE TABLE fruits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        image_path TEXT NOT NULL,
        predicted_class TEXT,
        confidence_score REAL,
        model_version TEXT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE xai_explanations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fruit_id INTEGER,
        method TEXT NOT NULL,
        explanation_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (fruit_id) REFERENCES fruits (id) ON DELETE CASCADE
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

# Connect to database and run tests
try:
    # Initialize the database
    initialize_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Database tables:", [table[0] for table in tables])
    
    # Insert test data
    cursor.execute('''
    INSERT INTO fruits (name, image_path, predicted_class, confidence_score, model_version, upload_date)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', ('Test Apple', 'static/uploads/test.jpg', 'Apple', 0.95, '1.0', datetime.now()))
    
    conn.commit()
    
    # Retrieve test data
    cursor.execute("SELECT * FROM fruits")
    result = cursor.fetchone()
    print("Test record:", result)
    
    # Test XAI explanation insertion
    cursor.execute('''
    INSERT INTO xai_explanations (fruit_id, method, explanation_data, created_at)
    VALUES (?, ?, ?, ?)
    ''', (1, 'Grad-CAM', '{"test": "data"}', datetime.now()))
    
    conn.commit()
    
    # Retrieve XAI explanation
    cursor.execute("SELECT * FROM xai_explanations WHERE fruit_id = ?", (1,))
    explanation_result = cursor.fetchone()
    print("Test explanation record:", explanation_result)
    
    conn.close()
    print("✅ Database test successful!")
    
except Exception as e:
    print(f"❌ Database test failed: {e}")