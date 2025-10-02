import sqlite3
import os

# Fix the database path to ensure it's created in the correct location
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'fruit_database.db')

def initialize_database():
    # Ensure the database directory exists
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

if __name__ == "__main__":
    initialize_database()