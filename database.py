import sqlite3
import json
import os
from datetime import datetime

# Fix the database path to ensure it's created in the correct location
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'fruit_database.db')

def get_connection():
    # Ensure the database directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def add_fruit(name, image_path, predicted_class=None, confidence_score=None, model_version=None):
    conn = get_connection()
    cursor = conn.cursor()
    
    # Use the correct column name (upload_date instead of timestamp)
    cursor.execute('''
    INSERT INTO fruits (name, image_path, predicted_class, confidence_score, model_version, upload_date)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, image_path, predicted_class, confidence_score, model_version, datetime.now()))
    
    fruit_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return fruit_id

def add_xai_explanation(fruit_id, method, explanation_data):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO xai_explanations (fruit_id, method, explanation_data, created_at)
    VALUES (?, ?, ?, ?)
    ''', (fruit_id, method, json.dumps(explanation_data), datetime.now()))
    
    conn.commit()
    conn.close()

def get_fruit_by_id(fruit_id):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM fruits WHERE id = ?', (fruit_id,))
    result = cursor.fetchone()
    
    conn.close()
    return result

def get_xai_explanations(fruit_id):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM xai_explanations WHERE fruit_id = ?', (fruit_id,))
    results = cursor.fetchall()
    
    conn.close()
    
    # Parse the JSON data for each explanation
    explanations = []
    for result in results:
        explanation = {
            'id': result[0],
            'fruit_id': result[1],
            'method': result[2],
            'explanation_data': json.loads(result[3]),
            'created_at': result[4]
        }
        explanations.append(explanation)
    
    return explanations