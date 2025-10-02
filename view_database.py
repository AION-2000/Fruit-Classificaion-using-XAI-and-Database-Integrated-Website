import sqlite3
import os
import json

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'database', 'fruit_database.db')

def view_database():
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Database tables:")
        for table in tables:
            print(f"- {table[0]}")
        
        print("\n" + "="*50 + "\n")
        
        # View fruits table
        print("Contents of 'fruits' table:")
        cursor.execute("SELECT * FROM fruits")
        fruits = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        print(f"Columns: {', '.join(column_names)}")
        print("-" * 80)
        
        for fruit in fruits:
            print(f"ID: {fruit[0]}")
            print(f"Name: {fruit[1]}")
            print(f"Image Path: {fruit[2]}")
            print(f"Predicted Class: {fruit[3]}")
            print(f"Confidence: {fruit[4]}")
            print(f"Model Version: {fruit[5]}")
            print(f"Upload Date: {fruit[6]}")
            print("-" * 40)
        
        print("\n" + "="*50 + "\n")
        
        # View xai_explanations table
        print("Contents of 'xai_explanations' table:")
        cursor.execute("SELECT * FROM xai_explanations")
        explanations = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        print(f"Columns: {', '.join(column_names)}")
        print("-" * 80)
        
        for explanation in explanations:
            print(f"ID: {explanation[0]}")
            print(f"Fruit ID: {explanation[1]}")
            print(f"Method: {explanation[2]}")
            print(f"Explanation Data: {json.loads(explanation[3])}")
            print(f"Created At: {explanation[4]}")
            print("-" * 40)
        
        # Close connection
        conn.close()
        
    except Exception as e:
        print(f"Error viewing database: {e}")

if __name__ == "__main__":
    view_database()