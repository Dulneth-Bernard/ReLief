import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'relief.db')

def get_db_connection():
    """Establish connection to SQLite DB"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the SQLite database with the history table"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL UNIQUE,
            image_url TEXT,
            filename TEXT,
            model_id TEXT,
            prediction TEXT,
            confidence REAL,
            probabilities TEXT,
            gradcam_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_image_upload(image_id, filename, image_url):
    """Save an initial record when image is uploaded"""
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT OR IGNORE INTO history (image_id, filename, image_url)
            VALUES (?, ?, ?)
        ''', (image_id, filename, image_url))
        conn.commit()
    except Exception as e:
        print(f"Error saving image upload to DB: {e}")
    finally:
        conn.close()

def save_prediction(image_id, model_id, prediction, confidence, probabilities):
    """Update record with prediction results"""
    conn = get_db_connection()
    c = conn.cursor()
    # Check if record exists
    c.execute('SELECT id FROM history WHERE image_id = ?', (image_id,))
    row = c.fetchone()
    
    probs_json = json.dumps(probabilities) if probabilities else None
    
    try:
        if row:
            c.execute('''
                UPDATE history 
                SET prediction = ?, confidence = ?, probabilities = ?, model_id = ?
                WHERE image_id = ?
            ''', (prediction, confidence, probs_json, model_id, image_id))
        else:
            # Fallback if image wasn't saved on upload
            c.execute('''
                INSERT INTO history (image_id, model_id, prediction, confidence, probabilities)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_id, model_id, prediction, confidence, probs_json))
        conn.commit()
    except Exception as e:
        print(f"Error saving prediction to DB: {e}")
    finally:
        conn.close()

def update_gradcam(image_id, gradcam_url):
    """Update record with explainability map URL"""
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            UPDATE history 
            SET gradcam_url = ?
            WHERE image_id = ?
        ''', (gradcam_url, image_id))
        conn.commit()
    except Exception as e:
        print(f"Error updating gradcam to DB: {e}")
    finally:
        conn.close()

def get_history():
    """Retrieve all history records"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        result.append({
            'id': row['id'],
            'image_id': row['image_id'],
            'image_url': row['image_url'],
            'filename': row['filename'],
            'model_id': row['model_id'],
            'prediction': row['prediction'],
            'confidence': row['confidence'],
            'probabilities': json.loads(row['probabilities']) if row['probabilities'] else {},
            'gradcam_url': row['gradcam_url'],
            'created_at': row['created_at']
        })
    return result

# Initialize DB when this module is imported
init_db()
