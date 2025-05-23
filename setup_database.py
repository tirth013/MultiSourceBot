import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_database():
    # Get database URI from environment variable
    database_uri = os.getenv('DATABASE_URI', 'research_data.db')
    
    # Create database connection
    conn = sqlite3.connect(database_uri)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        reliability FLOAT,
        last_updated TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS research_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER,
        content TEXT NOT NULL,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (source_id) REFERENCES sources (id)
    )
    ''')
    
    # Insert sample data
    sample_sources = [
        (1, "Google Scholar", "Academic", 0.9, "2024-03-20"),
        (2, "Wikipedia", "Encyclopedia", 0.7, "2024-03-20"),
        (3, "arXiv", "Preprint", 0.8, "2024-03-20")
    ]
    
    cursor.executemany('''
    INSERT OR IGNORE INTO sources (id, name, type, reliability, last_updated)
    VALUES (?, ?, ?, ?, ?)
    ''', sample_sources)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()