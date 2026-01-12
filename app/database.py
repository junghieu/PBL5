import os
import sqlite3
from datetime import datetime
from contextlib import contextmanager

DATABASE_PATH = "ocr_database.db"


def init_db():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            text_content TEXT,
            docx_path TEXT
        )
    """)
    
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_scan_record(filename: str) -> int:
    """Create a new scan record with Pending status."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO scans (filename, status) VALUES (?, ?)",
            (filename, "Pending")
        )
        conn.commit()
        return cursor.lastrowid


def update_scan_status(scan_id: int, status: str, text_content: str = None, docx_path: str = None):
    """Update scan record status and content."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE scans 
               SET status = ?, text_content = ?, docx_path = ?, updated_at = CURRENT_TIMESTAMP 
               WHERE id = ?""",
            (status, text_content, docx_path, scan_id)
        )
        conn.commit()


def get_scan_by_id(scan_id: int):
    """Get scan record by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
