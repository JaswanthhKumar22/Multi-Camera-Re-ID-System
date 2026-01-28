import sqlite3
from datetime import datetime

class ReIDDatabase:
    def __init__(self, db_path="reid_events.db"):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Create SQLite connection inside DB thread"""
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            global_id INTEGER,
            camera_id INTEGER,
            timestamp TEXT,
            is_suspect INTEGER,
            similarity REAL
        )
        """)
        self.conn.commit()

    def log_event(self, global_id, camera_id, is_suspect=False, similarity=0.0):
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO events (global_id, camera_id, timestamp, is_suspect, similarity)
        VALUES (?, ?, ?, ?, ?)
        """, (
            global_id,
            camera_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            int(is_suspect),
            float(similarity)
        ))
        self.conn.commit()
