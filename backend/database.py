from datetime import datetime, timezone
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "analysis_history.db"


def connection():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    with connection() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.commit()


def save_prediction(filename, prediction):
    timestamp = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        cursor = db.execute(
            "INSERT INTO predictions (filename, class_name, confidence, created_at) VALUES (?, ?, ?, ?)",
            (filename or "unknown", prediction["class_name"], prediction["confidence"], timestamp),
        )
        db.commit()
        return {"id": cursor.lastrowid, "created_at": timestamp}


def get_history(limit=50):
    with connection() as db:
        rows = db.execute(
            "SELECT id, filename, class_name, confidence, created_at FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def clear_history():
    with connection() as db:
        db.execute("DELETE FROM predictions")
        db.commit()
