import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from .config import STORAGE_DIR

DB_PATH = Path(STORAGE_DIR) / "db.sqlite"

def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL
            );
        """)

def create_chat(title="New chat") -> dict:
    now = datetime.utcnow().isoformat()
    chat_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO chats VALUES (?,?,?,?)", (chat_id, title, now, now))
    return {"id": chat_id, "title": title, "created_at": now}

def list_chats() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]

def get_chat(chat_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    return dict(row) if row else None

def delete_chat(chat_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))

def add_message(chat_id: str, role: str, content: str, sources: str = None) -> dict:
    import json
    now = datetime.utcnow().isoformat()
    msg_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("UPDATE chats SET updated_at=?, title=CASE WHEN title='New chat' AND ?='user' THEN substr(?,1,40) ELSE title END WHERE id=?",
                     (now, role, content, chat_id))
        conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                     (msg_id, chat_id, role, content, sources, now))
    return {"id": msg_id, "chat_id": chat_id, "role": role, "content": content}

def get_messages(chat_id: str) -> list:
    import json
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC", (chat_id,)).fetchall()
    return [dict(r) for r in rows]
