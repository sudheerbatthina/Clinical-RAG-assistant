import sqlite3
import uuid
from datetime import datetime, timedelta
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
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                message_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS query_logs (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                question TEXT NOT NULL,
                rewritten_question TEXT,
                answer_preview TEXT,
                sources_count INTEGER,
                latency_ms INTEGER,
                from_cache INTEGER DEFAULT 0,
                faithfulness_score INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hallucination_flags (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                faithfulness_score INTEGER NOT NULL,
                flagged_claims TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                created_at TEXT NOT NULL,
                last_login TEXT
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

def save_feedback(chat_id: str, message_id: str, question: str,
                  answer: str, rating: int) -> dict:
    now = datetime.utcnow().isoformat()
    fid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,?,?)",
            (fid, chat_id, message_id, question, answer, rating, now)
        )
    return {"id": fid, "rating": rating}

def save_query_log(chat_id: str, question: str, rewritten: str,
                   answer_preview: str, sources_count: int,
                   latency_ms: int, from_cache: bool,
                   faithfulness_score: int | None) -> None:
    now = datetime.utcnow().isoformat()
    lid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO query_logs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (lid, chat_id, question, rewritten, answer_preview,
             sources_count, latency_ms, int(from_cache),
             faithfulness_score, now)
        )

def create_user(username: str, password: str,
                email: str = None, role: str = "member") -> dict:
    from .auth import hash_password
    now = datetime.utcnow().isoformat()
    uid = str(uuid.uuid4())
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                (uid, username, email, hash_password(password), role, now, None),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists")
    return {"id": uid, "username": username, "role": role}


def get_user_by_username(username: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
    return dict(row) if row else None


def update_last_login(user_id: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (datetime.utcnow().isoformat(), user_id),
        )


def list_users() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, email, role, created_at, last_login "
            "FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_analytics() -> dict:
    """Return usage analytics aggregated from query_logs and feedback."""
    with get_conn() as conn:
        total_queries = conn.execute(
            "SELECT COUNT(*) FROM query_logs").fetchone()[0]
        cache_hits = conn.execute(
            "SELECT COUNT(*) FROM query_logs WHERE from_cache=1").fetchone()[0]
        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) FROM query_logs "
            "WHERE from_cache=0").fetchone()[0]
        avg_faithfulness = conn.execute(
            "SELECT AVG(faithfulness_score) FROM query_logs "
            "WHERE faithfulness_score IS NOT NULL").fetchone()[0]
        queries_by_day = conn.execute(
            """SELECT substr(created_at,1,10) as day, COUNT(*) as count
               FROM query_logs
               WHERE created_at >= datetime('now','-7 days')
               GROUP BY day ORDER BY day ASC"""
        ).fetchall()
        feedback_good = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating=1").fetchone()[0]
        feedback_bad = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating=-1").fetchone()[0]
    return {
        "total_queries": total_queries,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total_queries * 100, 1) if total_queries else 0,
        "avg_latency_ms": round(avg_latency or 0, 0),
        "avg_faithfulness": round(avg_faithfulness or 0, 1),
        "queries_by_day": [{"day": r[0], "count": r[1]} for r in queries_by_day],
        "feedback_good": feedback_good,
        "feedback_bad": feedback_bad,
    }


def purge_old_logs(days: int = 7) -> int:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM query_logs WHERE created_at < ?", (cutoff,)
        )
        deleted = cur.rowcount
        conn.execute(
            "DELETE FROM hallucination_flags WHERE created_at < ?", (cutoff,)
        )
    return deleted
