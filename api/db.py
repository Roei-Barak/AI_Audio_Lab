"""SQLite user store — stdlib sqlite3, no ORM."""
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

WORK_DIR = os.environ.get("KARAOKE_OUTPUT_DIR", "Karaoke_Output")
DB_PATH  = Path(WORK_DIR) / "users.db"


@dataclass
class User:
    id: int
    username: str
    password_hash: str
    role: str          # 'admin' | 'user'
    created_at: datetime


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'user',
                created_at    TEXT    NOT NULL
            )
        """)


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        password_hash=row["password_hash"],
        role=row["role"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def create_user(username: str, password_hash: str, role: str = "user") -> User:
    now = datetime.utcnow().isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
            (username, password_hash, role, now),
        )
        return get_user_by_id(cur.lastrowid)  # type: ignore[arg-type]


def get_user_by_id(user_id: int) -> "User | None":
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return _row_to_user(row) if row else None


def get_user_by_username(username: str) -> "User | None":
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return _row_to_user(row) if row else None


def list_users() -> list[User]:
    with _conn() as c:
        return [_row_to_user(r) for r in c.execute("SELECT * FROM users ORDER BY id")]


def delete_user(user_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM users WHERE id=?", (user_id,))


def update_password(user_id: int, new_hash: str) -> None:
    with _conn() as c:
        c.execute("UPDATE users SET password_hash=? WHERE id=?", (new_hash, user_id))
