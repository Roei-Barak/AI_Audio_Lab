import hashlib
import json
import os
import secrets
from datetime import datetime

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def _load() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _hash(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    """Gradio auth callback — returns True if username+password are approved."""
    data = _load()
    user = data.get(username)
    if not user or not user.get("active", True):
        return False
    return _hash(password, user["salt"]) == user["hash"]


def add_user(username: str, password: str, note: str = ""):
    data = _load()
    salt = secrets.token_hex(16)
    data[username] = {
        "hash":   _hash(password, salt),
        "salt":   salt,
        "active": True,
        "added":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note":   note,
    }
    _save(data)


def remove_user(username: str):
    data = _load()
    if username in data:
        del data[username]
        _save(data)
        return True
    return False


def deactivate_user(username: str):
    """Block access without deleting the account."""
    data = _load()
    if username in data:
        data[username]["active"] = False
        _save(data)
        return True
    return False


def activate_user(username: str):
    data = _load()
    if username in data:
        data[username]["active"] = True
        _save(data)
        return True
    return False


def list_users() -> list:
    """Return sorted list of (username, active, added, note) tuples."""
    data = _load()
    return sorted(
        [
            (u, info.get("active", True), info.get("added", "—"), info.get("note", ""))
            for u, info in data.items()
        ],
        key=lambda x: x[0],
    )


def change_password(username: str, new_password: str):
    data = _load()
    if username not in data:
        return False
    salt = secrets.token_hex(16)
    data[username]["hash"] = _hash(new_password, salt)
    data[username]["salt"] = salt
    _save(data)
    return True
