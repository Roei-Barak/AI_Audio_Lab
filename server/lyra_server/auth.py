"""Private accounts: no public signup. The first visitor creates the admin; everyone else joins with a one-time
invite link the admin copies and sends. Passwords are scrypt-hashed; sessions are an HttpOnly cookie (so
<video>/<audio> media requests are authenticated too) or a Bearer token.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

ROLES = ("admin", "editor", "viewer")
SESSION_DAYS = 30
COOKIE = "lyra_session"


class AuthError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32).hex()


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Accounts:
    def __init__(self, root: Path):
        self.file = root / "accounts.json"
        self.lock = threading.RLock()
        self.data = self._load()

    def _load(self) -> dict:
        if self.file.exists():
            try:
                return json.loads(self.file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"users": [], "sessions": {}, "invites": [], "resets": []}

    def _save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.file)
        try:
            os.chmod(self.file, 0o600)
        except OSError:
            pass

    @staticmethod
    def public(u: dict) -> dict:
        return {k: u[k] for k in ("id", "name", "user", "role", "createdAt")}

    def needs_setup(self) -> bool:
        return not self.data["users"]

    def _check_new(self, user: str, password: str) -> None:
        if not re.fullmatch(r"[a-z0-9._-]{3,32}", user or ""):
            raise AuthError(422, "שם משתמש: 3–32 תווים באנגלית קטנה, ספרות, נקודה או מקף.")
        if len(password or "") < 8:
            raise AuthError(422, "סיסמה: לפחות 8 תווים.")
        if any(u["user"] == user for u in self.data["users"]):
            raise AuthError(409, "שם המשתמש תפוס.")

    def _add_user(self, name: str, user: str, password: str, role: str) -> dict:
        salt = secrets.token_bytes(16)
        u = {"id": "u_" + secrets.token_hex(5), "name": name.strip() or user, "user": user, "role": role, "salt": salt.hex(), "pw": _hash(password, salt), "createdAt": time.time()}
        self.data["users"].append(u)
        return u

    def _session(self, uid: str) -> str:
        tok = secrets.token_urlsafe(32)
        now = time.time()
        self.data["sessions"] = {k: v for k, v in self.data["sessions"].items() if v["exp"] > now}
        self.data["sessions"][_sha(tok)] = {"uid": uid, "exp": now + SESSION_DAYS * 86400}
        return tok

    # ---- flows ----
    def setup(self, name: str, user: str, password: str) -> tuple[dict, str]:
        with self.lock:
            if not self.needs_setup():
                raise AuthError(409, "המערכת כבר הוגדרה.")
            self._check_new(user, password)
            u = self._add_user(name, user, password, "admin")
            tok = self._session(u["id"])
            self._save()
            return self.public(u), tok

    def login(self, user: str, password: str) -> tuple[dict, str]:
        with self.lock:
            u = next((x for x in self.data["users"] if x["user"] == (user or "").strip()), None)
            if not u or not hmac.compare_digest(u["pw"], _hash(password or "", bytes.fromhex(u["salt"]))):
                time.sleep(0.3)
                raise AuthError(401, "שם משתמש או סיסמה שגויים.")
            tok = self._session(u["id"])
            self._save()
            return self.public(u), tok

    def logout(self, tok: str | None) -> None:
        with self.lock:
            if tok and self.data["sessions"].pop(_sha(tok), None):
                self._save()

    def user_for(self, tok: str | None) -> dict | None:
        if not tok:
            return None
        s = self.data["sessions"].get(_sha(tok))
        if not s or s["exp"] < time.time():
            return None
        u = next((x for x in self.data["users"] if x["id"] == s["uid"]), None)
        return self.public(u) if u else None

    # ---- users ----
    def users(self) -> list[dict]:
        return [self.public(u) for u in self.data["users"]]

    def set_role(self, actor: dict, uid: str, role: str) -> dict:
        if role not in ROLES:
            raise AuthError(422, "תפקיד לא מוכר.")
        with self.lock:
            u = self._find(uid)
            if u["role"] == "admin" and role != "admin" and sum(x["role"] == "admin" for x in self.data["users"]) == 1:
                raise AuthError(409, "חייב להישאר לפחות מנהל אחד.")
            u["role"] = role
            self._save()
            return self.public(u)

    def remove(self, actor: dict, uid: str) -> None:
        with self.lock:
            if uid == actor["id"]:
                raise AuthError(409, "אי אפשר להסיר את עצמך.")
            self._find(uid)
            self.data["users"] = [x for x in self.data["users"] if x["id"] != uid]
            self.data["sessions"] = {k: v for k, v in self.data["sessions"].items() if v["uid"] != uid}
            self._save()

    def _find(self, uid: str) -> dict:
        u = next((x for x in self.data["users"] if x["id"] == uid), None)
        if not u:
            raise AuthError(404, "המשתמש לא נמצא.")
        return u

    # ---- one-time invites & resets ----
    def invite(self, name: str, role: str, days: int) -> tuple[dict, str]:
        if role not in ROLES:
            raise AuthError(422, "תפקיד לא מוכר.")
        with self.lock:
            tok = secrets.token_urlsafe(9)
            inv = {"id": "i_" + secrets.token_hex(4), "name": name.strip(), "role": role, "hash": _sha(tok), "expiresAt": time.time() + max(1, min(days, 30)) * 86400, "used": False, "createdAt": time.time()}
            self.data["invites"].insert(0, inv)
            self._save()
            return self._pub_inv(inv), tok

    @staticmethod
    def _pub_inv(i: dict) -> dict:
        return {k: i[k] for k in ("id", "name", "role", "expiresAt", "used", "createdAt")}

    def invites(self) -> list[dict]:
        return [self._pub_inv(i) for i in self.data["invites"]]

    def revoke(self, iid: str) -> None:
        with self.lock:
            self.data["invites"] = [i for i in self.data["invites"] if i["id"] != iid]
            self._save()

    def accept(self, token: str, user: str, password: str) -> tuple[dict, str]:
        with self.lock:
            inv = next((i for i in self.data["invites"] if hmac.compare_digest(i["hash"], _sha(token.strip()))), None)
            if not inv:
                raise AuthError(404, "הקישור לא תקין או שבוטל.")
            if inv["used"]:
                raise AuthError(410, "הקישור כבר שומש. בקש מהמנהל קישור חדש.")
            if inv["expiresAt"] < time.time():
                raise AuthError(410, "תוקף הקישור פג.")
            self._check_new(user, password)
            u = self._add_user(inv["name"], user, password, inv["role"])
            inv["used"] = True
            tok = self._session(u["id"])
            self._save()
            return self.public(u), tok

    def reset_link(self, uid: str) -> str:
        with self.lock:
            self._find(uid)
            tok = secrets.token_urlsafe(12)
            self.data["resets"] = [r for r in self.data["resets"] if r["uid"] != uid and r["exp"] > time.time()] + [{"uid": uid, "hash": _sha(tok), "exp": time.time() + 86400}]
            self._save()
            return tok

    def reset(self, token: str, password: str) -> tuple[dict, str]:
        with self.lock:
            r = next((x for x in self.data["resets"] if hmac.compare_digest(x["hash"], _sha(token.strip())) and x["exp"] > time.time()), None)
            if not r:
                raise AuthError(404, "קישור האיפוס לא תקין או שפג תוקפו.")
            if len(password or "") < 8:
                raise AuthError(422, "סיסמה: לפחות 8 תווים.")
            u = self._find(r["uid"])
            salt = secrets.token_bytes(16)
            u["salt"], u["pw"] = salt.hex(), _hash(password, salt)
            self.data["resets"].remove(r)
            self.data["sessions"] = {k: v for k, v in self.data["sessions"].items() if v["uid"] != u["id"]}
            tok = self._session(u["id"])
            self._save()
            return self.public(u), tok
