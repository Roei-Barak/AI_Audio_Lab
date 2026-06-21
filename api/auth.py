"""JWT auth + FastAPI router.

Set KARAOKE_AUTH_MODE=none to disable auth (Standalone variant).
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from api.db import (
    User, create_user, delete_user, get_user_by_id,
    get_user_by_username, list_users, update_password,
)

AUTH_MODE  = os.environ.get("KARAOKE_AUTH_MODE", "required")   # 'required' | 'none'
_DEFAULT_SECRET = "change-me-in-production-please"
JWT_SECRET = os.environ.get("KARAOKE_JWT_SECRET", _DEFAULT_SECRET)
JWT_ALG    = "HS256"
JWT_EXP_H  = 24 * 7   # 1 week

if AUTH_MODE != "none" and JWT_SECRET == _DEFAULT_SECRET:
    import sys
    print(
        "[SECURITY WARNING] KARAOKE_JWT_SECRET is not set — using the insecure default. "
        "Anyone who reads the source code can forge admin tokens. "
        "Set KARAOKE_JWT_SECRET to a long random string in production!",
        file=sys.stderr, flush=True,
    )


# ── Passwords ─────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── Tokens ────────────────────────────────────────────────────────────────────

def create_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub":      str(user_id),
        "username": username,
        "role":     role,
        "exp":      datetime.now(timezone.utc) + timedelta(hours=JWT_EXP_H),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def verify_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        return None


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_current_user(authorization: str = Header(default="")) -> Optional[User]:
    if AUTH_MODE == "none":
        return User(id=0, username="local", password_hash="", role="admin",
                    created_at=datetime.now(timezone.utc))
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = verify_token(authorization[7:])
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = get_user_by_id(int(payload["sub"]))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return user


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter()


@router.post("/login")
async def login(body: LoginRequest):
    user = get_user_by_username(body.username)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="שם משתמש או סיסמה שגויים")
    token = create_token(user.id, user.username, user.role)
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, username=user.username, role=user.role, created_at=user.created_at)


@router.get("/users")
async def get_users(admin: User = Depends(require_admin)):
    return [UserOut(id=u.id, username=u.username, role=u.role, created_at=u.created_at)
            for u in list_users()]


@router.post("/users", status_code=201)
async def add_user(body: CreateUserRequest, admin: User = Depends(require_admin)):
    if get_user_by_username(body.username):
        raise HTTPException(status_code=409, detail="שם משתמש כבר קיים")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="סיסמה חייבת להכיל לפחות 6 תווים")
    u = create_user(body.username, hash_password(body.password), body.role)
    return UserOut(id=u.id, username=u.username, role=u.role, created_at=u.created_at)


@router.delete("/users/{user_id}", status_code=204)
async def remove_user(user_id: int, admin: User = Depends(require_admin)):
    if admin.id == user_id:
        raise HTTPException(status_code=400, detail="לא ניתן למחוק את עצמך")
    delete_user(user_id)


@router.post("/change-password")
async def change_password(body: ChangePasswordRequest, user: User = Depends(get_current_user)):
    if AUTH_MODE == "none":
        raise HTTPException(status_code=400, detail="Auth disabled in standalone mode")
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="סיסמה נוכחית שגויה")
    if len(body.new_password) < 6:
        raise HTTPException(status_code=400, detail="סיסמה חייבת להכיל לפחות 6 תווים")
    update_password(user.id, hash_password(body.new_password))
    return {"ok": True}
