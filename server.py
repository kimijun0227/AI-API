"""Small single-instance API server for the Unity AI client."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("AI_DB_PATH", str(ROOT / "ai_usage.sqlite3")))
TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Seoul"))
USER_DAILY_LIMIT = int(os.getenv("USER_DAILY_LIMIT", "10"))
GLOBAL_DAILY_LIMIT = int(os.getenv("GLOBAL_DAILY_LIMIT", "1000"))
GLOBAL_RPM_LIMIT = int(os.getenv("GLOBAL_RPM_LIMIT", "20"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "30"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "160"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DB_LOCK = threading.Lock()
INSTALL_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def day_key() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")


def minute_key() -> str:
    return str(int(time.time() // 60))


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS user_daily (
                user_id TEXT NOT NULL, day TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, day)
            );
            CREATE TABLE IF NOT EXISTS global_daily (
                day TEXT PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS global_minute (
                minute TEXT PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0
            );
        """)


def reserve_request(user_id: str) -> tuple[str | None, int]:
    """Atomically check and reserve a request before calling Groq."""
    today, minute = day_key(), minute_key()
    with DB_LOCK, connect_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT OR IGNORE INTO user_daily(user_id, day) VALUES (?, ?)", (user_id, today))
        conn.execute("INSERT OR IGNORE INTO global_daily(day) VALUES (?)", (today,))
        conn.execute("INSERT OR IGNORE INTO global_minute(minute) VALUES (?)", (minute,))
        user_used = conn.execute("SELECT used FROM user_daily WHERE user_id=? AND day=?", (user_id, today)).fetchone()[0]
        global_used = conn.execute("SELECT used FROM global_daily WHERE day=?", (today,)).fetchone()[0]
        minute_used = conn.execute("SELECT used FROM global_minute WHERE minute=?", (minute,)).fetchone()[0]

        if global_used >= GLOBAL_DAILY_LIMIT:
            conn.rollback()
            return "global_limit", max(0, USER_DAILY_LIMIT - user_used)
        if minute_used >= GLOBAL_RPM_LIMIT:
            conn.rollback()
            return "rate_limit", max(0, USER_DAILY_LIMIT - user_used)
        if user_used >= USER_DAILY_LIMIT:
            conn.rollback()
            return "user_limit", 0

        conn.execute("UPDATE user_daily SET used=used+1 WHERE user_id=? AND day=?", (user_id, today))
        conn.execute("UPDATE global_daily SET used=used+1 WHERE day=?", (today,))
        conn.execute("UPDATE global_minute SET used=used+1 WHERE minute=?", (minute,))
        conn.commit()
        return None, max(0, USER_DAILY_LIMIT - user_used - 1)


def error_payload(code: str, message: str, remaining: int = -1, limit: int = USER_DAILY_LIMIT) -> dict:
    return {"error": code, "message": message, "remaining": remaining, "limit": limit}


class Handler(BaseHTTPRequestHandler):
    server_version = "PersonalAI/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Avoid logging user questions, install IDs, or provider credentials.
        print(f"{self.log_date_time_string()} {self.address_string()} {fmt % args}")

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "not_found", "message": "요청한 주소를 찾을 수 없습니다."})

    def do_POST(self) -> None:
        if self.path != "/ask":
            self.send_json(404, {"error": "not_found", "message": "요청한 주소를 찾을 수 없습니다."})
            return
        if not GROQ_API_KEY:
            self.send_json(503, error_payload("server_error", "서버에 Groq API 키가 설정되지 않았습니다."))
            return

        install_id = self.headers.get("X-Install-Id", "").strip()
        if not INSTALL_ID_RE.fullmatch(install_id):
            self.send_json(400, error_payload("invalid_user", "앱 식별 정보가 올바르지 않습니다."))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16_384:
                raise ValueError("invalid content length")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, error_payload("invalid_request", "요청 형식을 확인해 주세요."))
            return

        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(question, str) or not question.strip():
            self.send_json(400, error_payload("invalid_request", "질문을 입력해 주세요."))
            return
        question = question.strip()
        # Python len counts Unicode code points, matching AITextUtil.CountChars.
        if len(question) > MAX_QUESTION_CHARS:
            self.send_json(400, error_payload("too_long", f"질문은 {MAX_QUESTION_CHARS}자까지만 보낼 수 있어요."))
            return

        try:
            denial, remaining = reserve_request(install_id)
        except sqlite3.Error:
            self.send_json(503, error_payload("server_error", "사용량 저장소에 문제가 생겼습니다."))
            return
        if denial == "global_limit":
            self.send_json(429, error_payload("global_limit", "오늘 AI 사용량이 모두 소진되었습니다. 내일 다시 이용해주세요.", remaining))
            return
        if denial == "rate_limit":
            self.send_json(429, error_payload("rate_limit", "요청이 너무 많아요. 잠시 후 다시 시도해 주세요.", remaining))
            return
        if denial == "user_limit":
            self.send_json(429, error_payload("user_limit", "오늘 사용할 수 있는 질문을 모두 사용했어요.", 0))
            return

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "당신은 친절하고 존중하는 한국어 AI 도우미입니다. 모르는 내용이나 불확실한 내용은 확실하지 않다고 말하고 사실처럼 꾸며내지 마세요. 사용자를 욕하거나 비하하지 마세요. 간결하게 답하세요."},
                {"role": "user", "content": question},
            ],
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.5,
        }
        req = Request(GROQ_URL, data=json.dumps(payload).encode("utf-8"), headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }, method="POST")
        try:
            with urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = result["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("empty completion")
            self.send_json(200, {"answer": answer.strip(), "remaining": remaining, "limit": USER_DAILY_LIMIT})
        except HTTPError as exc:
            if exc.code == 429:
                self.send_json(429, error_payload("rate_limit", "Groq 요청 한도에 도달했습니다. 잠시 후 다시 시도해 주세요.", remaining))
            else:
                print(f"Groq returned HTTP {exc.code}", flush=True)
                self.send_json(502, error_payload("server_error", f"AI 제공자 통신 오류 (Groq HTTP {exc.code})", remaining))
        except (URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError):
            self.send_json(502, error_payload("server_error", "AI 답변을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.", remaining))


if __name__ == "__main__":
    init_db()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"AI server listening on {host}:{port}; model={MODEL}; timezone={TZ}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
