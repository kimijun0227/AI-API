"""Small single-instance API server for the Unity AI client with original Cloudflare-bridge specs."""
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
from urllib.parse import urlparse, parse_qs
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("AI_DB_PATH", str(ROOT / "ai_usage.sqlite3")))
TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Seoul"))
USER_DAILY_LIMIT = int(os.getenv("USER_DAILY_LIMIT", "10"))
GLOBAL_DAILY_LIMIT = int(os.getenv("GLOBAL_DAILY_LIMIT", "1000"))
GLOBAL_RPM_LIMIT = int(os.getenv("GLOBAL_RPM_LIMIT", "20"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "30"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "800"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://groq.com"

DB_LOCK = threading.Lock()
INSTALL_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

VOICE_SYSTEM_PROMPT = """
당신은 친절하고 존중하는 한국어 AI 도우미이며 이름은 보이스입니다.
정체를 물으면 '저는 당신의 개인 언어 모델 보이스입니다.'라고 답하세요.
OpenAI가 만들었다고 말하거나 자신을 OpenAI의 언어 모델이라고 소개하지 마세요.
불확실한 내용은 확실하지 않다고 말하고, 사용자를 욕하거나 비하하지 말며 간결하게 답하세요.

반드시 아래 JSON 객체 하나만 응답하세요. 코드 펜스나 JSON 바깥 설명은 쓰지 마세요.
{
  "answer": "사용자에게 할 원래 답변",
  "tts": "음성으로 읽을 전처리 문장",
  "visuals": ["화면에 순서대로 보여줄 짧은 단계"]
}
""".strip()


def preprocess_tts_text(text: str) -> str:
    """Strip common display-only markup from the model's TTS field."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-+*•]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"[*_`#\[\]{}]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]", "", text)
    return " ".join(text.split())


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
            CREATE TABLE IF NOT EXISTS user_bonus (
                user_id TEXT NOT NULL, day TEXT NOT NULL, bonus_remaining INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, day)
            );
        """)
def reserve_request(user_id: str) -> tuple[str | None, int]:
    """Atomically check and reserve a request using original logic row factory."""
    today, minute = day_key(), minute_key()
    with DB_LOCK, connect_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT OR IGNORE INTO user_daily(user_id, day) VALUES (?, ?)", (user_id, today))
        conn.execute("INSERT OR IGNORE INTO global_daily(day) VALUES (?)", (today,))
        conn.execute("INSERT OR IGNORE INTO global_minute(minute) VALUES (?)", (minute,))
        conn.execute("INSERT OR IGNORE INTO user_bonus(user_id, day) VALUES (?, ?)", (user_id, today))
        
        user_row = conn.execute("SELECT used FROM user_daily WHERE user_id=? AND day=?", (user_id, today)).fetchone()
        global_row = conn.execute("SELECT used FROM global_daily WHERE day=?", (today,)).fetchone()
        minute_row = conn.execute("SELECT used FROM global_minute WHERE minute=?", (minute,)).fetchone()
        bonus_row = conn.execute("SELECT bonus_remaining FROM user_bonus WHERE user_id=? AND day=?", (user_id, today)).fetchone()

        user_used = user_row["used"] if user_row else 0
        global_used = global_row["used"] if global_row else 0
        minute_used = minute_row["used"] if minute_row else 0
        bonus_remaining = bonus_row["bonus_remaining"] if bonus_row else 0

        if global_used >= GLOBAL_DAILY_LIMIT:
            conn.rollback()
            return "global_limit", max(0, USER_DAILY_LIMIT - user_used)
        if minute_used >= GLOBAL_RPM_LIMIT:
            conn.rollback()
            return "rate_limit", max(0, USER_DAILY_LIMIT - user_used)
            
        if user_used >= USER_DAILY_LIMIT:
            if bonus_remaining > 0:
                conn.execute("UPDATE user_bonus SET bonus_remaining = bonus_remaining - 1 WHERE user_id=? AND day=?", (user_id, today))
                conn.commit()
                return None, bonus_remaining - 1
            else:
                conn.rollback()
                return "user_limit", 0

        conn.execute("UPDATE user_daily SET used=used+1 WHERE user_id=? AND day=?", (user_id, today))
        conn.execute("UPDATE global_daily SET used=used+1 WHERE day=?", (today,))
        conn.execute("UPDATE global_minute SET used=used+1 WHERE minute=?", (minute,))
        conn.commit()
        return None, max(0, USER_DAILY_LIMIT - user_used - 1)


def add_user_bonus(user_id: str, amount: int = 5) -> None:
    """Increase rewarded bonus question count for the given user."""
    today = day_key()
    with DB_LOCK, connect_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT OR IGNORE INTO user_bonus(user_id, day) VALUES (?, ?)", (user_id, today))
        conn.execute("UPDATE user_bonus SET bonus_remaining = bonus_remaining + ? WHERE user_id=? AND day=?", (amount, user_id, today))
        conn.commit()


def get_user_bonus(user_id: str) -> int:
    """Fetch current rewarded bonus count for the given user."""
    today = day_key()
    with connect_db() as conn:
        row = conn.execute("SELECT bonus_remaining FROM user_bonus WHERE user_id=? AND day=?", (user_id, today)).fetchone()
        return row["bonus_remaining"] if row else 0


def error_payload(code: str, message: str, remaining: int = -1, limit: int = USER_DAILY_LIMIT) -> dict:
    return {"error": code, "message": message, "remaining": remaining, "limit": limit}
class Handler(BaseHTTPRequestHandler):
    server_version = "PersonalAI/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.log_date_time_string()} {self.address_string()} {fmt % args}")

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_HEAD(self) -> None:
        """Handle Render load balancer health checks to avoid 501 errors."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        if parsed_url.path == "/health":
            self.send_json(200, {"ok": True})
            return
            
        elif parsed_url.path == "/admob/ssv":
            user_ids = query_params.get("user_id")
            if not user_ids:
                self.send_response(200)
                self.end_headers()
                return

            install_id = user_ids.strip()
            if not INSTALL_ID_RE.fullmatch(install_id):
                self.send_json(400, {"error": "invalid_user"})
                return

            try:
                add_user_bonus(install_id, 5)
                print(f"AdMob SSV Successfully credited 5 questions to {install_id}", flush=True)
                self.send_response(200)
                self.end_headers()
            except sqlite3.Error:
                self.send_json(500, {"error": "database_error"})
            return
            
        elif parsed_url.path == "/quota":
            install_id = self.headers.get("X-Install-Id", "").strip()
            if not INSTALL_ID_RE.fullmatch(install_id):
                self.send_json(400, {"error": "invalid_user"})
                return
            
            try:
                bonus = get_user_bonus(install_id)
                self.send_json(200, {"bonus_remaining": bonus})
            except sqlite3.Error:
                self.send_json(500, {"error": "database_error"})
            return

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
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self.send_json(400, error_payload("invalid_request", "요청 형식을 확인해 주세요."))
            return

        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(question, str) or not question.strip():
            self.send_json(400, error_payload("invalid_request", "질문을 입력해 주세요."))
            return
        question = question.strip()
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
                {"role": "system", "content": VOICE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.5,
        }
        req = Request(GROQ_URL, data=json.dumps(payload).encode("utf-8"), headers={
            "Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json",
            "User-Agent": "UnityAI-Groq-Server/1.0"
        }, method="POST")
        try:
            with urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            completion = result["choices"]["message"]["content"].strip()
            if completion.startswith("```"):
                completion = re.sub(r"^```(?:json)?\s*|\s*```$", "", completion, flags=re.IGNORECASE)
            try: parsed = json.loads(completion)
            except json.JSONDecodeError: parsed = {"answer": completion}
            answer = parsed.get("answer", completion)
            tts_text = parsed.get("tts", answer)
            visual_steps = parsed.get("visuals", [])
            visuals = [preprocess_tts_text(step)[:100] for step in visual_steps if isinstance(step, str)]
            visuals = [s for s in visuals if s][:4]
            self.send_json(200, {
                "answer": answer.strip(), "tts": preprocess_tts_text(tts_text), "visuals": visuals,
                "remaining": remaining, "limit": USER_DAILY_LIMIT
            })
        except Exception:
            self.send_json(502, error_payload("server_error", "답변을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.", remaining))


if __name__ == "__main__":
    init_db()
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    print(f"AI server listening on {host}:{port}; model={MODEL}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
