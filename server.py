"""Small single-instance API server for the Unity AI client."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import asyncio
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import edge_tts


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("AI_DB_PATH", str(ROOT / "ai_usage.sqlite3")))
TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Seoul"))
USER_DAILY_LIMIT = int(os.getenv("USER_DAILY_LIMIT", "10"))
GLOBAL_DAILY_LIMIT = int(os.getenv("GLOBAL_DAILY_LIMIT", "1000"))
GLOBAL_RPM_LIMIT = int(os.getenv("GLOBAL_RPM_LIMIT", "20"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "30"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "800"))
TTS_DAILY_LIMIT = int(os.getenv("TTS_DAILY_LIMIT", "10"))
TTS_GLOBAL_DAILY_LIMIT = int(os.getenv("TTS_GLOBAL_DAILY_LIMIT", "1000"))
TTS_GLOBAL_RPM_LIMIT = int(os.getenv("TTS_GLOBAL_RPM_LIMIT", "20"))
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "2000"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TTS_VOICE = "ko-KR-InJoonNeural"
# These characters should be ignored by speech synthesis. Replace them with
# spaces so adjacent words do not accidentally run together.
TTS_IGNORED_CHARS = str.maketrans({char: " " for char in ".,!?/@*^$~`<>"})
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

TTS 전처리 규칙:
원래 답변의 의미와 문장 순서를 바꾸지 말고, 답변에 없는 내용을 더하지 마세요.
굵게, 기울임, 제목 표시, 불릿, 번호 등 마크다운 표시는 제거하고 내용은 유지하세요.
코드는 꼭 읽어야 하는 경우가 아니면 사람이 이해할 자연스러운 말로 바꾸세요.
별표, 해시, 백틱, 대괄호, 중괄호 같은 화면 표시 기호와 이모지, 장식 문자는 읽지 않게 제거하세요.
링크는 URL 전체 대신 링크 이름이나 자연스러운 설명만 남기세요.
수식이나 프로그래밍 기호가 뜻을 가지면 삭제하지 말고 말로 이해되게 표현하세요.
자연스러운 띄어쓰기와 문장부호만 남기세요. tts 필드에는 순수한 음성용 문장만 넣으세요.

시각 설명 규칙:
답변 내용을 그림, 비교, 순서, 관계로 보여주면 이해가 쉬울 때만 visuals에 2~4개의 짧은 단계로 적으세요.
단계는 답변에서 설명한 정보만 간결하게 뽑고, 새 사실을 추가하지 마세요.
단계별 문구는 화면에 카드로 차례로 나타나고 화살표로 이어집니다.
시각 설명이 필요하지 않으면 visuals는 빈 배열 []로 두세요.
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
            CREATE TABLE IF NOT EXISTS tts_user_daily (
                user_id TEXT NOT NULL, day TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, day)
            );
            CREATE TABLE IF NOT EXISTS tts_global_daily (
                day TEXT PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS tts_global_minute (
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


def reserve_tts(user_id: str) -> str | None:
    """Bound Azure speech usage per install, globally per day, and per minute."""
    today, minute = day_key(), minute_key()
    with DB_LOCK, connect_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT OR IGNORE INTO tts_user_daily(user_id, day) VALUES (?, ?)", (user_id, today))
        conn.execute("INSERT OR IGNORE INTO tts_global_daily(day) VALUES (?)", (today,))
        conn.execute("INSERT OR IGNORE INTO tts_global_minute(minute) VALUES (?)", (minute,))
        user_used = conn.execute("SELECT used FROM tts_user_daily WHERE user_id=? AND day=?", (user_id, today)).fetchone()[0]
        global_used = conn.execute("SELECT used FROM tts_global_daily WHERE day=?", (today,)).fetchone()[0]
        minute_used = conn.execute("SELECT used FROM tts_global_minute WHERE minute=?", (minute,)).fetchone()[0]
        if user_used >= TTS_DAILY_LIMIT:
            conn.rollback()
            return "user_limit"
        if global_used >= TTS_GLOBAL_DAILY_LIMIT:
            conn.rollback()
            return "global_limit"
        if minute_used >= TTS_GLOBAL_RPM_LIMIT:
            conn.rollback()
            return "rate_limit"
        conn.execute("UPDATE tts_user_daily SET used=used+1 WHERE user_id=? AND day=?", (user_id, today))
        conn.execute("UPDATE tts_global_daily SET used=used+1 WHERE day=?", (today,))
        conn.execute("UPDATE tts_global_minute SET used=used+1 WHERE minute=?", (minute,))
        conn.commit()
        return None


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
            self.send_json(200, {"ok": True, "tts_configured": True, "tts_provider": "edge-tts"})
        else:
            self.send_json(404, {"error": "not_found", "message": "요청한 주소를 찾을 수 없습니다."})

    def do_POST(self) -> None:
        if self.path == "/speak":
            self.do_speak()
            return
        if self.path == "/transcribe":
            self.do_transcribe()
            return
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
                {"role": "system", "content": VOICE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.5,
        }
        req = Request(GROQ_URL, data=json.dumps(payload).encode("utf-8"), headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            # Groq's Cloudflare edge can reject urllib's default Python-urllib UA
            # with HTTP 403 / error code 1010. Identify this API client explicitly.
            "User-Agent": "UnityAI-Groq-Server/1.0",
        }, method="POST")
        try:
            with urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            completion = result["choices"][0]["message"]["content"]
            if not isinstance(completion, str) or not completion.strip():
                raise ValueError("empty completion")
            structured = completion.strip()
            if structured.startswith("```"):
                structured = re.sub(r"^```(?:json)?\s*|\s*```$", "", structured, flags=re.IGNORECASE)
            try:
                parsed = json.loads(structured)
            except json.JSONDecodeError:
                parsed = {"answer": completion.strip()}
            answer = parsed.get("answer", completion.strip()) if isinstance(parsed, dict) else completion.strip()
            tts_text = parsed.get("tts", answer) if isinstance(parsed, dict) else answer
            visual_steps = parsed.get("visuals", []) if isinstance(parsed, dict) else []
            if not isinstance(answer, str) or not answer.strip():
                answer = completion.strip()
            if not isinstance(tts_text, str) or not tts_text.strip():
                tts_text = answer
            if not isinstance(visual_steps, list):
                visual_steps = []
            visuals = [preprocess_tts_text(step)[:100] for step in visual_steps if isinstance(step, str)]
            visuals = [step for step in visuals if step][:4]
            self.send_json(200, {
                "answer": answer.strip(),
                "tts": preprocess_tts_text(tts_text),
                "visuals": visuals,
                "remaining": remaining,
                "limit": USER_DAILY_LIMIT,
            })
        except HTTPError as exc:
            if exc.code == 429:
                self.send_json(429, error_payload("rate_limit", "Groq 요청 한도에 도달했습니다. 잠시 후 다시 시도해 주세요.", remaining))
            else:
                # Capture a short, sanitized provider response so permission errors
                # can be diagnosed without logging the request, headers, or API key.
                detail = ""
                raw_body = ""
                try:
                    raw_body = exc.read().decode("utf-8", errors="replace")
                    parsed = json.loads(raw_body)
                    provider_error = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
                    if isinstance(provider_error, dict):
                        code = str(provider_error.get("code") or provider_error.get("type") or "")
                        message = str(provider_error.get("message") or provider_error.get("detail") or "")
                        if code:
                            detail += f" [{code}]"
                        if message:
                            detail += f" {message}"
                    elif provider_error:
                        detail = f" {provider_error}"
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
                    detail = f" {raw_body}"
                if not detail and raw_body:
                    detail = f" {raw_body}"
                if GROQ_API_KEY:
                    detail = detail.replace(GROQ_API_KEY, "[redacted]")
                detail = " ".join(detail.split())[:360]
                request_id = exc.headers.get("x-request-id", "") if exc.headers else ""
                if request_id and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
                    detail += f" (request_id={request_id})"
                print(f"Groq HTTP {exc.code}{detail}", flush=True)
                self.send_json(502, error_payload("server_error", f"Groq HTTP {exc.code}{detail}", remaining))
        except (URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            # Keep the Unity response generic, while giving Render logs a safe
            # failure category to distinguish provider, schema, and empty-output issues.
            print(f"Groq completion handling failed: {type(exc).__name__}", flush=True)
            self.send_json(502, error_payload("server_error", "AI 답변을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.", remaining))

    def do_speak(self) -> None:
        install_id = self.headers.get("X-Install-Id", "").strip()
        if not INSTALL_ID_RE.fullmatch(install_id):
            self.send_json(400, error_payload("invalid_user", "앱 식별 정보가 올바르지 않습니다."))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 32_768:
                raise ValueError("invalid content length")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, error_payload("invalid_request", "요청 형식을 확인해 주세요."))
            return

        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            self.send_json(400, error_payload("invalid_request", "읽을 문장이 비어 있습니다."))
            return
        text = text.translate(TTS_IGNORED_CHARS)
        text = " ".join(text.split())
        if not text:
            self.send_json(400, error_payload("invalid_request", "읽을 문장에 음성으로 읽을 내용이 없습니다."))
            return
        if len(text) > MAX_TTS_CHARS:
            self.send_json(400, error_payload("too_long", f"음성으로 읽을 문장은 {MAX_TTS_CHARS}자까지만 가능합니다."))
            return

        try:
            denial = reserve_tts(install_id)
        except sqlite3.Error:
            self.send_json(503, error_payload("server_error", "사용량 저장소에 문제가 생겼습니다."))
            return
        if denial == "user_limit":
            self.send_json(429, error_payload("tts_user_limit", "오늘 음성 재생 횟수를 모두 사용했습니다."))
            return
        if denial == "global_limit":
            self.send_json(429, error_payload("tts_global_limit", "오늘 음성 사용량이 모두 소진되었습니다."))
            return
        if denial == "rate_limit":
            self.send_json(429, error_payload("rate_limit", "음성 요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."))
            return

        try:
            audio = asyncio.run(self.generate_edge_speech(text))
            if not audio:
                raise ValueError("empty speech response")
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(audio)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(audio)
        except HTTPError as exc:
            print(f"Edge TTS HTTP error {exc.code}", flush=True)
            self.send_json(502, error_payload("tts_provider_error", "음성 서비스 요청이 실패했습니다. 잠시 후 다시 시도해 주세요."))
        except Exception as exc:
            print(f"Edge TTS request failed: {type(exc).__name__}", flush=True)
            self.send_json(502, error_payload("tts_provider_error", "음성을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요."))

    def do_transcribe(self) -> None:
        if not GROQ_API_KEY:
            self.send_json(503, error_payload("server_error", "서버에 Groq API 키가 설정되지 않았습니다."))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 44 or length > 5 * 1024 * 1024:
                raise ValueError("invalid audio length")
            audio = self.rfile.read(length)
            if len(audio) != length or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
                raise ValueError("invalid wav audio")
        except (ValueError, OverflowError):
            self.send_json(400, error_payload("invalid_audio", "녹음 파일이 올바르지 않습니다."))
            return

        boundary = "----UnityVoiceBoundary7MA4YWxkTrZu0gW"
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3\r\n".encode("ascii"),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nko\r\n".encode("ascii"),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n".encode("ascii"),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"voice.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode("ascii"),
            audio,
            f"\r\n--{boundary}--\r\n".encode("ascii"),
        ])
        req = Request("https://api.groq.com/openai/v1/audio/transcriptions", data=body, headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "UnityAI-Groq-Server/1.0",
        }, method="POST")
        try:
            with urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
            transcript = result.get("text", "") if isinstance(result, dict) else ""
            if not isinstance(transcript, str) or not transcript.strip():
                self.send_json(422, error_payload("empty_transcription", "말한 내용을 알아듣지 못했어요. 다시 말해 주세요."))
                return
            self.send_json(200, {"text": transcript.strip()})
        except HTTPError as exc:
            detail = ""
            raw_body = ""
            try:
                raw_body = exc.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw_body)
                provider_error = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
                if isinstance(provider_error, dict):
                    code = str(provider_error.get("code") or provider_error.get("type") or "")
                    message = str(provider_error.get("message") or provider_error.get("detail") or "")
                    detail = " ".join(part for part in (f"[{code}]" if code else "", message) if part)
                elif provider_error:
                    detail = str(provider_error)
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
                detail = raw_body
            if GROQ_API_KEY:
                detail = detail.replace(GROQ_API_KEY, "[redacted]")
            detail = " ".join(detail.split())[:320]
            request_id = exc.headers.get("x-request-id", "") if exc.headers else ""
            if request_id and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
                detail += f" (request_id={request_id})"
            print(f"Groq transcription HTTP {exc.code}{(' ' + detail) if detail else ''}", flush=True)
            message = f"음성 인식 서비스 오류 (Groq HTTP {exc.code}{(' ' + detail) if detail else ''})"
            self.send_json(502, error_payload("transcription_error", message))
        except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(502, error_payload("transcription_error", "음성을 글자로 바꾸지 못했습니다. 잠시 후 다시 시도해 주세요."))

    @staticmethod
    async def generate_edge_speech(text: str) -> bytes:
        communicate = edge_tts.Communicate(text, TTS_VOICE, rate="-15%")
        chunks = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)


if __name__ == "__main__":
    init_db()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"AI server listening on {host}:{port}; model={MODEL}; timezone={TZ}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
