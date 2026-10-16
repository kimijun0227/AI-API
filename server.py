"""Voice AI API server for Unity.
Endpoints:
  GET  /health
  GET  /quota
  GET  /admob/ssv?user_id=<install-id>
  POST /ask          JSON: {"question": "..."}
  POST /transcribe   raw WAV bytes (Content-Type: audio/wav), returns {"text": "..."}
  POST /speak        JSON {"text": "..."}, returns Cloudflare MeloTTS MP3 audio
"""

from __future__ import annotations

import base64
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
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("AI_DB_PATH", str(ROOT / "ai_usage.sqlite3")))
TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Seoul"))

USER_DAILY_LIMIT = int(os.getenv("USER_DAILY_LIMIT", "10"))
GLOBAL_DAILY_LIMIT = int(os.getenv("GLOBAL_DAILY_LIMIT", "1000"))
GLOBAL_RPM_LIMIT = int(os.getenv("GLOBAL_RPM_LIMIT", "20"))
# The old default of 30 characters rejected many normal questions.
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "800"))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
MAX_TTS_TEXT_CHARS = int(os.getenv("MAX_TTS_TEXT_CHARS", "3000"))

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TRANSCRIPTION_MODEL = os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_TTS_MODEL = "@cf/myshell-ai/melotts"
CLOUDFLARE_TTS_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_TTS_MODEL}"
    if CLOUDFLARE_ACCOUNT_ID else ""
)

DB_LOCK = threading.Lock()
INSTALL_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

VOICE_SYSTEM_PROMPT = """
당신은 친절하고 존중하는 한국어 AI 도우미이며 이름은 보이스입니다.
정체를 물으면 '저는 당신의 개인 언어 모델 보이스입니다.'라고 답하세요.
불확실한 내용은 확실하지 않다고 말하고, 사용자를 욕하거나 비하하지 말며 간결하게 답하세요.

반드시 아래 형식의 JSON 객체 하나만 응답하세요. 코드 펜스나 JSON 바깥 설명은 쓰지 마세요.
{
  "answer": "사용자에게 할 원래 답변",
  "tts": "음성으로 읽을 전처리 문장",
  "visuals": ["화면에 순서대로 보여줄 짧은 단계"]
}
""".strip()


def preprocess_tts_text(text: str) -> str:
    """Remove display-only markup from text that will be spoken aloud."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-+*•]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"[*_`#\[\]{}]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)
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
    """Atomically check and reserve a question against existing usage limits."""
    today, minute = day_key(), minute_key()
    with DB_LOCK, connect_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT OR IGNORE INTO user_daily(user_id, day) VALUES (?, ?)", (user_id, today))
        conn.execute("INSERT OR IGNORE INTO global_daily(day) VALUES (?)", (today,))
        conn.execute("INSERT OR IGNORE INTO global_minute(minute) VALUES (?)", (minute,))
        conn.execute("INSERT OR IGNORE INTO user_bonus(user_id, day) VALUES (?, ?)", (user_id, today))

        user_row = conn.execute(
            "SELECT used FROM user_daily WHERE user_id=? AND day=?", (user_id, today)
        ).fetchone()
        global_row = conn.execute("SELECT used FROM global_daily WHERE day=?", (today,)).fetchone()
        minute_row = conn.execute("SELECT used FROM global_minute WHERE minute=?", (minute,)).fetchone()
        bonus_row = conn.execute(
            "SELECT bonus_remaining FROM user_bonus WHERE user_id=? AND day=?", (user_id, today)
        ).fetchone()

        user_used = user_row["used"] if user_row else 0
        global_used = global_row["used"] if global_row else 0
        minute_used = minute_row["used"] if minute_row else 0
        bonus_remaining = bonus_row["bonus_remaining"] if bonus_row else 0

        if global_used >= GLOBAL_DAILY_LIMIT:
            print(
                f"Global daily limit reached: used={global_used}, "
                f"limit={GLOBAL_DAILY_LIMIT}, user_used={user_used}, "
                f"user_limit={USER_DAILY_LIMIT}",
                flush=True,
            )
            conn.rollback()
            return "global_limit", max(0, USER_DAILY_LIMIT - user_used)
        if minute_used >= GLOBAL_RPM_LIMIT:
            conn.rollback()
            return "rate_limit", max(0, USER_DAILY_LIMIT - user_used)
        if user_used >= USER_DAILY_LIMIT:
            if bonus_remaining > 0:
                conn.execute(
                    "UPDATE user_bonus SET bonus_remaining=bonus_remaining-1 WHERE user_id=? AND day=?",
                    (user_id, today),
                )
                conn.commit()
                return None, bonus_remaining - 1
            conn.rollback()
            return "user_limit", 0

        conn.execute("UPDATE user_daily SET used=used+1 WHERE user_id=? AND day=?", (user_id, today))
        conn.execute("UPDATE global_daily SET used=used+1 WHERE day=?", (today,))
        conn.execute("UPDATE global_minute SET used=used+1 WHERE minute=?", (minute,))
        conn.commit()
        return None, max(0, USER_DAILY_LIMIT - user_used - 1)


def add_user_bonus(user_id: str, amount: int = 5) -> None:
    today = day_key()
    with DB_LOCK, connect_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT OR IGNORE INTO user_bonus(user_id, day) VALUES (?, ?)", (user_id, today))
        conn.execute(
            "UPDATE user_bonus SET bonus_remaining=bonus_remaining+? WHERE user_id=? AND day=?",
            (amount, user_id, today),
        )
        conn.commit()


def get_user_bonus(user_id: str) -> int:
    today = day_key()
    with connect_db() as conn:
        row = conn.execute(
            "SELECT bonus_remaining FROM user_bonus WHERE user_id=? AND day=?", (user_id, today)
        ).fetchone()
        return row["bonus_remaining"] if row else 0


def error_payload(code: str, message: str, remaining: int = -1,
                  limit: int = USER_DAILY_LIMIT) -> dict:
    return {"error": code, "message": message, "remaining": remaining, "limit": limit}


def groq_error_details(exc: Exception) -> str:
    """Return a bounded diagnostic string for server logs only."""
    if isinstance(exc, HTTPError):
        try:
            body = exc.read(2000).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return f"HTTP {exc.code}: {body}"
    return repr(exc)


class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceAI/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.log_date_time_string()} {self.address_string()} {fmt % args}", flush=True)

    def send_audio(self, status: int, audio: bytes, content_type: str = "audio/mpeg") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(audio)

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_body(self, max_bytes: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("invalid_content_length")
        if length <= 0:
            raise ValueError("empty_body")
        if length > max_bytes:
            raise OverflowError("body_too_large")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("incomplete_body")
        return body

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/health":
            self.send_json(200, {"ok": True})
            return

        if parsed.path == "/admob/ssv":
            user_ids = query.get("user_id", [])
            if not user_ids:
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            install_id = user_ids[0].strip()
            if not INSTALL_ID_RE.fullmatch(install_id):
                self.send_json(400, {"error": "invalid_user"})
                return
            try:
                add_user_bonus(install_id, 5)
                print(f"AdMob SSV credited 5 questions to install ID {install_id}", flush=True)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
            except sqlite3.Error as exc:
                print(f"AdMob database error: {exc!r}", flush=True)
                self.send_json(500, {"error": "database_error"})
            return

        if parsed.path == "/quota":
            install_id = self.headers.get("X-Install-Id", "").strip()
            if not INSTALL_ID_RE.fullmatch(install_id):
                self.send_json(400, {"error": "invalid_user"})
                return
            try:
                self.send_json(200, {"bonus_remaining": get_user_bonus(install_id)})
            except sqlite3.Error as exc:
                print(f"Quota database error: {exc!r}", flush=True)
                self.send_json(500, {"error": "database_error"})
            return

        self.send_json(404, {"error": "not_found", "message": "요청한 주소를 찾을 수 없습니다."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/transcribe":
            self.handle_transcribe()
        elif path == "/ask":
            self.handle_ask()
        elif path == "/speak":
            self.handle_speak()
        else:
            self.send_json(404, {"error": "not_found", "message": "요청한 주소를 찾을 수 없습니다."})

    def handle_transcribe(self) -> None:
        if not GROQ_API_KEY:
            self.send_json(503, {"error": "server_error", "message": "서버에 Groq API 키가 설정되지 않았습니다."})
            return

        content_type = self.headers.get("Content-Type", "").lower()
        if "audio/wav" not in content_type and "application/octet-stream" not in content_type:
            self.send_json(415, {"error": "unsupported_audio", "message": "WAV 음성 파일을 보내 주세요."})
            return

        try:
            audio = self.read_body(MAX_AUDIO_BYTES)
        except OverflowError:
            self.send_json(413, {"error": "audio_too_large", "message": "음성 파일이 너무 큽니다."})
            return
        except ValueError:
            self.send_json(400, {"error": "invalid_audio", "message": "음성 파일이 비어 있거나 올바르지 않습니다."})
            return

        # Unity sends raw WAV bytes. Groq expects multipart/form-data, so wrap the
        # bytes in a multipart body here rather than expecting Unity to do it.
        boundary = "----VoiceAIFormBoundary7MA4YWxkTrZu0gW"
        multipart = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"{TRANSCRIPTION_MODEL}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="language"\r\n\r\n'
            "ko\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="response_format"\r\n\r\n'
            "json\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="voice.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + audio + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = Request(
            GROQ_TRANSCRIPTION_URL,
            data=multipart,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "VoiceAI-Groq-Server/1.1",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
            text = result.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self.send_json(502, {"error": "transcription_empty", "message": "음성을 인식하지 못했습니다."})
                return
            self.send_json(200, {"text": text.strip()})
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print(f"Groq transcription failed: {groq_error_details(exc)}", flush=True)
            self.send_json(502, {"error": "transcription_failed", "message": "음성 인식에 실패했습니다. 잠시 후 다시 시도해 주세요."})
        except Exception as exc:
            print(f"Unexpected transcription error: {exc!r}", flush=True)
            self.send_json(502, {"error": "transcription_failed", "message": "음성 인식에 실패했습니다. 잠시 후 다시 시도해 주세요."})


    def handle_speak(self) -> None:
        """Generate Korean MP3 speech with Cloudflare Workers AI MeloTTS."""
        if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
            self.send_json(503, {
                "error": "tts_not_configured",
                "message": "서버에 Cloudflare TTS 설정이 없습니다."
            })
            return

        try:
            raw = self.read_body(1024 * 1024)
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_json(400, {"error": "invalid_request", "message": "음성 요청 형식을 확인해 주세요."})
            return

        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            self.send_json(400, {"error": "invalid_text", "message": "읽을 문장을 입력해 주세요."})
            return

        text = text.strip()
        if len(text) > MAX_TTS_TEXT_CHARS:
            self.send_json(400, {
                "error": "text_too_long",
                "message": f"음성으로 변환할 문장은 {MAX_TTS_TEXT_CHARS}자 이하여야 합니다."
            })
            return

        payload = {"prompt": text, "lang": "ko"}
        req = Request(
            CLOUDFLARE_TTS_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "audio/mpeg, application/json",
                "User-Agent": "VoiceAI-Cloudflare-TTS/1.0",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=90) as response:
                audio_or_json = response.read(20 * 1024 * 1024)
                content_type = (response.headers.get("Content-Type") or "").lower()

            # Workers AI may return raw audio/mpeg or a JSON envelope with base64 audio.
            if content_type.startswith("audio/"):
                audio = audio_or_json
            else:
                result = json.loads(audio_or_json.decode("utf-8"))
                if result.get("success") is False:
                    raise ValueError(f"Cloudflare Workers AI reported errors: {result.get('errors', [])}")
                result_body = result.get("result", result)
                encoded = result_body.get("audio") if isinstance(result_body, dict) else None
                if not isinstance(encoded, str) or not encoded:
                    raise ValueError("Cloudflare response did not contain audio data")
                audio = base64.b64decode(encoded, validate=True)

            if not audio:
                raise ValueError("Cloudflare TTS returned empty audio")
            self.send_audio(200, audio, "audio/mpeg")
        except HTTPError as exc:
            details = groq_error_details(exc)
            print(f"Cloudflare MeloTTS failed: {details}", flush=True)
            self.send_json(502, {
                "error": "tts_provider_failed",
                "message": "음성 생성 서비스 요청에 실패했습니다. Render 로그를 확인해 주세요."
            })
        except (URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"Cloudflare MeloTTS failed: {exc!r}", flush=True)
            self.send_json(502, {
                "error": "tts_failed",
                "message": "음성을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
            })
        except Exception as exc:
            print(f"Unexpected Cloudflare MeloTTS error: {exc!r}", flush=True)
            self.send_json(502, {
                "error": "tts_failed",
                "message": "음성을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
            })

    def handle_ask(self) -> None:
        if not GROQ_API_KEY:
            self.send_json(503, error_payload("server_error", "서버에 Groq API 키가 설정되지 않았습니다."))
            return

        install_id = self.headers.get("X-Install-Id", "").strip()
        if not INSTALL_ID_RE.fullmatch(install_id):
            self.send_json(400, error_payload("invalid_user", "앱 식별 정보가 올바르지 않습니다."))
            return

        try:
            raw = self.read_body(1024 * 1024)
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_json(400, error_payload("invalid_request", "요청 형식을 확인해 주세요."))
            return

        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(question, str) or not question.strip():
            self.send_json(400, error_payload("invalid_request", "질문을 입력해 주세요."))
            return

        question = question.strip()
        if len(question) > MAX_QUESTION_CHARS:
            self.send_json(400, error_payload(
                "too_long", f"질문은 {MAX_QUESTION_CHARS}자까지만 보낼 수 있어요."
            ))
            return

        try:
            denial, remaining = reserve_request(install_id)
        except sqlite3.Error as exc:
            print(f"Usage database error: {exc!r}", flush=True)
            self.send_json(503, error_payload("server_error", "사용량 저장소에 문제가 생겼습니다."))
            return

        if denial == "global_limit":
            self.send_json(429, error_payload(
                "global_limit",
                "오늘 AI 전체 사용량이 모두 소진되었습니다. 내일 다시 이용해 주세요.",
                remaining,
                limit=GLOBAL_DAILY_LIMIT,
            ))
            return
        if denial == "rate_limit":
            self.send_json(429, error_payload(
                "rate_limit", "요청이 너무 많아요. 잠시 후 다시 시도해 주세요.", remaining
            ))
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
        req = Request(
            GROQ_CHAT_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "VoiceAI-Groq-Server/1.1",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))

            choices = result.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                raise ValueError("Groq response did not contain choices[0]")
            message = choices[0].get("message", {})
            completion = message.get("content", "")
            if not isinstance(completion, str) or not completion.strip():
                raise ValueError("Groq response content was empty")
            completion = completion.strip()

            if completion.startswith("```"):
                completion = re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", completion, flags=re.IGNORECASE
                ).strip()

            try:
                parsed = json.loads(completion)
            except json.JSONDecodeError:
                parsed = {"answer": completion}

            if not isinstance(parsed, dict):
                parsed = {"answer": completion}
            answer = parsed.get("answer", completion)
            tts_text = parsed.get("tts", answer)
            visual_steps = parsed.get("visuals", [])
            if not isinstance(answer, str):
                answer = str(answer)
            if not isinstance(tts_text, str):
                tts_text = answer
            if not isinstance(visual_steps, list):
                visual_steps = []

            visuals = [
                preprocess_tts_text(step)[:100]
                for step in visual_steps
                if isinstance(step, str)
            ]
            visuals = [item for item in visuals if item][:4]

            self.send_json(200, {
                "answer": answer.strip(),
                "tts": preprocess_tts_text(tts_text),
                "visuals": visuals,
                "remaining": remaining,
                "limit": USER_DAILY_LIMIT,
            })
        except HTTPError as exc:
            print(f"Groq chat request failed: {groq_error_details(exc)}", flush=True)
            self.send_json(502, error_payload(
                "server_error", "AI 제공자 요청에 실패했습니다. Render 로그를 확인해 주세요.", remaining
            ))
        except (URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError, KeyError) as exc:
            print(f"Groq chat request failed: {groq_error_details(exc)}", flush=True)
            self.send_json(502, error_payload(
                "server_error", "답변을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.", remaining
            ))
        except Exception as exc:
            print(f"Unexpected chat error: {exc!r}", flush=True)
            self.send_json(502, error_payload(
                "server_error", "답변을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.", remaining
            ))


if __name__ == "__main__":
    init_db()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(
        f"Voice AI server listening on {host}:{port}; model={MODEL}; "
        f"transcription_model={TRANSCRIPTION_MODEL}; "
        f"tts_model={CLOUDFLARE_TTS_MODEL}; tts_configured={bool(CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN)}",
        flush=True,
    )
    ThreadingHTTPServer((host, port), Handler).serve_forever()
