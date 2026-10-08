# Unity 앱용 Groq 서버

Unity `AIClient`가 사용하는 `POST /ask` 형식과 호환되는 작은 Python 서버입니다. Groq API 키는 환경변수 `GROQ_API_KEY`에서만 읽습니다.

## 실행

Python 3.10 이상이 필요합니다. 이 서버는 Python 표준 라이브러리만 사용합니다.

1. 호스팅 서비스의 비밀 환경변수에 `.env.example`의 값을 등록합니다. `GROQ_API_KEY`는 실제 키로 설정합니다.
2. `python server.py`로 실행합니다.
3. Unity의 `AIClient.serverUrl`에 배포된 HTTPS 서버 주소를 넣고 `askPath`는 `/ask`로 둡니다.

개발 PC에서만 확인할 때는 `HOST=127.0.0.1`로 실행하고 `/health`를 확인할 수 있습니다. Android 기기의 `localhost`는 개발 PC가 아니라 Android 기기 자신을 가리킵니다.

## 제한과 저장

- 사용자별 일일 질문 10회, 질문 30자, 전체 분당 요청 수, 전체 일일 요청 수를 서버에서 검사합니다.
- 일일 날짜 경계는 `APP_TIMEZONE` 기준 자정이며 기본값은 한국 시간입니다.
- 요청을 Groq로 보내기 전에 횟수를 예약합니다. Groq 호출 실패도 서버의 예약량에 포함되어 반복 재시도로 한도를 우회하지 못합니다.
- 사용량은 SQLite에 저장됩니다. 호스팅 환경에서 재시작 후에도 유지되도록 영구 디스크를 연결하고 `AI_DB_PATH`를 그 경로로 지정해야 합니다.
- `X-Install-Id`는 현재 Unity 클라이언트가 보내는 임의 설치 ID입니다. 앱 재설치·기기 변경·ID 위조로 사용자 제한을 우회할 수 있으므로 실제 출시에서 강한 사용자별 제한을 원하면 Firebase Authentication 등 검증된 로그인 토큰을 서버에서 검증하도록 추가해야 합니다.
- 이 단일 프로세스 서버는 한 인스턴스의 SQLite 저장소를 기준으로 전체 한도를 계산합니다. 여러 서버 인스턴스를 운영하면 Redis나 관리형 데이터베이스로 한도 카운터를 공유해야 합니다.

Groq의 모델별 RPM/RPD/TPM 한도는 계정·모델·요금제에 따라 달라질 수 있으므로 `GLOBAL_DAILY_LIMIT`, `GLOBAL_RPM_LIMIT`, `GROQ_MODEL`을 실제 Groq 콘솔의 값에 맞춰 설정하세요. 앱 내부 카운터는 Groq 계정의 공식 제한을 대신하지 않습니다.

## API

요청:

```http
POST /ask
Content-Type: application/json
X-Install-Id: Unity가 생성한 설치 ID

{"question":"안녕하세요"}
```

성공 응답은 `{"answer":"...","remaining":9,"limit":10}`입니다. 오류의 `error` 코드는 Unity `AIClient`가 해석하는 `user_limit`, `global_limit`, `rate_limit`, `too_long`과 호환됩니다. 상태 확인은 `GET /health`를 사용합니다.
