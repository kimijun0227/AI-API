# Unity 보이스 AI 서버

Unity 앱에서 음성 인식, AI 응답, 음성 합성을 호출하는 Python 서버입니다. AI 응답은 Groq, Cerebras, Gemini, Cloudflare Workers AI 순으로 대체할 수 있으며, 음성 합성은 Cloudflare Workers AI의 MeloTTS를 사용합니다.

## Render 설정

Render 서비스의 Environment에 다음 환경변수를 설정하세요. 키와 계정 ID는 저장소나 Unity 앱에 넣지 말고 Render의 비밀 환경변수로만 관리합니다.

필수 설정:

- `GROQ_API_KEY`: AI 응답과 음성 인식에 사용합니다.
- `CLOUDFLARE_API_TOKEN`: Workers AI 호출 권한이 있는 Cloudflare API 토큰입니다. Workers AI Read/Edit 권한을 부여하세요.
- `CLOUDFLARE_ACCOUNT_ID`: Cloudflare 대시보드의 계정 ID입니다.

Cloudflare 변수 이름이 이미 Render에 있다면 값은 바꾸지 않아도 됩니다. TTS 기본 모델과 한국어 설정은 각각 `CLOUDFLARE_TTS_MODEL=@cf/myshell-ai/melotts`, `CLOUDFLARE_TTS_LANG=KR`입니다. 모델을 바꿀 때만 별도로 설정하세요.

변수를 저장한 뒤 Render에서 재배포하고, `GET /health` 응답의 `tts_configured`가 `true`인지 확인하세요. `false`이면 토큰 또는 계정 ID가 비어 있거나 변수 이름이 다릅니다. 실제 음성 요청은 `POST /speak`로 처리하며 MP3를 반환합니다.

## Unity 연결

Unity `AIClient.serverUrl`에 Render의 HTTPS 주소를 입력합니다. 기존 `/ask`, `/transcribe`, `/speak` 경로와 MP3 응답 형식을 유지하므로 TTS 공급자 변경 때문에 Unity 스크립트는 수정하지 않아도 됩니다.

## API

- `POST /ask`: JSON AI 응답 및 시각 설명 데이터를 반환합니다.
- `POST /transcribe`: WAV 녹음을 한국어 텍스트로 변환합니다.
- `POST /speak`: 텍스트를 한국어 MP3 음성으로 합성합니다.
- `GET /health`: 서버와 공급자 설정 여부를 반환합니다.

요청·응답 세부 형식은 [API_SPEC.md](API_SPEC.md)를 참고하세요.

## 제한과 저장

사용자별·전체 일일 호출 제한과 분당 호출 제한은 환경변수로 설정합니다. 사용량은 SQLite에 저장되므로 Render에서 재시작 후에도 보존하려면 영구 디스크를 연결하고 `AI_DB_PATH`를 그 경로로 설정해야 합니다. 현재 제한은 서버의 자체 카운터이며 각 공급자의 계정 한도를 대신하지 않습니다.
