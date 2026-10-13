# Unity AI API specification

All endpoints are served over HTTPS in production. JSON endpoints use UTF-8. `X-Install-Id` is used for server-side usage limits.

## `POST /ask`

Headers: `Content-Type: application/json`, `X-Install-Id: <install identifier>`

Body: `{"question":"...","history":[]}`

Success (`200`) includes `answer`, `tts`, `visuals`, `remaining`, and `limit`. `answer` is the on-screen response, `tts` is speech-ready text, and `visuals` is an array of short visual explanation steps.

## `POST /transcribe`

Send a WAV recording as the raw request body with `Content-Type: audio/wav` and `X-Install-Id`.

Success (`200`): `{"text":"recognized Korean speech"}`.

## `POST /speak`

Headers: `Content-Type: application/json`, `X-Install-Id: <install identifier>`

Body: `{"text":"text to speak"}`

Success (`200`): binary MPEG audio (`audio/mpeg`). The server synthesizes speech with Cloudflare Workers AI MeloTTS. Unity should continue to play this response as MP3; no Unity-side provider change is required.

Errors return JSON with `error`, `message`, `remaining`, and `limit`. TTS-specific errors include `tts_user_limit`, `tts_global_limit`, `rate_limit`, `too_long`, and `tts_provider_error`.

## `GET /health`

Returns process health and provider configuration flags, including `tts_configured`. This only confirms that Cloudflare token and account ID are present; it does not verify that the token works or that the model request succeeds.
