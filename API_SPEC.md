# API specification for Unity AIClient

## `POST /ask`

Headers: `Content-Type: application/json`, `X-Install-Id: <opaque install identifier>`

Body: `{"question":"..."}`

Success (`200`): `{"answer":"...","remaining":9,"limit":10}`

Errors return `{"error":"<code>","message":"...","remaining":0,"limit":10}`. Codes: `user_limit`, `global_limit`, `rate_limit`, `too_long`, `invalid_user`, `invalid_request`, `server_error`.

## `GET /health`

Returns `{"ok":true}` when the process is responding.
