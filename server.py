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
        f"transcription_model={TRANSCRIPTION_MODEL}",
        flush=True,
    )
    ThreadingHTTPServer((host, port), Handler).serve_forever()
