import json
import os
from typing import Any, Iterator

import requests

from bot.client.prompt import (
    CTX_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    REFINED_CTX_PROMPT_TEMPLATE,
    REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    TOOL_SYSTEM_TEMPLATE,
    generate_conversation_awareness_prompt,
    generate_ctx_prompt,
    generate_qa_prompt,
    generate_refined_ctx_prompt,
)
from bot.model.base_model import ModelSettings


class OllamaClient:
    """A lightweight Ollama HTTP client that mirrors the API surface used by the app.

    - Uses Ollama's local HTTP API (default: http://localhost:11434)
    - Implements streaming via the HTTP streaming endpoint
    - Yields chunks compatible with the OpenAI/Lama clients used elsewhere in the app
    """

    def __init__(self, model_name: str | None = None, model_settings: ModelSettings | None = None):
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", model_name or "llama3")

        # Minimal model_settings compatibility object used by the rest of the app
        class _MS:
            system_template = getattr(model_settings, "system_template", "You are a helpful assistant.")
            config = getattr(model_settings, "config", {})
            config_answer = getattr(model_settings, "config_answer", {"temperature": 0.0, "stop": []})
            reasoning = getattr(model_settings, "reasoning", False)
            reasoning_start_tag = getattr(model_settings, "reasoning_start_tag", None)
            reasoning_stop_tag = getattr(model_settings, "reasoning_stop_tag", None)

        self.model_settings = _MS()

    def _generate_payload(self, prompt: str, max_new_tokens: int = 512, stream: bool = False) -> dict:
        payload = {
            "model": self.model,
            "prompt": f"{self.model_settings.system_template}\n{prompt}",
            "max_tokens": max_new_tokens,
            "stream": stream,
        }

        # Merge any extra configs if present
        payload.update(self.model_settings.config_answer or {})
        return payload

    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        url = f"{self.host}/api/generate"
        payload = self._generate_payload(prompt, max_new_tokens, stream=False)

        try:
            resp = requests.post(url, json=payload, timeout=20)
            resp.raise_for_status()
        except Exception as exc:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.host}. Ensure Ollama is running and reachable. ({exc})"
            )

        try:
            data = resp.json()
            # Ollama non-stream response layout may include `text` or other top-level fields
            if isinstance(data, dict):
                text = data.get("text") or data.get("response") or ""
            else:
                text = ""
        except Exception:
            text = resp.text or ""

        return text

    async def async_generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        import asyncio

        return await asyncio.to_thread(self.generate_answer, prompt, max_new_tokens)

    def start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512) -> Iterator[dict]:
        url = f"{self.host}/api/generate"
        payload = self._generate_payload(prompt, max_new_tokens, stream=True)

        try:
            resp = requests.post(url, json=payload, stream=True, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.host}. Ensure Ollama is running and reachable. ({exc})"
            )

        # Ollama streams chunks as newline-delimited data frames. Each non-empty line may be:
        # - a raw text chunk
        # - or an SSE-like `data: {json}` line
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if line == "":
                continue

            # Support SSE-style `data: ` prefix
            if line.startswith("data:"):
                line = line[len("data:"):].strip()

            # Some implementations return a sentinel like [DONE]
            if line in ("[DONE]", "done"):
                break

            chunk_text = ""
            try:
                obj = json.loads(line)
                # Typical Ollama streaming JSON contains a `text` field or nested `output`/`choices`
                if isinstance(obj, dict):
                    chunk_text = obj.get("text") or obj.get("output") or obj.get("token") or ""
                    # Sometimes response contains a list under `choices` with partial `text` entries
                    if not chunk_text and obj.get("choices"):
                        try:
                            chunk_text = obj["choices"][0].get("text", "")
                        except Exception:
                            pass
            except Exception:
                # Not JSON; treat line as raw text content
                chunk_text = line

            if not chunk_text:
                continue

            yield {"choices": [{"delta": {"content": chunk_text}}]}

    async def async_start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512):
        import asyncio

        loop = asyncio.get_running_loop()

        def _sync_stream():
            return [c for c in self.start_answer_iterator_streamer(prompt, max_new_tokens)]

        chunks = await loop.run_in_executor(None, _sync_stream)
        for c in chunks:
            yield c

    @staticmethod
    def parse_token(token: dict) -> str:
        return token["choices"][0]["delta"].get("content", "")

    @staticmethod
    def generate_qa_prompt(question: str) -> str:
        return generate_qa_prompt(template=QA_PROMPT_TEMPLATE, question=question)

    @staticmethod
    def generate_ctx_prompt(question: str, context: str) -> str:
        return generate_ctx_prompt(template=CTX_PROMPT_TEMPLATE, question=question, context=context)

    @staticmethod
    def generate_refined_ctx_prompt(question: str, context: str, existing_answer: str) -> str:
        return generate_refined_ctx_prompt(
            template=REFINED_CTX_PROMPT_TEMPLATE, question=question, context=context, existing_answer=existing_answer
        )

    @staticmethod
    def generate_refined_question_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(
            template=REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
            question=question,
            chat_history=chat_history,
        )

    @staticmethod
    def generate_refined_answer_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(
            template=REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
            question=question,
            chat_history=chat_history,
        )

    def retrieve_tools(self, prompt: str, max_new_tokens: int = 512, tools: list[dict] = None, tool_choice: str = None):
        raise NotImplementedError("Tool calls are not implemented for OllamaClient")
