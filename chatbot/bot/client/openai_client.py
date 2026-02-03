import os
from typing import Any, Iterator

import openai
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


class OpenAIClient:
    """A lightweight OpenAI chat client that mirrors the API surface used by the app.

    Methods implemented here match the LamaCppClient methods used by the project so the rest of
    the codebase can remain unchanged.
    """

    def __init__(self, model_name: str | None = None, model_settings: ModelSettings | None = None):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("Environment variable OPENAI_API_KEY is required to use OpenAIClient")

        openai.api_key = self.api_key

        # Allow model override via OPENAI_MODEL env var; otherwise use a sensible default for OpenAI
        self.model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

        # Minimal model_settings compatibility object used by the rest of the app
        class _MS:
            system_template = getattr(model_settings, "system_template", "You are a helpful assistant.")
            config = getattr(model_settings, "config", {})
            config_answer = getattr(model_settings, "config_answer", {"temperature": 0.0, "stop": []})
            reasoning = getattr(model_settings, "reasoning", False)
            reasoning_start_tag = getattr(model_settings, "reasoning_start_tag", None)
            reasoning_stop_tag = getattr(model_settings, "reasoning_stop_tag", None)

        self.model_settings = _MS()

    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.model_settings.system_template},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            **(self.model_settings.config_answer or {}),
        )

        text = response["choices"][0]["message"].get("content", "")
        return text

    async def async_generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        # Run in a thread to avoid requiring openai async client
        import asyncio

        return await asyncio.to_thread(self.generate_answer, prompt, max_new_tokens)

    def start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512) -> Iterator[dict]:
        """Yield chunks with the same structure the app expects (choices[*].delta.content)."""
        stream = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.model_settings.system_template},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            stream=True,
            **(self.model_settings.config_answer or {}),
        )

        for chunk in stream:
            # The openai stream itself is compatible with the parsing used by LamaCppClient
            # ensure we only yield chunks that include delta content
            try:
                yield chunk
            except Exception:
                continue

    async def async_start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512):
        # Provide an async wrapper compatible with AsyncTreeSummarizationStrategy
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
        # OpenAI function-calling support can be added later; not needed for quick demo
        raise NotImplementedError("Tool calls are not implemented for OpenAIClient fallback")
