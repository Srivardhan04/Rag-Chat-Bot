import os
from typing import Any, Iterator

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, ChatGeneration

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


class LangChainOpenAIClient:
    """Simple LangChain-backed OpenAI client to match the project's LLM interface.

    This client implements the subset of methods used by the app (sync/async generate,
    streaming iterator) and token parsing that mimic the shape returned by llama-cpp
    streaming, so the rest of the codebase can remain unchanged.
    """

    def __init__(self, model_name: str | None = None):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key is None:
            raise RuntimeError("OPENAI_API_KEY not found in environment")

        # allow override via OPENAI_MODEL; fall back to a sensible default
        model_name = os.environ.get("OPENAI_MODEL", model_name or "gpt-3.5-turbo")
        # reasonable defaults, can be overridden by env vars in production
        self.llm = ChatOpenAI(model_name=model_name, temperature=float(os.environ.get("OPENAI_TEMPERATURE", 0.0)))
        self.model = model_name

        # lightweight default settings, no heavy local downloads
        self.model_settings = ModelSettings()

    # Public API that mirrors the LamaCppClient interface used by the project
    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [SystemMessage(content=""), HumanMessage(content=prompt)]
        resp = self.llm(messages)
        # concatenate message content
        return resp.content if isinstance(resp, ChatGeneration) else getattr(resp, "content", "")

    def async_generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        # langchain ChatOpenAI doesn't provide an async interface here; use the sync one
        return self.generate_answer(prompt, max_new_tokens=max_new_tokens)

    def start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512) -> Iterator[dict]:
        """Yield incremental token events in a shape compatible with LamaCppClient.parse_token.

        Implementation detail: Use OpenAI streaming at HTTP level via ChatOpenAI's streaming support.
        To keep this simple and robust, we call the model in streaming mode and yield small dicts
        with the same structure expected by parse_token in the codebase: {"choices": [{"delta": {"content": ..}}]}
        """
        # ChatOpenAI provides streaming when ``streaming=True`` and returns token events via callbacks.
        # Simpler approach: call the underlying model with streaming enabled using its `.generate()` API
        # and yield tokens as they come.
        for chunk in self.llm.generate([HumanMessage(content=prompt)], streaming=True):
            for gen in chunk.generations:
                # yield chunk as a dict compatible with the LamaCppClient.parse_token expectation
                yield {"choices": [{"delta": {"content": gen.text}}]}

    @staticmethod
    def parse_token(token: dict) -> str:
        return token["choices"][0]["delta"].get("content", "")

    # Prompt helpers mirrored from LamaCppClient
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
            template=REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE, question=question, chat_history=chat_history
        )

    @staticmethod
    def generate_refined_answer_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(
            template=REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE, question=question, chat_history=chat_history
        )

    def retrieve_tools(self, *args, **kwargs):
        # Not implemented for remote OpenAI via LangChain; keep signature for compatibility
        return None
