import argparse
import sys
import time
from pathlib import Path

import os
import streamlit as st
from typing import Any
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import answer_with_context, extract_content_after_reasoning, refine_question
from bot.conversation.ctx_strategy import (
    BaseSynthesisStrategy,
    get_ctx_synthesis_strategies,
    get_ctx_synthesis_strategy,
)
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.model.model_registry import get_model_settings, get_models
from helpers.log import get_logger
from helpers.prettier import prettify_source

logger = get_logger(__name__)

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", initial_sidebar_state="collapsed")


@st.cache_resource()
def load_llm_client(model_folder: Path, model_name: str):
    """Load the LLM client.

    Backend selection logic (lazy):
    - If `OLLAMA_HOST` is reachable -> use `OllamaClient` (local inference)
    - Else if `OPENAI_API_KEY` exists -> use `OpenAIClient` (remote OpenAI)
    - Else -> show a clear Streamlit error
    """
    import requests

    # Prefer Ollama local backend if reachable
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if ollama_host.endswith("/api"):
        probe_url = f"{ollama_host}/version"
    else:
        probe_url = f"{ollama_host}/api/version"
    try:
        probe = requests.get(probe_url, timeout=1)
        if 200 <= probe.status_code < 400:
            from bot.client.ollama_client import OllamaClient

            model_settings = get_model_settings(model_name)
            llm = OllamaClient(model_name=model_name, model_settings=model_settings)
            logger.info("Using OllamaClient (model=%s)", llm.model)
            return llm
        logger.warning("Ollama probe at %s returned status %s; falling back if possible.", probe_url, probe.status_code)
    except Exception as exc:
        # Ollama not reachable; fall through to other backends
        logger.warning("Ollama probe failed at %s; falling back if possible. Error: %s", probe_url, exc)

    # Prefer remote OpenAI backend when API key is provided
    if os.environ.get("OPENAI_API_KEY"):
        from bot.client.openai_client import OpenAIClient

        llm = OpenAIClient(model_name=model_name)
        logger.info("LLM backend selected: OpenAI (Ollama unreachable)")
        return llm

    # If we reach here, no backend is available
    st.error(
        "No LLM backend available. Set `OLLAMA_HOST` (and run Ollama + pull a model) or set `OPENAI_API_KEY` environment variable."
    )
    raise EnvironmentError("No LLM backend available. See Streamlit UI for details.")


@st.cache_resource()
def init_chat_history(total_length: int = 2) -> ChatHistory:
    chat_history = ChatHistory(total_length=total_length)
    return chat_history


@st.cache_resource()
def load_ctx_synthesis_strategy(ctx_synthesis_strategy_name: str, _llm: Any) -> BaseSynthesisStrategy:
    ctx_synthesis_strategy = get_ctx_synthesis_strategy(ctx_synthesis_strategy_name, llm=_llm)
    return ctx_synthesis_strategy


@st.cache_resource()
def load_index(vector_store_path: Path) -> Chroma:
    """
    Loads a Vector Database index based on the specified vector store path.

    Args:
        vector_store_path (Path): The path to the vector store.

    Returns:
        Chroma: An instance of the Vector Database.
    """
    embedding = Embedder()
    index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)

    return index


def init_page(root_folder: Path) -> None:
    """
    Initializes the page configuration for the application.
    """
    left_column, central_column, right_column = st.columns([2, 1, 2])

    with left_column:
        st.write(" ")

    with central_column:
        st.image(str(root_folder / "images/bot.png"), use_column_width="always")
        st.markdown("""<h4 style='text-align: center; color: grey;'></h4>""", unsafe_allow_html=True)

    with right_column:
        st.write(" ")

    st.sidebar.title("Options")


@st.cache_resource
def init_welcome_message() -> None:
    """
    Initializes a welcome message for the chat interface.
    """
    with st.chat_message("assistant"):
        st.write("How can I help you today?")


def reset_chat_history(chat_history: ChatHistory) -> None:
    """
    Initializes the chat history, allowing users to clear the conversation.
    """
    clear_button = st.sidebar.button("🗑️ Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        chat_history.clear()


def display_messages_from_history():
    """
    Displays chat messages from the history on app rerun.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main(parameters) -> None:
    """
    Main function to run the RAG Chatbot application.

    Args:
        parameters: Parameters for the application.
    """
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_name = parameters.model
    synthesis_strategy_name = parameters.synthesis_strategy
    max_new_tokens = parameters.max_new_tokens

    init_page(root_folder)
    llm = load_llm_client(model_folder, model_name)
    chat_history = init_chat_history(2)
    ctx_synthesis_strategy = load_ctx_synthesis_strategy(synthesis_strategy_name, _llm=llm)
    index = load_index(vector_store_path)
    reset_chat_history(chat_history)
    init_welcome_message()
    display_messages_from_history()

    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display retrieved documents with content previews, and updates the chat interface with the assistant's
        # responses.
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(
                text="Refining the question and Retrieving the docs – hang tight! This should take seconds."
            ):
                refined_user_input = refine_question(
                    llm, user_input, chat_history=chat_history, max_new_tokens=max_new_tokens
                )
                retrieved_contents, sources = index.similarity_search_with_threshold(
                    query=refined_user_input, k=parameters.k, threshold=parameters.threshold
                )
                if retrieved_contents:
                    full_response += "Here are the retrieved text chunks with a content preview: \n\n"
                    message_placeholder.markdown(full_response)

                    for source in sources:
                        full_response += prettify_source(source)
                        full_response += "\n\n"
                        message_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    full_response += "I did not detect any pertinent chunk of text from the documents. \n\n"
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Display assistant response in chat message container
        start_time = time.time()
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(text="Refining the context and Generating the answer for each text chunk – hang tight! "):
                streamer, _ = answer_with_context(
                    llm, ctx_synthesis_strategy, user_input, chat_history, retrieved_contents, max_new_tokens
                )
                for token in streamer:
                    full_response += llm.parse_token(token)
                    message_placeholder.markdown(full_response + "▌")

                if llm.model_settings.reasoning:
                    answer = extract_content_after_reasoning(full_response, llm.model_settings.reasoning_stop_tag)
                    if answer == "":
                        answer = "I wasn't able to provide the answer; Do you want me to try again?"
                else:
                    answer = full_response

                chat_history.append(f"question: {user_input}, answer: {answer}")

                message_placeholder.markdown(answer)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Chatbot")

    model_list = get_models()
    default_model = model_list[0]

    synthesis_strategy_list = get_ctx_synthesis_strategies()
    default_synthesis_strategy = synthesis_strategy_list[0]

    parser.add_argument(
        "--model",
        type=str,
        choices=model_list,
        help=f"Model to be used. Defaults to {default_model}.",
        required=False,
        const=default_model,
        nargs="?",
        default=default_model,
    )

    parser.add_argument(
        "--synthesis-strategy",
        type=str,
        choices=synthesis_strategy_list,
        help=f"Model to be used. Defaults to {default_synthesis_strategy}.",
        required=False,
        const=default_synthesis_strategy,
        nargs="?",
        default=default_synthesis_strategy,
    )

    parser.add_argument(
        "--k",
        type=int,
        help="Number of chunks to return from the similarity search. Defaults to 2.",
        required=False,
        default=2,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help="Relevance score threshold for retrieval (0-1). Defaults to 0.2.",
        required=False,
        default=0.2,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="The maximum number of tokens to generate in the answer. Defaults to 512.",
        required=False,
        default=512,
    )

    return parser.parse_args()


# streamlit run rag_chatbot_app.py
if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
