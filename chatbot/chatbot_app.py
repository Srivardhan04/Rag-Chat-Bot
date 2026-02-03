import argparse
import sys
import time
from pathlib import Path

import streamlit as st
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import answer, extract_content_after_reasoning
from bot.model.model_registry import get_model_settings, get_models
from helpers.log import get_logger
import os

logger = get_logger(__name__)

# Set page config at the very beginning
st.set_page_config(page_title="Chatbot", page_icon="💬", initial_sidebar_state="collapsed")


@st.cache_resource()
def load_llm(model_name: str, model_folder: Path):
    """
    Create a LLM session object that points to the model.

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
            # Local import to avoid importing the Ollama client when not needed
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
        # Local import to avoid importing the OpenAI client when not needed
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


def init_page(root_folder: Path) -> None:
    left_column, central_column, right_column = st.columns([2, 1, 2])

    with left_column:
        st.write(" ")

    with central_column:
        st.image(str(root_folder / "images/bot-small.png"), use_column_width="auto")
        st.markdown("""<h4 style='text-align: center; color: grey;'></h4>""", unsafe_allow_html=True)

    with right_column:
        st.write(" ")

    st.sidebar.title("Options")


@st.cache_resource
def init_welcome_message() -> None:
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
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main(parameters) -> None:
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model = parameters.model
    max_new_tokens = parameters.max_new_tokens

    init_page(root_folder)
    llm = load_llm(model, model_folder)
    chat_history = init_chat_history(2)
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

        # Display assistant response in chat message container
        start_time = time.time()
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for token in answer(llm=llm, question=user_input, chat_history=chat_history, max_new_tokens=max_new_tokens):
                full_response += llm.parse_token(token)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        if llm.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(full_response, llm.model_settings.reasoning_stop_tag)
            if final_answer == "":
                final_answer = "I didn't provide the answer; perhaps I can try again."
        else:
            final_answer = full_response

        message_placeholder.markdown(final_answer)
        # Add assistant response to chat history
        chat_history.append(f"question: {user_input}, answer: {final_answer}")
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot")

    model_list = get_models()
    default_model = model_list[0]

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
        "--max-new-tokens",
        type=int,
        help="The maximum number of tokens to generate in the answer. Defaults to 512.",
        required=False,
        default=512,
    )

    return parser.parse_args()


# streamlit run chatbot_app.py
if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
