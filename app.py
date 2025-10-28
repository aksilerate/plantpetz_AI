# app.py
# ---------------------------------------------------
# A modern Streamlit UI for plant image Q&A with OpenAI Vision
# ---------------------------------------------------
import os
import io
import base64
from typing import List, Dict, Optional

import streamlit as st
import hashlib
from PIL import Image

# Optional: Use the official OpenAI client if available
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


# ---------------------------
# Page Config + Small Theming
# ---------------------------
st.set_page_config(
    page_title="PlantPetz AI",
    page_icon="🌿",
    layout="centered",
    menu_items={
        "About": "PlantPetz AI • Streamlit + OpenAI Vision"
    },
)

# Subtle CSS polish for a friendlier, modern look
st.markdown(
    """
    <style>
    /* Tighter top padding */
      .block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
    /* Make the 'Current Image' fit column width but never exceed a safe height */
      .sticky-card [data-testid="stImage"] img {
        width: 100% !important;
        height: auto !important;
        max-height: min(72vh, 650px);
        object-fit: contain;
        display: block;
      }

      /* Optional: slightly tighter cap on small screens */
      @media (max-width: 600px) {
        .sticky-card [data-testid="stImage"] img {
          max-height: 48vh;
        }
      }

      /* Chat bubbles */
      .user-bubble, .assistant-bubble, .system-bubble {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        line-height: 1.55;
        border: 1px solid rgba(0,0,0,0.06);
      }
      .user-bubble {
        background: #eef6ff;
      }
      .assistant-bubble {
        background: #f8f9fb;
      }
      .system-bubble {
        background: #fff6e5;
      }

      /* Fix list formatting inside chat bubbles */
      .user-bubble ul, .assistant-bubble ul, .system-bubble ul,
      .user-bubble ol, .assistant-bubble ol, .system-bubble ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
      }

      .user-bubble li, .assistant-bubble li, .system-bubble li {
        margin: 0.25rem 0;
        line-height: 1.55;
      }

      /* Ensure paragraphs in lists have proper spacing */
      .user-bubble p, .assistant-bubble p, .system-bubble p {
        margin: 0.5rem 0;
      }

      .user-bubble p:first-child, .assistant-bubble p:first-child, .system-bubble p:first-child {
        margin-top: 0;
      }

      .user-bubble p:last-child, .assistant-bubble p:last-child, .system-bubble p:last-child {
        margin-bottom: 0;
      }

      /* Fix code blocks in bubbles */
      .user-bubble code, .assistant-bubble code, .system-bubble code {
        background: rgba(0, 0, 0, 0.05);
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
        font-size: 0.9em;
      }

      .user-bubble pre, .assistant-bubble pre, .system-bubble pre {
        background: rgba(0, 0, 0, 0.05);
        padding: 0.75rem;
        border-radius: 6px;
        overflow-x: auto;
        margin: 0.5rem 0;
      }

      .user-bubble pre code, .assistant-bubble pre code, .system-bubble pre code {
        background: none;
        padding: 0;
      }

      /* Constrain Streamlit chat blocks to their parent container (the chat column) */
        [data-testid="stChatMessage"] {
          max-width: 100% !important;
          width: 100% !important;
          margin-left: 0 !important;
          margin-right: 0 !important;
        }

      /* Ensure any long text/URLs wrap inside bubbles */
        .user-bubble, .assistant-bubble, .system-bubble {
          word-break: break-word;
          overflow-wrap: anywhere;
          white-space: normal;  /* Changed from pre-wrap to normal for better markdown rendering */
          max-width: 100%;
          box-sizing: border-box;
        }

      /* Subtle dividers */
      hr { margin: 1.2rem 0; border: none; border-top: 1px solid rgba(0,0,0,0.06); }

      /* Sticky preview card (on wide screens) */
      @media (min-width: 1000px) {
        .sticky-card {
          position: sticky;
          top: 1rem;
        }
      }

      /* Smaller file uploader label */
      .uploadedFile { font-size: 0.9rem; }

      /* Fix Chat container's min/max height and make it scroll */
        #chat-anchor + div[data-testid="stVerticalBlock"]{
          height: clamp(460px, 72vh, 84vh);
          min-height: 460px;
          max-height: 84vh;
          overflow-y: auto;
          position: relative;
        }

    /* Minimal circular loader used inside assistant bubbles */
        .loader {
          display: inline-block;
          width: 16px;
          height: 16px;
          border: 2px solid rgba(0,0,0,0.2);
          border-top-color: rgba(0,0,0,0.6);
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
          vertical-align: -2px;
          margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Helpers
# ---------------------------
def b64_from_image_bytes(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def ensure_client(api_key: Optional[str]) -> Optional["OpenAI"]:
    if not _HAS_OPENAI:
        st.error("The `openai` package is not available. Install with `pip install openai`.")
        return None
    if not api_key:
        st.warning("OpenAI API key missing. Set OPENAI_API_KEY in Streamlit Secrets.")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None


def build_messages(
    system_prompt: str,
    chat_history: List[Dict[str, str]],
    user_text: str,
    image_b64: Optional[str],
    image_mime: str = "image/jpeg",
) -> List[Dict]:
    """
    Construct messages for the Chat Completions API.
    We keep history as text-only; attach the latest image (if any) to the newest user turn.
    """
    messages: List[Dict] = []

    # System prompt
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    # Prior turns (as plain text)
    for turn in chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        messages.append({"role": role, "content": content})

    # Latest user message (text + optional image)
    if image_b64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            ],
        })
    else:
        messages.append({"role": "user", "content": user_text})

    return messages


def stream_chat_completion(
    client: "OpenAI",
    model: str,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 1000,
):
    """
    Streams tokens from the Chat Completions API. Yields text chunks.
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                yield delta
    except Exception as e:
        yield f"\n[Error] {e}"


def image_from_uploader(uploader_file) -> Optional[bytes]:
    if uploader_file is None:
        return None
    try:
        # Normalize the image (helpful for large PNGs, EXIF, etc.)
        image = Image.open(uploader_file).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception:
        # If we can’t process as PIL, fallback to raw bytes
        try:
            return uploader_file.getvalue()
        except Exception:
            return None


# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # List[{"role": "user"/"assistant", "content": str}]

if "image_b64" not in st.session_state:
    st.session_state.image_b64 = None

if "image_name" not in st.session_state:
    st.session_state.image_name = None

if "image_mime" not in st.session_state:
    st.session_state.image_mime = "image/jpeg"

if "auto_kickoff_image_id" not in st.session_state:
    st.session_state.auto_kickoff_image_id = None

# ---------------------------
# Defaults
# ---------------------------
# API key: Streamlit Secrets first (Cloud), then env (local)
api_key = (
    st.secrets["OPENAI_API_KEY"]
    if "OPENAI_API_KEY" in st.secrets
    else os.getenv("OPENAI_API_KEY", "")
)

system_prompt = (
    "You are a expert in botany and plant care. "
    "Be concise, friendly and casual in your responses. "
    "You can only answer questions related to botany and plant care. " 
)

# ---------------------------
# Sidebar (Controls)
# ---------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model = st.selectbox(
        "Model",
        options=["gpt-4o"],
        index=0,
        help="‘gpt-4o’ is powerful for multimodal (vision) tasks"
    )

    temperature = st.slider("Model Creativity", 0.0, 1.0, 0.2, 0.05)

    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.session_state.image_b64 = None
        st.session_state.image_name = None

# ---------------------------
# Header
# ---------------------------
st.markdown("# 🌿 PlantPetz AI")

# Sticky preview of the current image (if any)
with st.container():
    c1, c2 = st.columns([1, 2], vertical_alignment="top")
    with c1:
        if not st.session_state.image_b64:
            st.markdown("#### Add an Image to Begin")

            side_upload = st.file_uploader(
                "Upload plant image (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                key="sidebar_uploader",
            )

            # Nothing chosen yet: stop here and wait for user action
            if side_upload is None:
                st.stop()

            # If a file was chosen, process and store it, then rerun
            img_bytes = image_from_uploader(side_upload)
            if img_bytes:
                st.session_state.image_b64 = b64_from_image_bytes(img_bytes)
                st.session_state.image_name = side_upload.name or "uploaded.jpg"
                # Use the real mime type if available; default to jpeg
                st.session_state.image_mime = getattr(side_upload, "type", None) or "image/jpeg"
                st.rerun()
            else:
                st.warning("Could not read image. Please try another file.")
                st.stop()

        # --- If we’re here, the image exists; show it ---
        st.markdown("#### Current Image")
        st.markdown('<div class="sticky-card">', unsafe_allow_html=True)
        st.image(
            f"data:{st.session_state.image_mime};base64,{st.session_state.image_b64}",
            caption=st.session_state.image_name or "Image",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("#### Chat")
        st.markdown('<div id="chat-anchor"></div>', unsafe_allow_html=True)
        chat_container = st.container()


# ---------------------------
# Render history
# ---------------------------
with chat_container:
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        css_class = "assistant-bubble" if role == "assistant" else ("system-bubble" if role == "system" else "user-bubble")
        with st.chat_message(role):
            st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)

# ---------------------------
# Auto analysis on new image upload
# ---------------------------
if st.session_state.image_b64:
    # Stable id for the current image to avoid re-triggering on reruns
    image_id = hashlib.md5(st.session_state.image_b64.encode("utf-8")).hexdigest()

    # Only fire once per unique image
    if st.session_state.auto_kickoff_image_id != image_id:
        # Prepare an automatic "user" question that asks for species + health
        auto_prompt = (
            "Identify the likely species/common name (with confidence).\n"
            "Comment on the plant's health. Keep your response concise & casual.\n"
        )

        # Render inside chat area and stream the assistant reply
        with chat_container:

            client = ensure_client(api_key)
            if client:
                # Build messages using the existing helper (attach image)
                messages = build_messages(
                    system_prompt=system_prompt,
                    chat_history=st.session_state.messages[:-1],  # exclude the just-added auto prompt
                    user_text=auto_prompt,
                    image_b64=st.session_state.image_b64,
                    image_mime=st.session_state.image_mime,
                )

                with st.chat_message("assistant"):
                    # 1) Show a loader bubble immediately
                    response_placeholder = st.empty()
                    full_text_chunks: List[str] = []
                    response_placeholder.markdown(
                        '<div class="assistant-bubble"><span class="loader"></span>Thinking…</div>',
                        unsafe_allow_html=True,
                    )

                    # 2) Stream tokens into a separate placeholder                  
                    for chunk in stream_chat_completion(
                        client=client,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=1200,
                    ):
                        full_text_chunks.append(chunk)
                        response_placeholder.markdown(
                            f'<div class="assistant-bubble">{"".join(full_text_chunks)}</div>',
                            unsafe_allow_html=True,
                        )
                    assistant_text = "".join(full_text_chunks).strip()

                if assistant_text:
                    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                # Mark this image as processed so we don't trigger again on rerun
                st.session_state.auto_kickoff_image_id = image_id

# ---------------------------
# Chat input
# ---------------------------
user_prompt = st.chat_input("E.g. What plant is this? Is it healthy?")

# If a user sent a message, append and call the model (streaming)
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # ↓↓↓ ensure live bubbles render inside the chat column
    with chat_container:
        # Echo user bubble immediately
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{user_prompt}</div>', unsafe_allow_html=True)

        # Check client readiness
        client = ensure_client(api_key)
        if client:
            # Build messages: include history + latest user turn (with image if present)
            messages = build_messages(
                system_prompt=system_prompt,
                chat_history=st.session_state.messages[:-1],  # exclude the very latest we’ll rebuild
                user_text=user_prompt,
                image_b64=st.session_state.image_b64,
                image_mime=st.session_state.image_mime,
            )

            with st.chat_message("assistant"):
                    # 1) Show a loader bubble immediately
                    response_placeholder = st.empty()
                    full_text_chunks: List[str] = []
                    response_placeholder.markdown(
                        '<div class="assistant-bubble"><span class="loader"></span>Thinking…</div>',
                        unsafe_allow_html=True,
                    )

                    # 2) Stream tokens into a separate placeholder                  
                    for chunk in stream_chat_completion(
                        client=client,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=1200,
                    ):
                        full_text_chunks.append(chunk)
                        response_placeholder.markdown(
                            f'<div class="assistant-bubble">{"".join(full_text_chunks)}</div>',
                            unsafe_allow_html=True,
                        )
                    assistant_text = "".join(full_text_chunks).strip()

            # Persist assistant turn
            if assistant_text:
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
