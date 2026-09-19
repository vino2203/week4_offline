import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Spotify Architecture Chat", page_icon="🎵", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "pending_upload" not in st.session_state:
    st.session_state.pending_upload = None
if "upload_error" not in st.session_state:
    st.session_state.upload_error = None


def get_status() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        response.raise_for_status()
        return {"reachable": True, **response.json()}
    except requests.RequestException:
        return {"reachable": False, "has_document": False, "current_document": None}


def run_ingestion(filename: str, file_bytes: bytes) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/ingest",
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=300,
    )
    if response.status_code != 200:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(detail)
    return response.json()


def ask_question(question: str) -> dict:
    response = requests.post(f"{API_BASE_URL}/chat", json={"question": question}, timeout=180)
    if response.status_code != 200:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(detail)
    return response.json()


# --- Blocking upload flow: while a document is being processed, this is the
# *entire* page. Nothing else (sidebar, chat) renders until it finishes, so
# there's no way to interact with a stale/half-updated UI mid-ingestion. ---
if st.session_state.processing:
    st.title("🎵 Spotify Architecture Chat")
    st.warning("📤 Document upload is processing — please wait until it finishes.")

    filename, file_bytes = st.session_state.pending_upload
    try:
        with st.status(f"Processing '{filename}'... this may take a minute", expanded=True) as status:
            result = run_ingestion(filename, file_bytes)
            status.write(f"Loaded {result['pages']} page(s) from {result['filename']}")
            status.update(label="Document processed", state="complete")
        st.session_state.messages = []
        st.session_state.upload_error = None
    except Exception as exc:
        st.session_state.upload_error = str(exc)
    finally:
        st.session_state.processing = False
        st.session_state.pending_upload = None
        st.rerun()

else:
    status_info = get_status()

    with st.sidebar:
        st.header("📄 Document")

        if not status_info["reachable"]:
            st.error(f"Can't reach the backend API at {API_BASE_URL}")
        elif status_info["current_document"]:
            st.caption(f"Active: {status_info['current_document']}")
        elif status_info["has_document"]:
            st.caption("Active: previously ingested document")
        else:
            st.caption("No document ingested yet")

        if st.session_state.upload_error:
            st.error(f"Failed to process document: {st.session_state.upload_error}")

        with st.expander("➕ Upload document"):
            uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
            if uploaded_file is not None and st.button("Process document", type="primary"):
                st.session_state.pending_upload = (uploaded_file.name, uploaded_file.getvalue())
                st.session_state.processing = True
                st.session_state.upload_error = None
                st.rerun()

    st.title("🎵 Spotify Architecture Chat")

    if not status_info["reachable"]:
        st.error(f"Backend API is unreachable at {API_BASE_URL}. Start it and refresh this page.")
        st.stop()

    if not status_info["has_document"]:
        st.info("Upload a document using the sidebar to get started.")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("details"):
                with st.expander("Retrieval details"):
                    st.json(message["details"])

    question = st.chat_input("Ask about the document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            details = None
            with st.spinner("Thinking..."):
                try:
                    result = ask_question(question)
                    answer = result["answer"]
                    details = {
                        "retrieval_attempts": result["retrieval_attempts"],
                        "generation_attempts": result["generation_attempts"],
                        "generation_grade": result["generation_grade"],
                        "out_of_scope": result["out_of_scope"],
                    }
                except Exception as exc:
                    answer = f"Something went wrong answering that: {exc}"

            st.markdown(answer)
            if details:
                with st.expander("Retrieval details"):
                    st.json(details)

        st.session_state.messages.append({"role": "assistant", "content": answer, "details": details})
