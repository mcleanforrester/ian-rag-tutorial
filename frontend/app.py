import json

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", layout="wide")


@st.cache_data
def fetch_users():
    response = httpx.get(f"{API_BASE}/users")
    response.raise_for_status()
    return response.json()


# --- Sidebar: User Selection ---
with st.sidebar:
    st.header("User")

    users = fetch_users()
    user_options = {
        f"{u['name']} ({u['department']}, {u['permission_level']})": u["id"]
        for u in users
    }

    selected_label = st.selectbox("Logged in as:", list(user_options.keys()))
    selected_user_id = user_options[selected_label]

    selected_user = next(u for u in users if u["id"] == selected_user_id)
    st.caption(f"Department: **{selected_user['department']}**")
    st.caption(f"Permission: **{selected_user['permission_level']}**")

# --- Clear chat on user switch ---
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = selected_user_id
    st.session_state.messages = []

if st.session_state.current_user_id != selected_user_id:
    st.session_state.current_user_id = selected_user_id
    st.session_state.messages = []
    st.rerun()

# --- Main: Chat Interface ---
st.title("RAG Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("policy_summary"):
            with st.expander("Policy Summary"):
                ps = msg["policy_summary"]
                st.markdown(f"**{ps['title']}** ({ps['department']})")
                st.markdown(f"Effective: {ps['effective_date']}")
                for point in ps["key_points"]:
                    st.markdown(f"- {point}")

def _parse_sse_events(stream):
    """Yield (event_type, data) tuples from an SSE byte stream."""
    event_type = "message"
    for line in stream.iter_lines():
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data = line[len("data:"):].strip()
            yield event_type, data
            event_type = "message"


def _stream_tokens(user_id, query):
    """Stream tokens from /chat/stream, yielding text chunks for st.write_stream.

    Stores policy_summary in session state as a side effect.
    """
    st.session_state._pending_summary = None
    with httpx.stream(
        "POST",
        f"{API_BASE}/chat/stream",
        json={"query": query, "user_id": user_id},
        timeout=60.0,
    ) as response:
        response.raise_for_status()
        for event_type, data in _parse_sse_events(response):
            if event_type == "token":
                yield data
            elif event_type == "summary":
                st.session_state._pending_summary = json.loads(data)
            elif event_type == "done":
                return


if prompt := st.chat_input("Ask a question about your department's documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = st.write_stream(_stream_tokens(selected_user_id, prompt))

        policy_summary = st.session_state.pop("_pending_summary", None)
        if policy_summary:
            with st.expander("Policy Summary"):
                st.markdown(f"**{policy_summary['title']}** ({policy_summary['department']})")
                st.markdown(f"Effective: {policy_summary['effective_date']}")
                for point in policy_summary["key_points"]:
                    st.markdown(f"- {point}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "policy_summary": policy_summary,
    })
