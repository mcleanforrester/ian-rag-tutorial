import streamlit as st
import httpx

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

if prompt := st.chat_input("Ask a question about your department's documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = httpx.post(
                f"{API_BASE}/chat",
                json={"query": prompt, "user_id": selected_user_id},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

        st.markdown(data["response"])

        policy_summary = data.get("policy_summary")
        if policy_summary:
            with st.expander("Policy Summary"):
                st.markdown(f"**{policy_summary['title']}** ({policy_summary['department']})")
                st.markdown(f"Effective: {policy_summary['effective_date']}")
                for point in policy_summary["key_points"]:
                    st.markdown(f"- {point}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": data["response"],
        "policy_summary": policy_summary,
    })
