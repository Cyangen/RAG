import streamlit as st

from graph import workflow

st.title("Starter Pack Chatbot")
st.markdown("by Kredivo")


def stream_data():
    for msg, metadata in workflow.stream(
        {"question": prompt,}, stream_mode="messages",
    ):
        # print(metadata)
        if metadata["langgraph_node"] == "responder":
            stream = msg.content

            yield stream


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        print(prompt)

        response = st.write_stream(stream_data)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
