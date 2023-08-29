import streamlit as st
from streamlit_chat import message
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

openai_api_key = st.secrets["OPENAI_API_KEY"]




def init():

    st.set_page_config(
        page_title="MinckatGPT",

    )

def main():
    init()

    chat = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("MinckaGPT")

    # sidebar with user input
    with st.container():
        user_input = st.text_input("Your message: ", key="user_input")

        # handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(
                AIMessage(content=response.content))

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user', avatar_style='no-avatar')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai', avatar_style='no-avatar')



if __name__ == '__main__':
    main()