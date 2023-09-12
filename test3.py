# app.py
# from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
import streamlit as st
from langchain.chat_models import ChatOpenAI
import os 
from PIL import Image, ImageDraw
image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

os.environ["OPENAI_API_KEY"] = ""


avatar_bot = Image.open(r"utils\logo_gold_Icon.png")
avatar_user = Image.open(r"utils\logo_white_icon.png")


def init_page():
    st.set_page_config(
        page_title="MinckaGPT"
    )
    
    st.markdown(
        f"""
        <style>
            body {{
                background-color: #cd0000;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    logo_image = Image.open(r"utils\White_Logo_Black_text_Transparent.png")
    st.image(logo_image, width=300, channels="RGB")

    st.header("MinckaGPT")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Respond your answer in mardkown format.")
        ]
        st.session_state.costs = []


def select_model():
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gpt-3.5-turbo-0613", "gpt-4"))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    return model_name,temperature


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost


def main():

    init_page()
    model_name,temperature = select_model()
    llm = ChatOpenAI(temperature=0.5)

    init_messages()

    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("MinckaGPT is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar=avatar_bot):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user", avatar=avatar_user):
                st.markdown(message.content)

    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == "__main__":
    main()