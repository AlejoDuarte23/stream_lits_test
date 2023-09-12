# app.py
# from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
import streamlit as st
from langchain.chat_models import ChatOpenAI
import os 
from PIL import Image, ImageDraw
from langchain.utilities import PythonREPL

image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
# %% api key 

os.environ["OPENAI_API_KEY"] = ""
openai_api_key = ""
# %% Shot agent 
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools, initialize_agent, AgentType,Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain import LLMMathChain


#%% Tools 
# DuckDuck
search = DuckDuckGoSearchRun()
# Python repl
python_repl = PythonREPL()

_llm = ChatOpenAI(temperature=0.5,openai_api_key=openai_api_key,model_name="gpt-3.5-turbo-16k-0613", streaming=True )
llm_math_chain = LLMMathChain.from_llm(llm=_llm, verbose=True)


tools = [  Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="useful for when you need to answer questions about current events"
),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )]

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    # agent="zero-shot-react-description",
    # agent=AgentType.SELF_ASK_WITH_SEARCH,
    tools=tools,
    llm=_llm,
    verbose=True,
    max_iterations=10,
)
    

content = """Complete the objective as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:



Begin!"""




# %% logo 
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
                                  ("gpt-3.5-turbo-0613", "gpt-4", "zero_shot_agent"))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    return model_name,temperature


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost

def get_answer_agent( messages,st_callback):
    response = zero_shot_agent.run(st.session_state.messages,callbacks=[st_callback])

    return response



def main():

    init_page()
    st_callback = StreamlitCallbackHandler(st.container())

    model_name,temperature = select_model()
    if model_name != "zero_shot_agent": 
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
    else:
        init_messages()
    
        # Supervise user input
        if user_input := st.chat_input("Input your question!"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("MinckaGPT is typing ..."):
                answer = get_answer_agent(st.session_state.messages,st_callback)
            st.session_state.messages.append(AIMessage(content=answer))
    
        # Display chat history
        messages = st.session_state.get("messages", [])
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar=avatar_bot):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user", avatar=avatar_user):
                    st.markdown(message.content)
    

        
    

            
        
# zero_shot_agent.run(

if __name__ == "__main__":
    main()