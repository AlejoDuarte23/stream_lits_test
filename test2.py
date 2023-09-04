import streamlit as st
import os
from streamlit_chat import message
from langchain.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.utilities import SerpAPIWrapper
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun

from langchain.utilities import PythonREPL

#%% Api keys
openai_api_key = ""
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools, initialize_agent, AgentType,Tool


# SerpAPI_api_key=

#%% Tools 
# DuckDuck
search = DuckDuckGoSearchRun()
# Python repl
python_repl = PythonREPL()

# tools = [  Tool(
#     name='DuckDuckGo Search',
#     func= search.run,
#     description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
# )]
    

# python_repl = Tool(
#     name = "python repl",
#     func=python_repl.run,
#     description="useful for when you need to use python to answer a question. You should input python code or math questions"
# )
    

tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for searching the web",
    )
]
    
# tools.append(python_repl)


llm = OpenAI(temperature=0.5,openai_api_key=openai_api_key,model_name="gpt-4" )


# tools = load_tools(["serpapi"], llm=llm)
self_ask_with_search = initialize_agent(
    # agent="zero-shot-react-description",
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=10,
)


#%% app



def init():

    st.set_page_config(
        page_title="MinckatGPT",

    )

def main():
    init()

    # chat = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant that provides succint answers using clear language, you have some tool to search the web and ask yourself question whe providing answers clean them avoiding thirdparty information in the web just the content")
        ]

    st.header("MinckaGPT")

    # sidebar with user input
    with st.container():
        user_input = st.text_input("Your message: ", key="user_input")

        # handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response =self_ask_with_search.run(st.session_state.messages)
            st.session_state.messages.append(
                AIMessage(content=response))

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user', avatar_style='no-avatar')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai', avatar_style='no-avatar')



if __name__ == '__main__':
    main()