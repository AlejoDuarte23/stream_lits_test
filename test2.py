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
os.environ["LANGCHAIN_TRACING"] = "true"
#%% Api keys
openai_api_key = ""
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools, initialize_agent, AgentType,Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain import LLMMathChain

# SerpAPI_api_key=

#%% Tools 
# DuckDuck
search = DuckDuckGoSearchRun()
# Python repl
python_repl = PythonREPL()

llm = ChatOpenAI(temperature=0.5,openai_api_key=openai_api_key,model_name="gpt-3.5-turbo-16k-0613", streaming=True )
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)


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
    


   


#%% streaming 

            
#%%             


# tools = [
#     Tool(
#         name="Intermediate Answer",
#         func=search.run,
#         description="useful for searching the web",
#     )
# ]
    
# tools.append(python_repl)




# tools = load_tools(["serpapi"], llm=llm)
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    # agent="zero-shot-react-description",
    # agent=AgentType.SELF_ASK_WITH_SEARCH,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=10,
)


#%% app


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

from langchain.callbacks import StreamlitCallbackHandler

def init():
    st.set_page_config(
        page_title="MinckatGPT",
    )

def main():
    init()
    
    # Create StreamlitCallbackHandler instance
    st_callback = StreamlitCallbackHandler(st.container())

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=content)
        ]

    st.header("MinckaGPT")

    # sidebar with user input
    with st.container():
        user_input = st.text_input("Your message: ", key="user_input")

        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = zero_shot_agent.run(st.session_state.messages,callbacks=[st_callback])
            st.session_state.messages.append(
                AIMessage(content=response))
            
            # If you want to use the StreamlitCallbackHandler, you may need to link it here.

    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user', avatar_style='no-avatar')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai', avatar_style='no-avatar')

if __name__ == '__main__':
    main()