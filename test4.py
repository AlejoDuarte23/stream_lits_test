# app.py
# from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
import streamlit as st
from langchain.chat_models import ChatOpenAI
import os 
from PIL import Image, ImageDraw
from langchain.utilities import PythonREPL
from langchain.llms import OpenAI

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


import pdfplumber
from bs4 import BeautifulSoup
import urllib.request
from io import BytesIO


#%% Tools 
# DuckDuck
search = DuckDuckGoSearchRun()
# Python repl
python_repl = PythonREPL()

_llm = ChatOpenAI(temperature=0.5,openai_api_key=openai_api_key,model_name="gpt-4", streaming=True )
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


# %%  PDF LOADER
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pickle
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
import base64
import urllib.request
def chucks_engine(pdf): 
    st.write(pdf.name)

    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()

    #langchain_textspliter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text=text)

    
    #store pdf name
    store_name = pdf.name[:-4]
    
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl","rb") as f:
            vectorstore = pickle.load(f)
        #st.write("Already, Embeddings loaded from the your folder (disks)")
    else:
        #embedding (Openai methods) 
        embeddings = OpenAIEmbeddings()

        #Store the chunks part in db (vector)
        vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

        with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(vectorstore,f)
    return vectorstore

def get_response_RQA(vectorstore,query):
    # docs = vectorstore.similarity_search(query=query,k=3)
    
    llm = OpenAI(temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type = "refine",
    retriever=vectorstore.as_retriever())
    # chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    # result = qa_chain({"query": query})
    return qa_chain.run(query)#result["result"]

# def displayPDF(file):
#     # Reading file directly
#     base64_pdf = base64.b64encode(file.read()).decode('utf-8')

#     # Embedding PDF in HTML
#     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="950" type="application/pdf"></iframe>'

#     # Displaying File
#     st.markdown(pdf_display, unsafe_allow_html=True)


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
                                  ("gpt-3.5-turbo-0613", "gpt-4", "zero_shot_agent","PDF Loader"))
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


def get_general_answer(method, *args, **kwargs):
    if method == "llm":
        return get_answer(*args, **kwargs)
    elif method == "agent":
        return get_answer_agent(*args, **kwargs)
    elif method == "PDF Loader":
        return get_response_RQA(*args, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

def pdf_to_html_display(pdf_data):
    html = BeautifulSoup("<html><head></head><body></body></html>", "html.parser")
    head = html.head
    body = html.body

    # Add some CSS styling within <style> tags
    style = html.new_tag("style")
    style.string = """
    .outer-container {
        background-color: #272731;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 800px;
    }
    .outer-container .pdf-container {
        width: 600px;
        height: 800px;
        overflow-y: scroll;
        padding: 10px;
        background-color: #272731;
    }
    .outer-container .pdf-container p {
        font-family: Arial, sans-serif;
        line-height: 1.5;
        text-align: justify;
        # color: white;
    }
    .outer-container .pdf-container .page-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    """
    head.append(style)

    # Create an outer div to hold the background color
    outer_container = html.new_tag("div", **{"class": "outer-container"})
    body.append(outer_container)

    # Create a div to hold the PDF content
    pdf_container = html.new_tag("div", **{"class": "pdf-container"})
    outer_container.append(pdf_container)

    with pdfplumber.open(BytesIO(pdf_data)) as pdf:
        for i, page in enumerate(pdf.pages):
            # Add a title using the page number
            title = html.new_tag("div", **{"class": "page-title"})
            title.string = f"Page {i + 1}"
            pdf_container.append(title)

            # Extract and add the text content
            text = page.extract_text()
            p = html.new_tag("p")
            p.string = text
            pdf_container.append(p)

    html_content = str(html)
    st.markdown(html_content, unsafe_allow_html=True)
    
def displayPDF(file):
    if isinstance(file, str):
        with urllib.request.urlopen(file) as f:
            pdf_data = f.read()
    else:
        pdf_data = file.read()
        
    pdf_to_html_display(pdf_data)
    
    
def main():
    init_page()

    model_name, temperature = select_model()

    if model_name == "PDF Loader":
        pdf = st.file_uploader("Upload your PDF", type='pdf')
        if pdf is not None:
            displayPDF(pdf)
            pdf.seek(0)  # Reset file pointer for further processing
            vectorstore = chucks_engine(pdf)

    st_callback = StreamlitCallbackHandler(st.container())
    init_messages()
    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("MinckaGPT is typing ..."):
            if model_name in ["gpt-3.5-turbo-0613", "gpt-4"]:
                answer, cost = get_general_answer("llm", ChatOpenAI(model_name=model_name, temperature=temperature), st.session_state.messages)
                st.session_state.costs.append(cost)
            elif model_name == "zero_shot_agent":
                answer = get_general_answer("agent", st.session_state.messages, st_callback)
            # Uncomment the line below when you add RQA
            else:
                answer = get_general_answer("PDF Loader", vectorstore, user_input)
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

    # Display costs
    if 'costs' in st.session_state:
        costs = st.session_state.get("costs", [])
        st.sidebar.markdown("## Costs")
        st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
        for cost in costs:
            st.sidebar.markdown(f"- ${cost:.5f}") 

            
        
# zero_shot_agent.run(

if __name__ == "__main__":
    main()