#coding part
import streamlit as st
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

import base64
from langchain.callbacks import get_openai_callback
import base64
import pickle
import os
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.prompts import PromptTemplate
import urllib.request

os.environ["OPENAI_API_KEY"] = ""


import pdfplumber
from bs4 import BeautifulSoup
import urllib.request
from io import BytesIO




# %% tempate 
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain

# %% Init sesion messages 
def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Respond your answer in mardkown format.")
        ]
# %% chunk engine
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
    

#%% Side bar definitions 
with st.sidebar:
    st.title('Side Bar')
    st.write('Side Bar content')
# %%  Chain definition
def llm_engine():
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm=llm, chain_type= "stuff")
    return chain

# %% response 
def get_response(chain, vectorstore,query):
    docs = vectorstore.similarity_search(query=query,k=3)
    response = chain.run(input_documents = docs, question = query)
    return response 
def get_response_RQA(vectorstore,query,QA_CHAIN_PROMPT):
    # docs = vectorstore.similarity_search(query=query,k=3)
    
    llm = OpenAI(temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type = "refine",
    retriever=vectorstore.as_retriever())
    # chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    # result = qa_chain({"query": query})
    return qa_chain.run(query)#result["result"]

#%%  Main
def main():
    st.header("test pdf reader")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    #  chuncks engine
    if pdf is not None:
        vectorstore = chucks_engine(pdf)
        # chain = llm_engine()


        init_messages()
    
        # Supervise user input
        if user_input := st.chat_input("Input your question!"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("MinckaGPT is typing ..."):
                # answer = get_response(chain, vectorstore,user_input)
                answer = get_response_RQA(vectorstore,user_input,QA_CHAIN_PROMPT)
            st.session_state.messages.append(AIMessage(content=answer))
    
        # Display chat history
        messages = st.session_state.get("messages", [])
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)





# def displayPDF(file):
#     # Opening file from file path. this is used to open the file from a website rather than local
#     with urllib.request.urlopen(file) as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     # Embedding PDF in HTML
#     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="950" type="application/pdf"></iframe>'

#     # Displaying File
#     st.markdown(pdf_display, unsafe_allow_html=True)

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

# def main2():
#     st.header("test pdf reader")
#     pdf = st.file_uploader("Upload your PDF", type='pdf')

#     if pdf is not None:
#         displayPDF(pdf)
#         pdf.seek(0)  # Reset file pointer for further processing
#         vectorstore = chucks_engine(pdf)

#         init_messages()
    
#         if user_input := st.chat_input("Input your question!"):
#             st.session_state.messages.append(HumanMessage(content=user_input))
#             with st.spinner("MinckaGPT is typing ..."):
#                 answer = get_response_RQA(vectorstore, user_input, QA_CHAIN_PROMPT)
#             st.session_state.messages.append(AIMessage(content=answer))

#         messages = st.session_state.get("messages", [])
#         for message in messages:
#             if isinstance(message, AIMessage):
#                 with st.chat_message("assistant"):
#                     st.markdown(message.content)
#             elif isinstance(message, HumanMessage):
#                 with st.chat_message("user"):
#                     st.markdown(message.content)

def main2():
    st.header("test pdf reader")
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        displayPDF(pdf)
        pdf.seek(0)  # Reset file pointer for further processing
        vectorstore = chucks_engine(pdf)

        init_messages()
    
        if user_input := st.chat_input("Input your question!"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("MinckaGPT is typing ..."):
                answer = get_response_RQA(vectorstore, user_input, QA_CHAIN_PROMPT)
            st.session_state.messages.append(AIMessage(content=answer))

        messages = st.session_state.get("messages", [])
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)

if __name__=="__main__":
    main2()