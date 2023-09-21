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



import pdfplumber
from bs4 import BeautifulSoup
import urllib.request
from io import BytesIO

# %%  whisper implementation 
import os 
import openai
import faiss
import tempfile

from moviepy.editor import *
from pytube import YouTube
from urllib.parse import urlparse, parse_qs

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
# %% Open Ai key 
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = ""




# %% dowload from gdrive 
def download_file_from_google_drive(file_id):
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(URL)
    if response.status_code == 200:
        return response.content
    else:
        return None
# %% Transcribe audio 
def transscribe_audio(file_path):
    file_size = os.path.getsize(file_path)
    file_size_in_mb = file_size / (1024 * 1024)
    if file_size_in_mb < 25:
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

        return transcript
    else:
        print("Please provide a smaller audio file (max 25mb).")

def divide_segments():
    return

#%% transcrip 
def transcript(URL): 
    # Get YouTube video URL from user
    
    # Extract the video ID from the url
    query = urlparse(URL).query
    params = parse_qs(query)
    video_id = params["v"][0]

    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Download video audio
        yt = YouTube(URL)

        # Get the first available audio stream and download this stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download(output_path=temp_dir)

        # Convert the audio file to MP3
        audio_path = os.path.join(temp_dir, audio_stream.default_filename)
        audio_clip = AudioFileClip(audio_path)
        audio_clip.write_audiofile(os.path.join(temp_dir, f"{video_id}.mp3"))

        # Keep the path of the audio file
        audio_path = f"{temp_dir}/{video_id}.mp3"

        # Transscripe the MP3 audio to text
        transcript = transscribe_audio(audio_path)
        
        # Delete the original audio file
        os.remove(audio_path)
        return transcript.text
        
# %% chunchs engines 

def chucks_engine(text): 


    #langchain_textspliter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text=text)

    
    #store pdf name
    store_name = "test"
    
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
# Api key 
os.environ["OPENAI_API_KEY"] = ""
#%% Side bar definitions 
with st.sidebar:
    st.title('Side Bar')
    st.write('Side Bar content')
# %% response 
def get_response(chain, vectorstore,query):
    docs = vectorstore.similarity_search(query=query,k=3)
    response = chain.run(input_documents = docs, question = query)
    return response 
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

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Respond your answer in mardkown format.")
        ]

def main2():

    st.header("test whisper")
    URL = st.text_input('Youtube URL','')
    print(URL)

    if URL != '':
        text = transcript(URL)
        
        vectorstore = chucks_engine(text)
        init_messages()

    
        if user_input := st.chat_input("Input your question!"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("MinckaGPT is typing ..."):
                answer = get_response_RQA(vectorstore,  user_input)
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