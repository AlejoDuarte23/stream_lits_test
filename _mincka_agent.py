# %% General imports 
import os 
import streamlit as st
import ast
import pickle
import pandas as pd 
# %% Agents imports 
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent , Tool ,create_csv_agent
from langchain.callbacks import StreamlitCallbackHandler
# %% txt llm 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.chains import RetrievalQA
# %% Google GLobal Search 
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import json


# TO DOP : GET FAISS DATA BASE OUTSIDE
# %% Open Ai key 
os.environ["OPENAI_API_KEY"] = ""
def create_mincka_agent():
    #  %% Create Faiss vector data base 
    # def chucks_engine_txt(file_name): 
    #     with open(file_name, 'r') as file:
    #         text = file.read()
            
    #     #langchain_textspliter
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size = 1000,
    #         chunk_overlap = 200,
    #         length_function = len
    #     )
    
    #     chunks = text_splitter.split_text(text=text)
    
        
    #     #store pdf name
    #     store_name = file_name[:-4]
        
    #     if os.path.exists(f"{store_name}.pkl"):
    #         with open(f"{store_name}.pkl","rb") as f:
    #             vectorstore = pickle.load(f)
    #         #st.write("Already, Embeddings loaded from the your folder (disks)")
    #     else:
    #         #embedding (Openai methods) 
    #         embeddings = OpenAIEmbeddings()
    
    #         #Store the chunks part in db (vector)
    #         vectorstore = FAISS.from_texts(chunks,embedding=embeddings)
    
    #         with open(f"{store_name}.pkl","wb") as f:
    #             pickle.dump(vectorstore,f)
    #     return vectorstore
    
    # %% Functions 
    #  Get Project names 
    def get_response_RQA_project_names(query):
        with open("projects_names.pkl","rb") as f:
            vectorstore = pickle.load(f)   
        llm = OpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "refine",
        retriever=vectorstore.as_retriever())
    
        return qa_chain.run(query)
    
    #  Get Project start dates 
    def get_project_start_dates(wrike_ids_str):
        # Convert the string to a list if it's a string
        if isinstance(wrike_ids_str, str):
            wrike_ids = ast.literal_eval(wrike_ids_str)
        else:
            wrike_ids = wrike_ids_str
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv("full_projects_data.csv")
        
        # Convert single Wrike ID to a list if needed
        if not isinstance(wrike_ids, list):
            wrike_ids = [wrike_ids]
            
        # Filter the DataFrame based on the Wrike IDs
        df_filtered = df[df['id'].isin(wrike_ids)]
        
        # Initialize the result list to store start date information
        result = []
        
        # Loop through the filtered DataFrame and construct the list of start dates
        for _, row in df_filtered.iterrows():
            project_id = row['id']
            start_date = row.get('startDate') if pd.notna(row.get('startDate')) else "this project doesn't have startDate assigned"
            
            date_info = start_date
            result.append(date_info)
            
        return result
    
    #  Get Project end dates 
    def get_project_end_dates(wrike_ids_str):
        # Convert the string to a list if it's a string
        if isinstance(wrike_ids_str, str):
            wrike_ids = ast.literal_eval(wrike_ids_str)
        else:
            wrike_ids = wrike_ids_str
    
        # Read the CSV file into a DataFrame
        df = pd.read_csv("full_projects_data.csv")
        
        # Convert single Wrike ID to a list if needed
        if not isinstance(wrike_ids, list):
            wrike_ids = [wrike_ids]
            
        # Filter the DataFrame based on the Wrike IDs
        df_filtered = df[df['id'].isin(wrike_ids)]
        
        # Initialize the result list to store end date information
        result = []
        
        # Loop through the filtered DataFrame and construct the list of end dates
        for _, row in df_filtered.iterrows():
            project_id = row['id']
            end_date = row.get('endDate') if pd.notna(row.get('endDate')) else "this project doesn't have endDate assigned"
            
            date_info = end_date
            result.append(date_info)
            
        return result
    
    
    def get_wrike_ids(project_names_str):
        
        # Convert the string representation of the list to an actual list
        try:
            project_names = ast.literal_eval(project_names_str)
        except:
            
            return "the input should be a list , return the information you have at the moment"
        
        df = pd.read_csv('wrike_projects_names_id.csv', sep=',')
        return df[df['Project_Names'].apply(lambda x: any(project_name in x for project_name in project_names))]['Wrike_ID'].tolist()
    
    
    # %% Golbal search google drvie 
    
    def search_gdrive_files(query):
        # Authenticate with Google Drive
        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = None
    
        if os.path.exists('token.json'):
            with open('token.json', 'r') as token_file:
                creds = Credentials.from_authorized_user_info(json.load(token_file), SCOPES)
    
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=8000)
    
            with open('token.json', 'w') as token_file:
                json.dump(json.loads(creds.to_json()), token_file)
    
        service = build('drive', 'v3', credentials=creds)
    
        # Initialize results list
        final_results = []
    
        # Perform the search
        query_str = f"(mimeType='application/pdf' or mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document') and (name contains '{query}' or fullText contains '{query}')"
        response = service.files().list(
            q=query_str,
            spaces='drive',
            fields='files(id, name)',
            pageSize=15,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
    
        files = response.get('files', [])
    
        for file in files:
            result_str = f"{file['name']} - https://drive.google.com/file/d/{file['id']}/view?usp=drive_link"
    
            final_results.append(result_str)
    
        return final_results
    
    
    # %% Function descriptions 
        
    description_pjid  = 'use full to determine the Wrike id  searching by Project_Names, requires first search in project_names  , input should be a a list separated by projec1,projec2....,input should be a a list "  '
    description_pjnmd  = 'use full to response question related to projects and projectlis base on querys and similarity search on them  '
    
    description_get_project_start_dates = 'Useful for finding the start dates of projects based on Wrike IDs. The function takes a single Wrike ID or a list of Wrike IDs as input. It queries a CSV file to match and return a list of lists, where each inner list contains the project ID and its start date.'
    description_get_project_end_dates = 'Useful for finding the end dates of projects based on Wrike IDs. The function takes a single Wrike ID or a list of Wrike IDs as input. It queries a CSV file to match and return a list of lists, where each inner list contains the project ID and its end date.'
    
    
    description_gd_global_search = 'Useful for looking information in google drive about projects, project titles,mincka codes, input is a query return is a list of files realted to the query '
    
    # %% Tools definitions 
    gd_global_search = Tool(
        name="global_search_gd",
        func= search_gdrive_files,
        description=description_gd_global_search,
        verbose=True,
    )
    
    
    get_wrike_ids_tool = Tool(
        name="get_wrike_ids",
        func=get_wrike_ids,
        description=description_pjid ,
        verbose=True,
    )
    
    
    project_names = Tool(
        name="project_names",
        func=get_response_RQA_project_names,
        description=description_pjnmd ,
        verbose=True,
    )
    
    start_dates_by_wrike_id = Tool(
        name="get_project_start_dates",
        func=get_project_start_dates,
        description=description_get_project_start_dates,
        verbose=True,
    )
    
    end_dates_by_wrike_id = Tool(
        name="get_project_end_dates",
        func=get_project_end_dates,
        description=description_get_project_end_dates,
        verbose=True,
    )
    
    
    
    tools = [get_wrike_ids_tool, project_names,end_dates_by_wrike_id,start_dates_by_wrike_id,gd_global_search]
    tool_names= {}
    tool_names["tool_names"] = ", ".join([tool.name for tool in tools])
    # %% Zero Shot Agent 
    _llm = ChatOpenAI(temperature=0,model_name="gpt-4" )
    # _llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo-0613", streaming=True )
    
    
    
    zero_shot_agent_mincka = initialize_agent(
        agent="zero-shot-react-description",
        tools=tools,
        llm=_llm,
        verbose=True,
        max_iterations=4,
    )
    
    template = """Answer the following questions as best you can. You have access to the following tools:
    
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of tool_names {['tool_names']}
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    IF you dont know the answer use Google drive related tool to provide files with general informations
    
    Important Respond your answer in mardkown format
    IF you dont know the answer use Google drive related tool to provide files with general informations
    
    
    """
    
    
    return  zero_shot_agent_mincka,template
    
