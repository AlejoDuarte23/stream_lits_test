from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import json

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
        pageSize=5,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    files = response.get('files', [])

    for file in files:
        result_str = f"{file['name']} - https://drive.google.com/file/d/{file['id']}/view?usp=drive_link"

        final_results.append(result_str)

    return final_results

# Example usage
if __name__ == '__main__':
    query = "conveyor"  # Replace with your query
    results = search_gdrive_files(query)
    print(results)