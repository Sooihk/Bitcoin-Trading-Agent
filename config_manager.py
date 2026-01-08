import os
import json
import pickle
from dotenv import load_dotenv
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

load_dotenv()

class ConfigManager:
    def __init__(self):
        self.sheet_id = os.getenv("GOOGLE_SHEET_ID")
        self.local_cache = "config_cache.json"
        self.config = {}

    def load_google_sheet(self):
        creds = None
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/gmail.send']

        # Load token if exists
        if os.path.exists('token.pkl'):
            with open('token.pkl', 'rb') as token:
                creds = pickle.load(token)

        # Refresh or request new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'client_secrets.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pkl', 'wb') as token:
                pickle.dump(creds, token)

        # Connect to Google Sheets
        client = gspread.authorize(creds)
        sheet = client.open_by_key(self.sheet_id).sheet1
        self.config = sheet.get_all_records()[0]
        self._cache_to_local()

    def _cache_to_local(self):
        with open(self.local_cache, "w") as f:
            json.dump(self.config, f, indent=2)

    def load_cached(self):
        if os.path.exists(self.local_cache):
            with open(self.local_cache, "r") as f:
                self.config = json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

# Test block
if __name__ == "__main__":
    cm = ConfigManager()
    try:
        cm.load_google_sheet()
        print("Loaded from Google Sheet:", cm.config)
    except Exception as e:
        print("Google Sheet failed, loading from cache:", e)
        cm.load_cached()
        print("Loaded config from cache:", cm.config)