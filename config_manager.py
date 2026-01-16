
import os
import json
import pickle
from dotenv import load_dotenv
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

load_dotenv() # Load environment variables from .env file
"""
script for securely managing configuration settings by loading them from a Google Sheet using OAuth2 authentication.
It caches the settings locally in a JSON file for offline access.

The code does the following:
1. Authenticates the user with Google using OAuth (Sheets + Gmail access)
2. Connects to a Google Sheet (via ID from .env)
3. Reads configuration values (vertical format: key/value pairs)
4. Parses them into proper Python types (bools, ints, floats, strings)
5. Saves the config to a local cache (config_cache.json)
6. Allows fallback to cached config if Google API is unavailable
"""
class ConfigManager:
    """
    Manages configuration settings loaded from a Google Sheet and cached locally.
    """
    def __init__(self):
        # pull sheet ID from environment variable
        self.sheet_id = os.getenv("GOOGLE_SHEET_ID")
        # Initialize local cache file path
        self.local_cache = "config_cache.json"
        # Initialize empty config dictionary
        self.config = {}

    def _parse_value(self, value):
        """
        Converts string values to appropriate Python types (bool, int, float, str).
        """
        if isinstance(value, str):
            value = value.strip()
            if value.lower() == "true":
                return True
            if value.lower() == "false":
                return False
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value
        return value

    def load_google_sheet(self):
        creds = None
        # set up OAuth2 scopes
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/gmail.send']

        # Load token if exists
        if os.path.exists('token.pkl'):
            with open('token.pkl', 'rb') as token:
                creds = pickle.load(token)

        # Refresh or request new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            # otherwise, launch OAuth browser to login flow
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'client_secrets.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pkl', 'wb') as token:
                pickle.dump(creds, token)

        # Connect to Google Sheets
        client = gspread.authorize(creds)
        sheet = client.open_by_key(self.sheet_id).sheet1
        records = sheet.get_all_values() # retrieve all records
        # build up the config dictionary, convert rows in dictionary
        self.config = {row[0]: self._parse_value(row[1]) for row in records if len(row) >= 2}
        self._cache_to_local() # write config dictionary to local cache

    def _cache_to_local(self):
        # dumps current config as formmatted JSON to local cache file
        with open(self.local_cache, "w") as f:
            json.dump(self.config, f, indent=2)

    def load_cached(self):
        # loads previously cached config from local JSON file
        if os.path.exists(self.local_cache):
            with open(self.local_cache, "r") as f:
                self.config = json.load(f)

    def get(self, key, default=None):
        # Retrieves a specific config value by key (like get("DCA_AMOUNT_USD"))
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