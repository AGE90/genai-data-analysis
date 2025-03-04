"""
Module for loading credentials
"""
import os

from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()

# Load environment variables from .env file
GENAI_API_KEY = os.getenv("GENAI_API_KEY") # Google AI Studio API key
SA_CREDS_PATH = os.getenv("SA_CREDS_PATH") # Service account credentials path


# Load credentials
vertex_app_creds = service_account.Credentials.from_service_account_file(
    SA_CREDS_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
