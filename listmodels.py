import os
import google.generativeai as gpt
from dotenv import load_dotenv

def list_available_models():
    # Load environment variables from .env
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")

    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return

    # Configure the API
    gpt.configure(api_key=API_KEY)

    try:
        # List models
        models = gpt.list_models()
        print("Available Models:")
        for model in models:
            print(f"Model Name: {model.name}, Supported Methods: {model.supported_methods}")
    except Exception as e:
        print(f"An error occurred while listing models: {e}")

if __name__ == "__main__":
    list_available_models()
