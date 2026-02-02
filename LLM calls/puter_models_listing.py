import os
import time
import pandas as pd
from dotenv import load_dotenv
from puter import PuterAI  # Assuming this matches your local SDK wrapper
from prompt_hub import Dilemma_types, Prompt_Types

load_dotenv()

# --- Configuration & Initialization ---
USERNAME = os.getenv("PUTER_USERNAME")
PASSWORD = os.getenv("PUTER_PASSWORD")
Model_family = "deepseek"

puter_ai = PuterAI(username=USERNAME, password=PASSWORD)
all_results = []

# --- Authentication ---
if not puter_ai.login():
    print("❌ Login failed. Check your credentials.")
    exit()
try:
    # 1. Fetch the models (this returns a list of strings)
    available_models = puter_ai.get_available_models()
    
    # 2. Filter for Models (no need for ['id'] access)
    models = [m for m in available_models if Model_family in m.lower()]
    
    if not models:
        print(f"⚠️ No {Model_family} models found. Check if the family name is correct.")
        # Optional: print first 5 to see what the format looks like
        print(f"Debug - First 5 models: {available_models[:5]}")
        exit()
        
    print(f"🔍 Found {len(models)} {Model_family} models: {', '.join(models)}")

except Exception as e:
    print(f"❌ Error fetching models: {e}")
    exit()