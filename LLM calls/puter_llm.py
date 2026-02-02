import os
import time
import pandas as pd
from dotenv import load_dotenv
from puter import PuterAI
from prompt_hub import Dilemma_types, Prompt_Types

load_dotenv()

# --- Configuration & Initialization ---
USERNAME = os.getenv("PUTER_USERNAME")
PASSWORD = os.getenv("PUTER_PASSWORD")
MODEL_NAME = os.getenv("PUTER_DEEPSEEK_V3_1")  # Or your preferred model ID
FILE_NAME = "deepseek_v3_1.xlsx"

puter_ai = PuterAI(username=USERNAME, password=PASSWORD,timeout=300)
all_results = []

# --- Authentication ---
if not puter_ai.login():
    print("❌ Login failed. Check your credentials.")
    exit()

if not puter_ai.set_model(MODEL_NAME):
    print(f"❌ Failed to set model: {MODEL_NAME}")
    exit()

print(f"🚀 Starting inference with: {puter_ai.current_model}")

for prompt_item in Prompt_Types:
    for dilemma in Dilemma_types:
        
        prompt = f"{dilemma.value} \n {prompt_item.value}"
        
        start_time = time.time()
    
        full_response = puter_ai.chat(prompt)
        
        inference_time = round(time.time() - start_time, 2)

        row = {
            "model_name": MODEL_NAME,
            "dilemma_type": dilemma.name,
            "prompt_type" : prompt_item.name,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "response": full_response,
            "response_length": len(full_response) if full_response else 0,
            "inference_time": inference_time,
            "api_source": "puter",
            "temperature" : 0.7, 
        }
        
        all_results.append(row)
        print(f"✅ Processed: {dilemma.name} | {prompt_item.name}")

# --- Data Persistence ---
df_new = pd.DataFrame(all_results)

if os.path.exists(FILE_NAME):
    df_existing = pd.read_excel(FILE_NAME)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_excel(FILE_NAME, index=False)
    print(f"📊 Appended {len(df_new)} rows to {FILE_NAME}")
else:
    df_new.to_excel(FILE_NAME, index=False)
    print(f"📁 Created new file: {FILE_NAME}")