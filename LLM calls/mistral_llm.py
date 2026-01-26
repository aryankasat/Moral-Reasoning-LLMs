import os
import time
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral
from prompt_hub import Dilemma_types, Prompt_Types

# Load environment variables
load_dotenv()

# Initialize Mistral client
# Ensure MISTRAL_API_KEY and MISTRAL_MODEL (e.g., "open-mistral-7b") are in your .env
api_key = os.getenv("MISTRAL_API_KEY")
model_name = os.getenv("MISTRAL_MISTRAL_TINY_MODEL")
client = Mistral(api_key=api_key)

all_results = []

# Iterate through dilemmas and prompts exactly like the Groq script
for prompt_item in Prompt_Types:
    for dilemma in Dilemma_types:
        
        prompt = f"{dilemma.value} \n {prompt_item.value}"
        
        start_time = time.time()
        full_response = ""
        
        # Using Mistral's streaming chat completion
        stream_response = client.chat.stream(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        
        for chunk in stream_response:
            content = chunk.data.choices[0].delta.content or ""
            full_response += content

        inference_time = round(time.time() - start_time, 2)

        # Structure the data row
        row = {
            "model_name": model_name,
            "dilemma_type": dilemma.name,
            "prompt_type" : prompt_item.name,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "response": full_response,
            "response_length": len(full_response),
            "inference_time": inference_time,
            "api_source": "mistral",
            "temperature" : 0.7,
        }
        all_results.append(row)

# DataFrame handling and Excel Export
df_new = pd.DataFrame(all_results)
file_name = "mistral_tiny.xlsx" # Named differently to avoid overwriting your Groq results

if os.path.exists(file_name):
    df_existing = pd.read_excel(file_name)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_excel(file_name, index=False)
    print(f"✅ Appended to {file_name}")
else:
    df_new.to_excel(file_name, index=False)
    print(f"✅ Created new file: {file_name}")