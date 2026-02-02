from groq import Groq
from prompt_hub import Dilemma_types, Prompt_Types
from dotenv import load_dotenv
import os
import time
import pandas as pd

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))
model_name = os.getenv("GROQ_QWEN3_32B")
all_results = []

for prompt_item in Prompt_Types:
    for dilemma in Dilemma_types:
    
        prompt = f"{dilemma.value} \n {prompt_item.value}"
        # print(prompt)
        
        start_time = time.time()
        full_response = ""
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True
        )
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            full_response += content


        inference_time = round(time.time() - start_time, 2)

        row = {
            "model_name": model_name,
            "dilemma_type": dilemma.name,
            "prompt_type" : prompt_item.name,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "response": full_response,
            "response_length": len(full_response),
            "inference_time": inference_time,
            "api_source": "groq",
            "temperature" : 0.7,
        }
        all_results.append(row)


df_new = pd.DataFrame(all_results)
file_name = "qwen3_32b.xlsx"

if os.path.exists(file_name):
    df_existing = pd.read_excel(file_name)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_excel(file_name, index=False)
    print(f"✅ Appended to {file_name}")
else:
    df_new.to_excel(file_name, index=False)
    print(f"✅ Created new file: {file_name}")
