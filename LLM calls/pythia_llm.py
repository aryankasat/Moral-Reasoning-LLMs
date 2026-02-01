import os
import time
import torch
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from prompt_hub import Dilemma_types, Prompt_Types

load_dotenv()

# --- 2. MODEL SETUP ---
model_id = "EleutherAI/" + os.getenv("PYTHIA_1_4B")
revision_step = os.getenv("PYTHIA_143K_CHECKPOINT")

print(f"Loading model: {model_id} ({revision_step})...")
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision_step)
model = GPTNeoXForCausalLM.from_pretrained(model_id, revision=revision_step)

# Detect and use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on: {device.upper()}")

all_results = []

# --- 3. INFERENCE LOOP ---
for prompt_item in Prompt_Types:
    for dilemma in Dilemma_types:
        
        prompt_text = f"{dilemma.value} \n {prompt_item.value}"
        start_time = time.time()
        
        # Use torch.no_grad() to save memory/compute during inference
        with torch.no_grad():
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7
            )
            
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time = round(time.time() - start_time, 2)

        row = {
            "model_name": f"{model_id}-{revision_step}",
            "dilemma_type": dilemma.name,
            "prompt_type" : prompt_item.name,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "response": full_response,
            "response_length": len(full_response),
            "inference_time": inference_time,
            "api_source": "huggingface",
            "temperature" : 0.7,
        }
        all_results.append(row)
        print(f"Processed: {dilemma.name} | {prompt_item.name} ({inference_time}s)")


# --- 5. DATA EXPORT ---
df_new = pd.DataFrame(all_results)
file_name = f"{os.getenv('PYTHIA_1_4B')}-{revision_step}-results.xlsx"

if os.path.exists(file_name):
    df_existing = pd.read_excel(file_name)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_excel(file_name, index=False)
    print(f"✅ Appended results to {file_name}")
else:
    df_new.to_excel(file_name, index=False)
    print(f"✅ Created {file_name}")