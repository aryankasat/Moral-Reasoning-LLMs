import pandas as pd
import json
import os
import re

# Set your folder path here
folder_path = './' 

# 1. Pick all the excel files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        file_path = os.path.join(folder_path, filename)
        
        # Load the excel file
        df = pd.read_excel(file_path,engine= "openpyxl")
        
        # Check if the target column exists
        if 'raw_analysis_json' in df.columns:
            
            def extract_json_keys(row):
                cleaned = row.strip()
                if cleaned.startswith("```"):
                    # Remove opening ```json or ```
                    cleaned = re.sub(r'^```(?:json)?', '', cleaned)
                    # Remove closing ```
                    cleaned = re.sub(r'```$', '', cleaned)
                
                cleaned = cleaned.strip()
                try:
                    # 2. Pick json from the column
                    data = json.loads(cleaned)
                    # 3. Extract the specific keys
                    return data.get("stage_mix"), data.get("reasoning_quality")
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    return None, None

            # Apply the extraction
            df[['stage_mix', 'reasoning_quality']] = df['raw_analysis_json'].apply(
                lambda x: pd.Series(extract_json_keys(x))
            )
            
            # Save the updated excel (overwrites original or change filename)
            df.to_excel(file_path, index=False)
            print(f"Updated: {filename}")
        else:
            print(f"Skipped {filename}: Column 'raw_analysis_json' not found.")

print("Processing complete.")