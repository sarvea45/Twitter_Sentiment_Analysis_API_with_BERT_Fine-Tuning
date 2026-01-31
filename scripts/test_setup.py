import os
import pandas as pd

try:
    print("Checking folders...")
    os.makedirs('data/processed', exist_ok=True)
    
    print("Creating a test file...")
    df = pd.DataFrame({'text': ['hello'], 'label': [1]})
    df.to_csv('data/processed/train.csv', index=False)
    
    if os.path.exists('data/processed/train.csv'):
        print("!!! SUCCESS: The file 'data/processed/train.csv' was created successfully !!!")
    else:
        print("??? ERROR: The file was not created for some reason ???")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
