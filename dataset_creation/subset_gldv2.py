import os
import pandas as pd
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO

# CONFIG
CSV_URL = "https://s3.amazonaws.com/google-landmark/metadata/train.csv"
OUTPUT_DIR = "data/raw/"
NUM_IMAGES = 200  # small subset for testing

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download metadata
print("Downloading metadata...")
df = pd.read_csv(CSV_URL)

# Pick a subset of rows
subset = df.sample(NUM_IMAGES, random_state=42)

print(f"Downloading {NUM_IMAGES} images...")
for idx, row in tqdm(subset.iterrows(), total=len(subset)):
    try:
        img_id = row['id']
        url = row['url']
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(os.path.join(OUTPUT_DIR, f"{img_id}.jpg"))
    except Exception as e:
        print(f"Failed: {row['id']} {e}")
