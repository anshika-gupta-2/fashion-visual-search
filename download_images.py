import pandas as pd
import os
import requests
import ast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# File paths
csv_path = r"C:\Users\91981\Downloads\Dresses\data\jeans_bd_processed_data.csv"
checkpoint_path = "data/jeans_with_image_paths.csv"

# Load CSV (check if checkpoint already exists)
if os.path.exists(checkpoint_path):
    df = pd.read_csv(checkpoint_path)
    print("📌 Resuming from checkpoint.")
else:
    df = pd.read_csv(csv_path)
    df['feature_image_path'] = ""
    df['pdp_image_paths'] = ""

# Make directories
os.makedirs("jeans_images/feature", exist_ok=True)
os.makedirs("jeans_images/pdp", exist_ok=True)

# Download function
def download_image(url, path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            return path
    except Exception as e:
        return None
    return None

# Image download wrapper per row
def process_row(idx, row):
    product_id = str(row['product_id'])

    # Feature image
    feature_url = row['feature_image_s3']
    feature_path = ""
    if isinstance(feature_url, str) and feature_url.startswith("http"):
        feature_path = f"jeans_images/feature/{product_id}.jpg"
        result = download_image(feature_url, feature_path)
        feature_path = result if result else ""

    # PDP images
    pdp_paths = []
    try:
        pdp_urls = ast.literal_eval(row['pdp_images_s3']) if isinstance(row['pdp_images_s3'], str) else []
        for i, url in enumerate(pdp_urls):
            if isinstance(url, str) and url.startswith("http"):
                pdp_path = f"jeans_images/pdp/{product_id}_{i}.jpg"
                result = download_image(url, pdp_path)
                if result:
                    pdp_paths.append(result)
    except:
        pass

    return idx, feature_path, str(pdp_paths)

# Filter unprocessed rows
unprocessed = df[df['feature_image_path'].isna() | (df['feature_image_path'].str.strip() == "")]

# Threaded execution
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_row, idx, row) for idx, row in unprocessed.iterrows()]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
        idx, feature_path, pdp_paths = future.result()
        df.at[idx, 'feature_image_path'] = feature_path
        df.at[idx, 'pdp_image_paths'] = pdp_paths
        df.to_csv(checkpoint_path, index=False)

print("✅ Done! All images downloaded and checkpoint saved.")
