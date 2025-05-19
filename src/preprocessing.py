import pandas as pd
import numpy as np
import json
import logging
import re
import string
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract text from a list and remove emojis

def extract_text(lst):
    if isinstance(lst, list) and len(lst) > 0:
        text = ' '.join([str(item).strip() for item in lst])
        # Remove emojis
        text = re.sub(
            r'[\U0001F300-\U0001F5FF'
            r'\U0001F600-\U0001F64F'
            r'\U0001F680-\U0001F6FF'
            r'\U0001F700-\U0001F77F'
            r'\U0001F780-\U0001F7FF'
            r'\U0001F800-\U0001F8FF'
            r'\U0001F900-\U0001F9FF'
            r'\U0001FA00-\U0001FA6F'
            r'\U0001FA70-\U0001FAFF'
            r'\U00002702-\U000027B0'
            r'\U000024C2-\U0001F251]', '', text)
        return text.strip()
    return ''

# Function to preprocess text: lowercase, remove punctuation, remove special characters

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(text.split())

# Read reviews file
file = "/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/raw/CDs_and_Vinyl.jsonl"
data = []
with open(file, 'r') as fp:
    for line in fp:
        data.append(json.loads(line))
df_reviews = pd.DataFrame(data)

# Drop unneeded columns in reviews
df_reviews = df_reviews.drop(columns=['images', 'asin'], axis=1)

# Convert timestamp
df_reviews["timestamp"] = pd.to_datetime(df_reviews["timestamp"], unit="ms", errors="coerce")
if df_reviews["timestamp"].dtype != "datetime64[ns]":
    raise ValueError("Timestamp column is not datetime64[ns] after conversion")

# Drop unneeded columns and sort
# Note: sentiment analysis columns have been removed per request
df_reviews = df_reviews.drop(columns=['helpful_vote','title','text'], axis=1)
df_reviews = df_reviews.sort_values('timestamp')
df_reviews.to_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/reviews_CD.parquet", index=False)
logging.info("Saved reviews_CD.parquet successfully.")

# Read meta file
file_meta = "/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/raw/meta_CDs_and_Vinyl.jsonl"
data_meta = []
with open(file_meta, 'r') as fp:
    for line in fp:
        data_meta.append(json.loads(line))
df_meta = pd.DataFrame(data_meta)


bins   = [-np.inf, 1.5, 2, 3, 4, 4.8, np.inf]
labels = ["Very Bad","Bad","Medium","Above Medium","Good","Excellent"]

# suppose df_reviews has a column `rating` holding the 1–5 score
df_meta["rating_category"] = pd.cut(df_meta["average_rating"], bins=bins, labels=labels)


bins   = [-np.inf, 10, 30, 100, np.inf]
labels = ["Low Popularity","Medium Low Popularity","Medium Popularity","Popular"]

# suppose df_reviews has a column `rating` holding the 1–5 score
df_meta["rating_category_number"] = pd.cut(df_meta["rating_number"], bins=bins, labels=labels)



df_meta["stores"] = (
    df_meta["store"]
      .astype(str)
      .str.replace("Format: Audio CD", "", regex=False)
      .str.strip()
)


# Drop unneeded columns in metadata
df_meta = df_meta.drop(columns=[
    'details','images','features','videos','bought_together','price','subtitle','author','average_rating','store','rating_number','description','categories'
], axis=1)

logging.info("Preprocessing metadata ...")
for col in ['description','categories']:
    if col in df_meta.columns:
        df_meta[col] = df_meta[col].apply(extract_text)

# Then preprocess those columns
for col in ['title','description','categories']:
    if col in df_meta.columns:
        df_meta[col] = df_meta[col].apply(preprocess_text)

# Save metadata
df_meta.to_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/metadata_CD.parquet", index=False)
logging.info("Saved metadata_CD.parquet successfully.")
