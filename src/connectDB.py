import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB

import json
import math
engine = create_engine(f"postgresql+psycopg2://daniellai:admin@localhost:5432/postgres")

# Read reviews file
file_meta = "/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/raw/meta_CDs_and_Vinyl.jsonl"
data_meta = []
with open(file_meta, 'r') as fp:
    for line in fp:
        data_meta.append(json.loads(line))
df_meta = pd.DataFrame(data_meta)

json_cols = [
    "features",
    "description",
    "images",
    "videos",
    "categories",
    "details",
    "author"
]

def ensure_json(v):
    if v is None:
        return None
    # turn any dict or list into a JSON string
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v

for c in json_cols:
    if c in df_meta:
        df_meta[c] = df_meta[c].apply(ensure_json)


df_meta.to_sql("metadata",engine)

# Read reviews file
file = "/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/raw/CDs_and_Vinyl.jsonl"
data = []
with open(file, 'r') as fp:
    for line in fp:
        data.append(json.loads(line))
df_reviews = pd.DataFrame(data) 

df_reviews['images'] = df_reviews['images'].apply(ensure_json)

df_reviews.to_sql('interaction',engine)