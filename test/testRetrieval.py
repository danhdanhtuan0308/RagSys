
import pandas as pd  
import numpy as np
import faiss


#RECOMMEND TOP-10 ITEM FOR USER 0 
user_embeddings = np.load("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/model/retrieval/user_embeddings.npy").astype("float32")
#item_embeddings = np.load("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/model/retrieval/item_embedding.npy").astype("float32")

df_meta = pd.read_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/metadata_CD.parquet")
df_meta = df_meta.drop_duplicates("parent_asin").reset_index(drop=True)
df_reviews = pd.read_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/reviews_CD.parquet")
index = faiss.read_index("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/model/vectordb/item_index.faiss")

faiss.normalize_L2(user_embeddings)

#Get the top-10 most similar items for each user
#    D: (num_users, 10) similarity scores
#    I: (num_users, 10) integer item positions
D, I = index.search(user_embeddings, 10)

target = "AEEW7R4GEEETWOZCCFQAOW2LYPCA"
user_ids_list = df_reviews["user_id"].drop_duplicates().tolist()

try:
    user_pos = user_ids_list.index(target)
except ValueError:
    raise ValueError(f"{target} not found in reconstructed user list")

scores = D[user_pos]       # shape (10,)
idxs   = I[user_pos]       # shape (10,)

#Sortting
order = np.argsort(-scores)
idxs   = idxs[order]
scores = scores[order]

# select metadata rows by positional index
top10 = df_meta.iloc[idxs].copy()

#ranking top 10 
top10.insert(0, "rank",  np.arange(1, 11))
top10["score"] = scores

cols = ["rank", "parent_asin", "title", "main_category",
        "rating_category", "rating_category_number", "stores", "score"]
top10 = top10[cols]

print(top10)