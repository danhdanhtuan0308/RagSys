import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker
import logging
import joblib

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
META_PATH      = Path("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/metadata_CD.parquet")
REVIEWS_PATH   = Path("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/reviews_CD.parquet")
USER_EMB_PATH  = Path("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/model/retrieval/user_embeddings.npy")
ITEM_EMB_PATH  = Path("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/model/retrieval/item_embeddings.npy")
OUTPUT_MODEL = Path("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/model/ranking/xgbranker.pkl")

df_reviews   = pd.read_parquet(REVIEWS_PATH)
df_meta      = pd.read_parquet(META_PATH)
user_embs    = np.load(USER_EMB_PATH).astype(np.float32)
item_embs    = np.load(ITEM_EMB_PATH).astype(np.float32)

df = df_reviews.merge(
    df_meta[["parent_asin"]], on="parent_asin", how="left"
)

# Binarize purchase
df["y"] = df["verified_purchase"].astype(int)
users           = df["user_id"].unique()
train_users, valid_users = train_test_split(users, test_size=0.3, random_state=42)
logging.info("Spliting successfully")


train_df = df[df['user_id'].isin(train_users)].copy()
valid_df = df[df['user_id'].isin(valid_users)].copy()

#Build maps on the conver user-item id -> index
all_users = df['user_id'].unique()
all_items = df['parent_asin'].unique()
user2idx  = {u: i for i, u in enumerate(all_users)}
item2idx  = {p: i for i, p in enumerate(all_items)}

# Map both splits
for d in (train_df, valid_df):
    d['u_idx'] = d['user_id'].map(user2idx).astype(int)
    d['i_idx'] = d['parent_asin'].map(item2idx).astype(int)

train_df = train_df.sort_values('u_idx').reset_index(drop=True)
valid_df = valid_df.sort_values('u_idx').reset_index(drop=True)

#  Now build features
def build_X(df_subset):
    return np.vstack([
        np.concatenate([
            user_embs[row.u_idx],
            item_embs[row.i_idx]
        ])
        for row in df_subset.itertuples()
    ]).astype(np.float32)

X_train, y_train = build_X(train_df), train_df['y'].values
X_valid, y_valid = build_X(valid_df), valid_df['y'].values
# Group definitions
group_train = train_df.groupby("u_idx").size().sort_index().values
group_valid = valid_df.groupby("u_idx").size().sort_index().values

#ranker model
ranker = XGBRanker(
    n_estimators=250,
    objective="rank:ndcg",
    tree_method="hist",
    learning_rate=0.01,
    max_depth=6,
    eval_metric="ndcg@10",
    random_state=42,
)

logging.info("Training ranker...")
#training
ranker.fit(
    X_train, y_train,
    group=group_train,
    eval_set=[(X_valid, y_valid)],
    eval_group=[group_valid],
    verbose=True,
)
#save file
joblib.dump(ranker, OUTPUT_MODEL)
logging.info(f"Saved ranker to {OUTPUT_MODEL}")

logging.info("Evaluating nDCG@10...")
ndcgs = []
for uidx in np.unique(valid_df["u_idx"]):
    mask = valid_df["u_idx"].to_numpy() == uidx
    if mask.sum() < 2 or y_valid[mask].sum() == 0:
        continue
    y_true = y_valid[mask].reshape(1, -1)
    y_pred = ranker.predict(X_valid[mask]).reshape(1, -1)
    ndcgs.append(ndcg_score(y_true, y_pred, k=10))

logging.info(f"Mean nDCG@10: {np.mean(ndcgs):.6f}")
