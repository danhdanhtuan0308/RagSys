import pandas as pd
import numpy as np
import logging
import os
import sys
from scipy.sparse import coo_matrix
import itertools
import matplotlib.pyplot as plt
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.preprocessing        import MinMaxScaler
from sklearn.model_selection      import train_test_split
import optuna
import faiss


from lightfm import cross_validation
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from recommenders.models.lightfm.lightfm_utils import (
    track_model_metrics,
    prepare_test_df,
    prepare_all_predictions,
    compare_metric,
    similar_users,
    similar_items,
)
from recommenders.utils.notebook_utils import store_metadata

try:
    df_reviews = pd.read_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/reviews_CD.parquet")
    df_meta = pd.read_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/metadata_CD.parquet")
    logging.info("Loaded Parquet files successfully.")
except Exception as e:
    logging.error(f"Error loading Parquet files: {e}")
    raise

enriched_df = pd.merge(
    df_reviews,
    df_meta[['parent_asin']],
    on='parent_asin',
    how='left'
)

# Add product features
product_features = df_meta.drop_duplicates('parent_asin')[['parent_asin','main_category','stores','rating_category','rating_category_number']]
df = pd.merge(
    enriched_df,
    product_features,
    on='parent_asin',
    how='left'
)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

def make_train_binary(d):
    d2 = d[d.verified_purchase == True][['user_id','parent_asin']].copy()
    d2['weight'] = 1.0
    return d2.drop_duplicates(['user_id','parent_asin'])

train = make_train_binary(train_df)

train_df = pd.get_dummies(
    train_df,
    columns=["rating_category", "rating_category_number"],
    prefix=["rating_cat", "rating_cat_num"],
    prefix_sep="_",
    dtype=float
)
valid_df = pd.get_dummies(
    valid_df,
    columns=["rating_category", "rating_category_number"],
    prefix=["rating_cat", "rating_cat_num"],
    prefix_sep="_",
    dtype=float)

# ensure both splits have same dummy columns
dummy_cols = sorted([c for c in train_df.columns if c.startswith("rating_cat_")])
for df_ in (train_df, valid_df):
    for c in dummy_cols:
        if c not in df_:
            df_[c] = 0.0


# Build vocabulary lists
combined     = pd.concat([train_df, valid_df], axis=0)
all_users    = combined['user_id'].astype(str).unique().tolist()
all_items    = combined['parent_asin'].astype(str).unique().tolist()
all_stores   = combined['stores'].astype(str).unique().tolist()
all_maincats = combined['main_category'].astype(str).unique().tolist()

user_feat_names    = [f"user_id:{u}"        for u in all_users]
store_feat_names   = [f"stores:{s}"         for s in all_stores]
maincat_feat_names = [f"main_cat:{m}"       for m in all_maincats]

item_feat_names = store_feat_names + maincat_feat_names + dummy_cols
# Fit Dataset
dataset = Dataset()
dataset.fit(
    users=all_users,
    items=all_items,
    user_features=user_feat_names,
    item_features=item_feat_names
)

# Build user_features matrix
user_features = dataset.build_user_features(
    ((u, [f"user_id:{u}"]) for u in all_users)
)
feat_list = []
meta_idx = combined.drop_duplicates('parent_asin').set_index('parent_asin')
for asin, row in meta_idx.iterrows():
    feats = {
        f"stores:{row.stores}":          1.0,
        f"main_cat:{row.main_category}": 1.0
    }
    for col in dummy_cols:
        feats[col] = float(row[col])
    feat_list.append((asin, feats))
item_features = dataset.build_item_features(feat_list)

train_interactions, train_weights = dataset.build_interactions(
    [(r.user_id, r.parent_asin, r.weight) for r in train.itertuples()]
)
train_pairs = set(zip(train.user_id, train.parent_asin))
valid_filt = valid_df[
    ~valid_df.apply(lambda r: (r.user_id, r.parent_asin) in train_pairs, axis=1)
]
test_interactions, _ = dataset.build_interactions(
    [(r.user_id, r.parent_asin, 1) for r in valid_filt.itertuples()]
)

model = LightFM(no_components=10,  
                			learning_schedule='adagrad', 
                			loss='warp', 
                			learning_rate=0.05, 
                			item_alpha=0.0,
                			user_alpha=0.0,
                			max_sampled=30,
                			random_state=123)

model.fit(
    train_interactions,
    sample_weight=train_weights,
    user_features=user_features,    
    item_features=item_features,    
    epochs=30,
    verbose=True
)

recall = recall_at_k(
        model,
        test_interactions,
        train_interactions=train_interactions,
        k=50,
        user_features=user_features,
        item_features=item_features
    ).mean()


print(f'Recall@50: {recall}')

#Saving model
user_embeddings = model.get_user_representations(features=user_features)[1]
item_embeddings = model.get_item_representations(features=item_features)[1]
#save user-item embedding 
np.save('/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/model/retrieval/user_embeddings.npy', user_embeddings)
np.save('/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/model/retrieval/item_embeddings.npy', item_embeddings)

#contruct FAISS index 
d = item_embeddings.shape[1]
index = faiss.IndexHNSWFlat(d, 32)      # HNSW with M=32
index = faiss.IndexIDMap(index)         # allow custom IDs
#Normalized before save into index
faiss.normalize_L2(item_embeddings)     # normalize for cosine
index.add_with_ids(item_embeddings, np.arange(len(all_items)))
# save item embedding into FAISS
faiss.write_index(index, '/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/vectordb/item_index.faiss')