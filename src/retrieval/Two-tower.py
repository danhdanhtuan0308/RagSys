import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    StringLookup,
    TextVectorization,
)
from tensorflow.keras import Model, Sequential
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import faiss


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

'''
BATCH_SIZE : Size of batch 
EMB_DIM : Embedding Dimension ( nodes )
NUM_NEG : Number of negative sampling 
MARGIN : margin for BPR loss 
BCE_WEIGHT : weight for BCE 
L2_REG : regulization 
'''
BATCH_SIZE    = 1024
EMB_DIM       = 128
NUM_NEG       = 20
MARGIN        = 0.4
BCE_WEIGHT    = 0.6
L2_REG        = 1e-4

df_reviews = pd.read_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/reviews_CD.parquet")
df_meta = pd.read_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/metadata_CD.parquet")

# pull in your actual 'stores' column
enriched_df = pd.merge(df_reviews, df_meta[['parent_asin']], on='parent_asin', how='left')
product_feats = (
    df_meta
    .drop_duplicates('parent_asin')[['parent_asin','stores','main_category','title',"rating_category_number"]]
)

#merge
enriched_df = pd.merge(enriched_df, product_feats, on='parent_asin', how='left')

#Split 
train_df, val_df = train_test_split(enriched_df, train_size=0.8, random_state=42)
for df in (train_df, val_df):
    df["stores"]                  = df["stores"].fillna("UNK").astype(str)
    df["verified_purchase"]       = df["verified_purchase"].map({True:1, False:0}).astype(np.float32)
    df["main_category"]           = df["main_category"].fillna("UNK").astype(str)
    df["title"]                   = df["title"].astype(str)
    df["rating_category_number"]  = df["rating_category_number"].astype(str)


# Data PIPELINES 
def df_to_dataset(df):
    features = {
        "user_id":                  df["user_id"].values.astype(str),
        "parent_asin":              df["parent_asin"].values.astype(str),
        "stores":                   df["stores"].values,                             # now matches your column
        "main_category":            df["main_category"].values,
        "title":                    df["title"].values.reshape(-1,1).astype(str),
        "rating_category_number":   df["rating_category_number"].values.reshape(-1,1).astype(str),
        "verified_purchase":        df["verified_purchase"].values,
    }
    return (
        tf.data.Dataset.from_tensor_slices(features)
                         .shuffle(len(df))
                         .batch(BATCH_SIZE)
                         .cache()
    )

def df_to_cand_dataset(df):
    feats = {
        "parent_asin":             df["parent_asin"].values.astype(str),
        "stores":                  df["stores"].values,
        "main_category":           df["main_category"].values,
        "title":                   df["title"].values.reshape(-1,1).astype(str),
        "rating_category_number":  df["rating_category_number"].values.reshape(-1,1).astype(str),
    }
    return tf.data.Dataset.from_tensor_slices(feats).batch(BATCH_SIZE)

train_ds     = df_to_dataset(train_df)
val_ds       = df_to_dataset(val_df)
unique_items = train_df.drop_duplicates("parent_asin").reset_index(drop=True)
item_ds      = df_to_cand_dataset(unique_items)

# User Tower
class QueryTower(Model):
    def __init__(self, user_ids):
        super().__init__()
        self.user_emb = Sequential([
            StringLookup(vocabulary=user_ids, mask_token=None),
            Embedding(len(user_ids)+1, EMB_DIM, embeddings_regularizer=tf.keras.regularizers.l2(L2_REG)),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(EMB_DIM, activation="relu"),
            Dropout(0.3),
            Dense(EMB_DIM, activation="relu"),
            Dropout(0.3),
        ])
    def call(self, features):
        return self.user_emb(features["user_id"])

#Item Tower
class ItemTower(Model):
    def __init__(self, item_ids, stores, main_cats, rating_cats):
        super().__init__()
        self.item_emb   = Sequential([
            StringLookup(vocabulary=item_ids, mask_token=None),
            Embedding(len(item_ids)+1, EMB_DIM, embeddings_regularizer=tf.keras.regularizers.l2(L2_REG))
        ])
        self.store_emb  = Sequential([
            StringLookup(vocabulary=stores, mask_token=None),
            Embedding(len(stores)+1, EMB_DIM, embeddings_regularizer=tf.keras.regularizers.l2(L2_REG))
        ])
        self.cat_emb    = Sequential([
            StringLookup(vocabulary=main_cats, mask_token=None),
            Embedding(len(main_cats)+1, EMB_DIM, embeddings_regularizer=tf.keras.regularizers.l2(L2_REG))
        ])
        self.title_emb  = Sequential([
            TextVectorization(max_tokens=50, output_sequence_length=20),
            Embedding(50, EMB_DIM, embeddings_regularizer=tf.keras.regularizers.l2(L2_REG)),
            GlobalAveragePooling1D(),
        ])
        self.rating_lookup = StringLookup(vocabulary=rating_cats, mask_token=None)
        self.fnn = Sequential([
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(EMB_DIM, activation="relu"),
            Dropout(0.3),
            Dense(EMB_DIM, activation="relu"),
            Dropout(0.3),
        ])

    def call(self, features):
        i = self.item_emb(features["parent_asin"])
        s = self.store_emb(features["stores"])          # use "stores"
        c = self.cat_emb(features["main_category"])
        t = self.title_emb(features["title"])
        r_idx = self.rating_lookup(tf.squeeze(features["rating_category_number"], axis=1))
        r_oh  = tf.one_hot(r_idx, depth=len(rating_cats)+1)
        x = tf.concat([i, s, c, t, r_oh], axis=1)
        return self.fnn(x)

# prepare and adapt
user_ids    = train_df["user_id"].unique().astype(str).tolist()
item_ids    = train_df["parent_asin"].unique().astype(str).tolist()
stores      = train_df["stores"].unique().tolist()
main_cats   = train_df["main_category"].unique().tolist()
rating_cats = train_df["rating_category_number"].unique().tolist()

query_model = QueryTower(user_ids)
item_model  = ItemTower(item_ids, stores, main_cats, rating_cats)

item_model.title_emb.layers[0].adapt(train_df["title"].values.reshape(-1,1))

#TWO-TOWER WITH BPR + BCE LOSS
class TwoTowerModel(tfrs.Model):
    def __init__(self, query_tower, item_tower):
        super().__init__()
        self.query_model = query_tower
        self.item_model  = item_tower
        self.bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def compute_loss(self, features, training=False):
        user_emb = self.query_model(features)
        pos_emb  = self.item_model(features)

        negs = []
        for _ in range(NUM_NEG):
            neg_feats = {k: tf.random.shuffle(v) for k,v in features.items()}
            negs.append(self.item_model(neg_feats))
        neg_emb = tf.concat(negs, axis=0)

        user_rep = tf.repeat(user_emb, NUM_NEG, axis=0)
        pos_rep  = tf.repeat(pos_emb,  NUM_NEG, axis=0)
        diffs     = tf.reduce_sum(user_rep * (pos_rep - neg_emb), axis=1)
        weights   = tf.where(features["verified_purchase"]==1.0, 1.0, 0.3)
        weights_r = tf.repeat(weights, NUM_NEG, axis=0)
        bpr_loss  = tf.reduce_mean(weights_r * tf.nn.relu(MARGIN - diffs))

        pos_logits = tf.reduce_sum(user_emb * pos_emb, axis=1)
        bce_loss   = self.bce_loss_fn(tf.ones_like(pos_logits), pos_logits)

        reg_loss = tf.add_n(self.query_model.losses + self.item_model.losses)

        return bpr_loss + BCE_WEIGHT*bce_loss + L2_REG*reg_loss

model = TwoTowerModel(query_model, item_model)
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.004))

early_stopping = EarlyStopping(monitor='val_total_loss', patience=3, restore_best_weights=True)
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stopping], verbose=1)
# Gather all item embeddings (N×D) and corresponding string IDs
item_embeddings, item_ids = [], []
for batch in item_ds:  # item_ds yields dicts with "parent_asin" and other features
    embs = item_model(batch).numpy()        # (batch_size, EMB_DIM)
    item_embeddings.append(embs)
    item_ids += batch["parent_asin"].numpy().tolist()

item_embeddings = np.vstack(item_embeddings).astype("float32")  # (N, D)
item_ids = [i.decode() for i in item_ids]    

faiss.normalize_L2(item_embeddings)

# Build a FAISS HNSWFlat index for inner-product (cosine) search
d = item_embeddings.shape[1]
index = faiss.IndexHNSWFlat(d, 32)        # graph degree M=32
index = faiss.IndexIDMap(index)           # allow our own integer IDs
# Map string IDs → ints and add to index
str2int = {s: idx for idx, s in enumerate(item_ids)}
int_ids = np.array([str2int[s] for s in item_ids], dtype="int64")
index.add_with_ids(item_embeddings, int_ids)  

# Save the index to disk
def recommend_faiss(user_ids, k=50):
    # ➤ Make a string tensor of shape (batch,)
    user_tensor = tf.constant(user_ids, dtype=tf.string)

    user_embs = query_model({"user_id": user_tensor}).numpy()
    faiss.normalize_L2(user_embs)

    D, I = index.search(user_embs, k)
    rev = {v: k for k, v in str2int.items()}
    return [[rev[int(i)] for i in row] for row in I] 

from collections import defaultdict

def build_ground_truth(df):
    gt = defaultdict(set)
    for u, item in zip(df.user_id.astype(str), df.parent_asin.astype(str)):
        gt[u].add(item)
    return gt

ground_truth = build_ground_truth(val_df) 

def compute_recall_at_k(ground_truth, recommendations, k):
    recalls = []
    for user, true_items in ground_truth.items():
        if not true_items: continue
        recs = recommendations.get(user, [])[:k]
        hits = len(true_items.intersection(recs))
        recalls.append(hits / len(true_items))
    return float(np.mean(recalls))

from tqdm import tqdm

# assume ground_truth is your dict of users→true‐items
user_list = list(ground_truth.keys())

# get all recommendations in one batch (more efficient than one‐by‐one)
# returns a list of lists in the same order as user_list
all_recs = recommend_faiss(user_list, k=50)

# now build your recs dict with a progress bar
recs = {}
for u, rec in tqdm(zip(user_list, all_recs), total=len(user_list), desc="Building recs"):
    recs[u] = rec

r30 = compute_recall_at_k(ground_truth, recs, k=50)
print(f"Recall@50 = {r30}")

