import pickle as pkl
from functools import lru_cache
import pandas as pd
import numpy as np
import faiss
from fastapi import APIRouter, Depends, HTTPException
from pathlib import Path

from model import RecommnedationRequest, RecommendationItem
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = PROJECT_ROOT / "dataset" / "preprocess"
MODEL_DIR = PROJECT_ROOT / "model"


class DataManager:
    def __init__(
        self,
        df_meta: Path          = DATA_DIR / "metadata_CD.parquet",
        df_review: Path        = DATA_DIR / "reviews_CD.parquet",
        user_emb_path: Path    = MODEL_DIR / "retrieval" / "user_embeddings.npy",
        item_emb_path: Path    = MODEL_DIR / "retrieval" / "item_embeddings.npy",
        faiss_index_path: Path = MODEL_DIR / "vectordb"   / "item_index.faiss",
        ranker_path: Path      = MODEL_DIR / "ranking"    / "xgbranker.pkl",

    ):
        #  Load metadata & reviews
        self.df_meta   = pd.read_parquet(df_meta)
        self.df_review = pd.read_parquet(df_review)
        #Load embeddings
        self.user_embeddings = np.load(user_emb_path).astype(np.float32)
        self.item_embeddings = np.load(item_emb_path).astype(np.float32)

        # Load FAISS index for retrieval
        index_path = str(faiss_index_path)
        self.index = faiss.read_index(index_path)

        #Load pre-trained ranker
        with open(ranker_path, "rb") as f:
            self.ranker = pkl.load(f)

        # Build lookup maps
        self.user_to_idx = {
            u: i for i, u in enumerate(self.df_review["user_id"].unique())
        }
        # item_ids in same order as item_embeddings
        self.item_ids = list(self.df_meta["parent_asin"].unique())

    def retrieval(self, user_id: str, k: int):
        if user_id not in self.user_to_idx:
            raise ValueError(f"User ID {user_id} not found.")
        ui  = self.user_to_idx[user_id]
        uq  = self.user_embeddings[ui:ui+1]
        faiss.normalize_L2(uq)
        _, I = self.index.search(uq, k)
        return I[0].tolist()

    retrieve = retrieval  # alias for form_post usage

    def rank(self, user_id: str, candidates):
        ui       = self.user_to_idx[user_id]
        user_emb = self.user_embeddings[ui].flatten()
        # build features as [user_emb ∥ item_emb]
        feats = [
            np.concatenate([user_emb, self.item_embeddings[item_idx]])
            for item_idx in candidates
        ]
        X = np.vstack(feats)
        scores = self.ranker.predict(X)
        probas = 1.0 / (1.0 + np.exp(-scores))

        # sort by descending probability
        order = np.argsort(probas)[::-1]
        return [
            (candidates[i], float(probas[i]))
            for i in order
        ]

#Store DataManager to cache
@lru_cache()
def get_data_manager() -> DataManager:
    return DataManager()

api_router = APIRouter()

@api_router.post(
    "/recommend",
    response_model=list[RecommendationItem],
    summary="Retrieve + re-rank for a given user_id",
)
#Create req and flow
def recommend_api(
    req: RecommnedationRequest,
    dm: DataManager = Depends(get_data_manager),
):
    try:
        cands  = dm.retrieve(req.user_id, req.top_k)
        ranked = dm.rank(req.user_id, cands)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return [
        RecommendationItem(item_id=dm.item_ids[idx], score=score)
        for idx, score in ranked
    ]
