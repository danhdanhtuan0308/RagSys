from fastapi import APIRouter, Depends, HTTPException
from typing import List
import pandas as pd
from model import Metadata


router = APIRouter(prefix="/metadata", tags=["metadata"]) 

# Function to load the metadata DataFrame
def get_df_meta() -> pd.DataFrame:
    """
    Function to load the metadata DataFrame.
    """
    df_meta = pd.read_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/preprocess/metadata_CD.parquet")
    return df_meta

@router.get("/", response_model=List[Metadata])
def read_metadata(limit: int = 5, df: pd.DataFrame = Depends(get_df_meta)):
    """
    Return up to `limit` rows of metadata.
    """
    if limit < 1 or limit > 100:
        #litmit to 100
        raise HTTPException(400, "limit must be between 1 and 100")
    records = df.head(limit).to_dict(orient="records")
    return records