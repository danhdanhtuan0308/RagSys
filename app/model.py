from pydantic import BaseModel
from datetime import datetime

'''
THIS IS THE MODEL SCHEMA 
'''
#Interaction
class Interaction(BaseModel):
    rating: float
    parent_asin: str
    user_id: str
    timestamp: datetime
    verified_purchase: bool
#Metadata
class Metadata(BaseModel):
    main_category: str
    title: str
    parent_asin: str
    rating_category: str
    rating_category_number: str
    stores: str
#Request for Recsys for 10 user
class RecommnedationRequest(BaseModel):
    user_id: str
    top_k: int = 10

#Recommend item with a score (sigmoid score)
class RecommendationItem(BaseModel):
    item_id: str
    score: float
