from datetime import timedelta
import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
)
from feast.types import Float32, Int32, String, Bool

# Define a project for the feature repo
project = Project(name="ragsys", description="A project for Amazon dataset features")

# Define entities used for joining
user = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="Unique identifier for a user"
)

item = Entity(
    name="parent_asin",
    join_keys=["parent_asin"],
    description="Unique identifier for an item"
)

# Define FileSources reading from your parquet files
amazon_reviews_source = FileSource(
    name="amazon_reviews_source",
    path="/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/reviews.parquet",
    timestamp_field="timestamp"
)

amazon_metadata_source = FileSource(
    name="amazon_metadata_source",
    path="/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/metadata.parquet",
    # If your metadata parquet file includes a timestamp column, you can set it here.
    # Otherwise, set timestamp_field to None.
    timestamp_field= "timestamp" # Disable inference for created timestamp
)

# Define a FeatureView for Amazon reviews (user-item interactions)
amazon_reviews_fv = FeatureView(
    name="amazon_reviews",
    entities=[user, item],
    ttl=timedelta(days=730),  # 2 years history
    schema=[
        Field(name="rating", dtype=Float32, description="User rating for the item"),
        Field(name="title", dtype=String, description="Review title"),
        Field(name="text", dtype=String, description="Review text"),
        Field(name="helpful_vote", dtype=Int32, description="Number of helpful votes"),
        Field(name="verified_purchase", dtype=Bool, description="Verified purchase flag"),
        Field(name="sentiment_category", dtype=Bool, description="Negative/Positive/Neutral"),

    ],
    online=False,
    source=amazon_reviews_source,
    tags={"team": "recommendation_system"},
)

# Define a FeatureView for Amazon item metadata
amazon_metadata_fv = FeatureView(
    name="amazon_item_metadata",
    entities=[item],
    ttl=None,  # Static metadata
    schema=[
        Field(name="title", dtype=String, description="Product title"),
        Field(name="average_rating", dtype=Float32, description="Average product rating"),
        Field(name="rating_number", dtype=Int32, description="Total number of ratings"),
        Field(name="features", dtype=String, description="Product features"),
        Field(name="price", dtype=Float32, description="Product price"),
    ],
    online=False,
    source=amazon_metadata_source,
    tags={"team": "recommendation_system"},
)