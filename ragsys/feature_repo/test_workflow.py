from feast import FeatureStore
import pandas as pd

fs = FeatureStore(repo_path=".")
df_reviews = pd.read_parquet("/Users/daniellai/MSDS_2026/MSDS_2024_2026/Spring_2025/Projects/dataset/reviews.parquet")
entity_df = pd.DataFrame({
    "user_id": df_reviews["user_id"].tail(100),
    "parent_asin": df_reviews["parent_asin"].tail(100),
    "timestamp": df_reviews["timestamp"].tail(100)
})
features = [
    "amazon_reviews:rating",
    "amazon_reviews:title",
    "amazon_reviews:text",
    "amazon_reviews:helpful_vote",
    "amazon_reviews:verified_purchase",
]
training_df = fs.get_historical_features(entity_df=entity_df, features=features).to_df()
print(training_df.head())
print("\nTotal rows:", df_reviews.shape[0])
