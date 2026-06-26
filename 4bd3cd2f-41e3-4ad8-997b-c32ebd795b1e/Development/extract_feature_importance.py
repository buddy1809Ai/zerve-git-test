import pandas as pd
import matplotlib.pyplot as plt

# Extract feature importance metrics from the trained XGBoost model
# Get gain-based feature importance (average gain across all splits)
gain_importance = model.get_booster().get_score(importance_type='gain')

# Get weight-based feature importance (number of times feature is used in splits)
weight_importance = model.get_booster().get_score(importance_type='weight')

# Create DataFrames for both metrics
gain_df = pd.DataFrame([
    {'feature': k, 'gain': v} 
    for k, v in gain_importance.items()
]).sort_values('gain', ascending=False)

weight_df = pd.DataFrame([
    {'feature': k, 'weight': v} 
    for k, v in weight_importance.items()
]).sort_values('weight', ascending=False)

# Merge both metrics into a single DataFrame
feature_importance_df = gain_df.merge(weight_df, on='feature', how='outer').fillna(0)
feature_importance_df = feature_importance_df.sort_values('gain', ascending=False).reset_index(drop=True)

# Show top 20 features
print("Top 20 Most Important Features for Claim Prediction:\n")
print(feature_importance_df.head(20).to_string(index=False))
print(f"\n\nTotal features used in model: {len(feature_importance_df)}")
