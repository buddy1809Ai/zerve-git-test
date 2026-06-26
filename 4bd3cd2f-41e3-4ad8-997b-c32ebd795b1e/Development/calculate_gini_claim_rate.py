from sklearn.metrics import roc_auc_score

# Generate training predictions (probabilities)
train_predictions = model.predict_proba(X)[:, 1]

# Calculate AUC score
auc_score = roc_auc_score(y, train_predictions)

# Calculate Gini coefficient (normalized Gini = 2*AUC - 1)
gini_coefficient = 2 * auc_score - 1

# Calculate overall claim rate from training data
claim_rate = (y.sum() / len(y)) * 100

# Display results with clear formatting
print("=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"\n📊 Gini Coefficient: {gini_coefficient:.4f}")
print(f"   (Normalized AUC metric: 2 × {auc_score:.4f} - 1)")
print(f"\n📈 Overall Claim Rate: {claim_rate:.2f}%")
print(f"   ({y.sum():,} claims out of {len(y):,} total records)")
print("\n" + "=" * 50)
