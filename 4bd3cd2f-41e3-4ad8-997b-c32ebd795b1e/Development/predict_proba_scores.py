preds = model.predict_proba(test_df)[:, 1]
print(preds[:10])
