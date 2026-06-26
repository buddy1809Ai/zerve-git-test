import pandas as pd

submission = pd.DataFrame({
    "id": test_df.index,
    "target": preds
})

submission.to_csv("submission.csv", index=False)

submission.head()
