import pandas as pd

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test_data.csv")

print("Train:", train_df.shape)
print("Test :", test_df.shape)
