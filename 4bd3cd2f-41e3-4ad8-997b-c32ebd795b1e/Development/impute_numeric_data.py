import pandas as pd
from sklearn.impute import SimpleImputer

# 1. Keep only numeric columns
X = X.select_dtypes(include=["number"])
test_df = test_df.select_dtypes(include=["number"])

# 2. Median imputation (now safe)
imp = SimpleImputer(strategy="median")

X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
test_df = pd.DataFrame(imp.transform(test_df), columns=test_df.columns)

print("✅ Non-numeric columns dropped + missing values handled")
print(X.shape, test_df.shape)
