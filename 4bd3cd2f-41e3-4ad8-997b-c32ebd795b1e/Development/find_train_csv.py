import os

for root, dirs, files in os.walk("."):
    if "train.csv" in files:
        print("FOUND train.csv at:", root)
