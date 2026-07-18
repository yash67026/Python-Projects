import os
import pandas as pd

path = r"C:\Users\yyash\Desktop\FraudDetection\Fraud Detection System\data\fraudTrain.csv"

print("Exists:", os.path.exists(path))
print("Size:", os.path.getsize(path))

with open(path, "r") as f:
    print("\nFirst 5 lines of file:")
    for i in range(5):
        print(f.readline())

df = pd.read_csv(path)
print("\nLoaded successfully")
print(df.head())