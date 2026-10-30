import pandas as pd

df = pd.read_csv("creditcard.csv")
df.iloc[:142403].to_csv("data/creditcard_prefix.csv", index=False)
df.iloc[142403:142403+142404].to_csv("data/creditcard_suffix.csv", index=False)
print("Hoàn tất phân tách dữ liệu.")