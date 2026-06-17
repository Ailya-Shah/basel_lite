import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:America23@localhost:3306/basel_lite")

df = pd.read_csv("loans_200k.csv", low_memory=False)
print("CSV read:", df.shape)        # confirms pandas found and read the file

df.to_sql("loans", engine, if_exists="replace", index=False, chunksize=10000)
print("Loaded into MySQL!")