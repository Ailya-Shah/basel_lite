import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:America23@localhost:3306/basel_lite")
df = pd.read_sql("SELECT * FROM loans_clean", engine)

# overall default rate
print("Overall default rate:", round(df["default"].mean(), 4), "\n")

# default rate by LendingClub's own risk grade
print("By grade:")
print(df.groupby("grade")["default"].agg(["mean", "count"]).round(3), "\n")

# default rate by loan term
print("By term:")
print(df.groupby("term")["default"].agg(["mean", "count"]).round(3), "\n")

# default rate across FICO bands (5 equal-size groups)
df["fico_band"] = pd.qcut(df["fico_score"], q=5)
print("By FICO band:")
print(df.groupby("fico_band", observed=True)["default"].agg(["mean", "count"]).round(3), "\n")

# average values: defaulters (1) vs payers (0)
print("Defaulted vs paid — averages:")
print(df.groupby("default")[["annual_inc", "dti", "int_rate", "loan_amnt", "fico_score"]].mean().round(2))