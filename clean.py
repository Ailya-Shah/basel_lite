import pandas as pd
from sqlalchemy import create_engine

# connect to your database
engine = create_engine("mysql+pymysql://root:America23@localhost:3306/basel_lite")

# 1. read the raw data from MySQL
df = pd.read_sql("SELECT * FROM loans", engine)
print("Start:", df.shape)

# 2. keep only loans with a final outcome (drop 'Current', etc.)
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()

# 3. build the target: 1 = defaulted, 0 = paid back
df["default"] = (df["loan_status"] == "Charged Off").astype(int)

# 4. keep ONLY application-time features (the anti-leakage step)
keep = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "purpose", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "application_type", "mort_acc", "pub_rec_bankruptcies", "addr_state",
    "fico_range_low", "fico_range_high", "default",
]
df = df[keep]

# 5. collapse the FICO range into one score (the midpoint)
df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
df = df.drop(columns=["fico_range_low", "fico_range_high"])

# 6. sanity checks
print("After cleaning:", df.shape)
print("Default rate:", round(df["default"].mean(), 4))
print(df["default"].value_counts())

# 7. store the cleaned table back in MySQL
df.to_sql("loans_clean", engine, if_exists="replace", index=False, chunksize=10000)
print("Done — 'loans_clean' saved in MySQL!")