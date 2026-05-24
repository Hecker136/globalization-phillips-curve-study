import pandas as pd
import numpy as np

start = 1954
end = 2026

# LOAD
cpi = pd.read_csv("C:/phillips_project/Data/Raw/CPIAUCSL.csv")
unrate = pd.read_csv("C:/phillips_project/Data/Raw/UNRATE.csv")
imports = pd.read_csv("C:/phillips_project/Data/Raw/IMPGS.csv")
exports = pd.read_csv("C:/phillips_project/Data/Raw/EXPGS.csv")
gdp = pd.read_csv("C:/phillips_project/Data/Raw/GDP.csv")
mich = pd.read_csv("C:/phillips_project/Data/Raw/MICH.csv")

# RENAME
cpi.columns = ["DATE", "CPI"]
unrate.columns = ["DATE", "UNRATE"]
imports.columns = ["DATE", "IMPORTS"]
exports.columns = ["DATE", "EXPORTS"]
gdp.columns = ["DATE", "GDP"]
mich.columns = ["DATE", "MICH"]

# MERGE
df = cpi.merge(unrate, on="DATE") \
        .merge(imports, on="DATE") \
        .merge(exports, on="DATE") \
        .merge(gdp, on="DATE") \
        .merge(mich, on="DATE")

# SORT FIRST (CRITICAL)
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)
df["year"] = df["DATE"].dt.year

# FILTER FIRST
df = df[df["year"].between(start, end)].copy()

# INFLATION
df["INFLATION"] = np.log(df["CPI"]).diff() * 100
df["INFLATION_LAGGED"] = df["INFLATION"].shift(1)

# TRADE
df["TRADE"] = df["IMPORTS"] + df["EXPORTS"]
df["TRADE_GDP"] = df["TRADE"] / df["GDP"] * 100

# CLEAN
df = df.dropna()

df.to_csv("C:/phillips_project/data/processed/clean_data.csv", index=False)

print(df.head())
print("Clean data saved successfully.")