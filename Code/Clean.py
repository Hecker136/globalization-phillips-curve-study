import pandas as pd

# =========================
# LOAD DATA
# =========================

cpi = pd.read_csv("C:/phillips_project/Data/Raw/CPIAUCSL.csv")
unrate = pd.read_csv("C:/phillips_project/Data/Raw/UNRATE.csv")
imports = pd.read_csv("C:/phillips_project/Data/Raw/IMPGS.csv")
exports = pd.read_csv("C:/phillips_project/Data/Raw/EXPGS.csv")
gdp = pd.read_csv("C:/phillips_project/Data/Raw/GDP.csv")

# =========================
# CLEAN COLUMN NAMES
# =========================

cpi.columns = ["DATE", "CPI"]
unrate.columns = ["DATE", "UNRATE"]
imports.columns = ["DATE", "IMPORTS"]
exports.columns = ["DATE", "EXPORTS"]
gdp.columns = ["DATE", "GDP"]

# =========================
# MERGE DATASETS
# =========================

df = cpi.merge(unrate, on="DATE")
df = df.merge(imports, on="DATE")
df = df.merge(exports, on="DATE")
df = df.merge(gdp, on="DATE")

# =========================
# Normalize year column and convert to datetime
# =========================

df["DATE"] = pd.to_datetime(df["DATE"])
df["year"] = df["DATE"].dt.year

def get_period(start, end):
    return df[(df["year"] >= start) & (df["year"] < end)][["UNRATE", "INFLATION", "MICH"]].dropna()

# =========================
# CREATE VARIABLES
# =========================

# Inflation
df["INFLATION"] = df["CPI"].pct_change() * 100

# Lagged inflation
df["INFLATION_LAG"] = df["INFLATION"].shift(1)

# Trade openness
df["TRADE"] = df["IMPORTS"] + df["EXPORTS"]
df["TRADE_GDP"] = df["TRADE"] / df["GDP"] * 100

# Interaction term
df["UNRATE_x_TRADE"] = df["UNRATE"] * df["TRADE_GDP"]

# Inflation gap (2% target)
df["INFLATION_GAP"] = df["INFLATION"] - 2

# =========================
# REMOVE MISSING VALUES
# =========================

df = df.dropna()

# =========================
# SAVE CLEAN DATA
# =========================

df.to_csv("C:/phillips_project/data/processed/clean_data.csv", index=False)

print(df.head())
print("Clean data saved successfully.")