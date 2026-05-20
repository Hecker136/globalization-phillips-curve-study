import pandas as pd 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

unemployment = pd.read_csv("C:/Users/brend/Downloads/UNRATE.csv")
inflation = pd.read_csv("C:/Users/brend/Downloads/CPIAUCSL.csv")
expectations = pd.read_csv("C:/Users/brend/Downloads/MICH.csv")

unemployment.columns = ["DATE", "UNRATE"]
inflation.columns = ["DATE", "CPI"]
expectations.columns = ["DATE", "MICH"]

data = unemployment.merge(inflation, on="DATE").merge(expectations, on="DATE")

data["INFLATION"] = data["CPI"].pct_change(periods=12, fill_method=None) * 100
data = data.dropna()

data["DATE"] = pd.to_datetime(data["DATE"])
data["year"] = data["DATE"].dt.year

def get_period(start, end):
    return data[(data["year"] >= start) & (data["year"] < end)][["UNRATE", "INFLATION", "MICH"]].dropna()

period1 = get_period(1960, 1980)
period2 = get_period(1980, 2000)
period3 = get_period(2000, 2026)

Export = pd.read_csv("C:/Users/brend/Downloads/EXPGS.csv")
Import = pd.read_csv("C:/Users/brend/Downloads/IMPGS.csv")
GDP = pd.read_csv("C:/Users/brend/Downloads/GDP.csv")

Export.columns = ["DATE", "EXPGS"]
Import.columns = ["DATE", "IMPGS"]
GDP.columns = ["DATE", "GDP"]

trade = Export.merge(Import, on="DATE").merge(GDP, on="DATE")

trade["TRADE"] = trade["EXPGS"] + trade["IMPGS"]
trade["TRADE_GDP"] = trade["TRADE"] / trade["GDP"] * 100

trade["DATE"] = pd.to_datetime(trade["DATE"])

df = trade.merge(data, on="DATE", how="inner")

def inputs():
  print("OLS regression model (1954-2026)")
  while True:
    try:
      input_1 = int(input("Enter start year: (From 1954 to 2025)"))
      if input_1 < 1954 or input_1 > 2025:
        print("Invalid date. Start year must be between 1954 and 2025. Please retry.")
      else:
        break
    except ValueError:
      print("Invalid input. Please enter a number for the start year. Please retry.")

  while True:
    try:
      input_2 = int(input("Enter end year: (From 1955 to 2026)"))
      if input_2 > 2026 or input_2 < input_1:
        print("Invalid date. End year must be between 1955 and 2026, and not before start year. Please retry.")
      else:
        break
    except ValueError:
      print("Invalid input. Please enter a number for the end year. Please retry.")

  return input_1, input_2

input_1, input_2 = inputs()

df = df[df["year"].between(input_1, input_2)]

df["UNRATE_c"] = df["UNRATE"] - df["UNRATE"].mean()
df["TRADE_GDP_c"] = df["TRADE_GDP"] - df["TRADE_GDP"].mean()

df["UNRATE_x_TRADE"] = df["UNRATE_c"] * df["TRADE_GDP_c"]

df["INFLATION"] = df["INFLATION"] - df["MICH"]

df["INFLATION_LAGGED"] = df["INFLATION"].shift(1)

df = df.dropna()

X = df[["UNRATE_c","TRADE_GDP_c", "UNRATE_x_TRADE", "INFLATION_LAGGED"]]
X = sm.add_constant(X)

y = df["INFLATION"]

model = sm.OLS(y, X).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 12}
)
print(model.summary())