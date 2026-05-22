import pandas as pd

df = pd.read_csv("C:/phillips_project/data/processed/clean_data.csv")

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

df["INFLATION"] = df["INFLATION"] - df["INFLATION"].mean()

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