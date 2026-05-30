import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("C:/phillips_project/data/processed/clean_data.csv")

# -----------------------------
# USER INPUT
# -----------------------------
def inputs():
    print("OLS regression model (1954-2026)")

    while True:
        try:
            input_1 = int(input("Enter start year: "))
            if 1954 <= input_1 <= 2025:
                break
            else:
                print("Invalid start year.")
        except ValueError:
            print("Enter a valid number.")

    while True:
        try:
            input_2 = int(input("Enter end year: "))
            if input_1 <= input_2 <= 2026:
                break
            else:
                print("Invalid end year.")
        except ValueError:
            print("Enter a valid number.")

    return input_1, input_2

input_1, input_2 = inputs()

df["post_wto"] = (df["year"] >= 2001).astype(int)

# -----------------------------
# FILTER DATA
# -----------------------------
df = df[df["year"].between(input_1, input_2)].copy()

# -----------------------------
# CREATE VARIABLES
# -----------------------------
df["UNRATE_c"] = df["UNRATE"] - df["UNRATE"].mean()

df["UNRATE_post"] = df["UNRATE_c"] * df["post_wto"]

df["LAGGED_INFLATION"] = df["INFLATION"].shift(1)

df = df.dropna()

X = df[["UNRATE_c", "post_wto", "UNRATE_post","LAGGED_INFLATION"]]
X = sm.add_constant(X)

y = df["INFLATION"]

model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
print(model.summary())

adf_result = adfuller(model.resid)
print("\nADF Test Results:")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value}")

residuals = model.resid
fitted = model.fittedvalues

sm.qqplot(residuals, line="45")
plt.title("QQ Plot of Residuals")
plt.show()

plt.figure()
plt.hist(residuals, bins=30, edgecolor="black")
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.scatter(fitted, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()