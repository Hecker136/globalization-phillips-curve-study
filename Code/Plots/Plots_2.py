import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the dataset

df = pd.read_csv("C:/phillips_project/data/processed/clean_data.csv")

X = df[[
    "UNRATE",
    "TRADE_GDP",
    "UNRATE_x_TRADE",
    "INFLATION_LAG"
]]

X = sm.add_constant(X)

y = df["INFLATION"]

model = sm.OLS(y, X).fit()

# Check for normality of residuals using a histogram of residuals

plt.figure(figsize=(8,6))

plt.hist(model.resid, bins=20)

plt.xlabel("Residual")
plt.ylabel("Frequency")

plt.title("Histogram of Residuals")

plt.tight_layout()

plt.savefig("C:/phillips_project/Paper/Figures/residuals_histogram.png")

plt.show()