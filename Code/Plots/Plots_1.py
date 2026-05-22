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

# Check for heteroscedasticity using residuals vs fitted values plot

plt.figure(figsize=(8,6))

plt.scatter(model.fittedvalues, model.resid)

plt.axhline(y=0)

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")

plt.title("Residuals vs Fitted")

plt.tight_layout()

plt.savefig("C:/phillips_project/Paper/Figures/residuals_vs_fitted.png")

plt.show()