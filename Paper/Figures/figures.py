import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("C:/Users/brend/Downloads/Model_.py")

X = df[[
    "UNRATE",
    "TRADE_GDP",
    "UNRATE_x_TRADE",
    "INFLATION_LAG"
]]

X = sm.add_constant(X)

y = df["INFLATION"]

model = sm.OLS(y, X).fit()