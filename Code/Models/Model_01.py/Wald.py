import numpy as np
import pandas as pd
import statsmodels.api as sm

# =========================================================
# 1. DATA PREPARATION
# =========================================================

df = pd.read_csv("C:/phillips_project/data/processed/clean_data.csv")

# Create post-2001 dummy (WTO break)
df["post2001"] = (df["year"] >= 2001).astype(int)

# Drop missing values from lag
df = df.dropna()

# Interaction term (key for slope change)
df["u_post"] = df["UNRATE_c"] * df["post2001"]

# =========================================================
# 2. MODEL 1 (FULL STRUCTURAL PHILLIPS CURVE)
# =========================================================

X = df[[
    "UNRATE_c",
    "TRADE_GDP_c",
    "UNRATE_x_TRADE",
    "INFLATION_LAGGED",
    "post2001",
    "u_post"
]]

X = sm.add_constant(X)
y = df["INFLATION"]

# =========================================================
# 3. ESTIMATE MODEL WITH HAC (NEW KEY STEP)
# =========================================================

model = sm.OLS(y, X).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 1}
)

print(model.summary())

# =========================================================
# 4. WALD TESTS
# =========================================================

# (A) Test slope change only (Phillips Curve change)
wald_slope = model.f_test("u_post = 0")
print("\nWald test (slope change - u_post = 0):")
print(wald_slope)

# (B) Test full structural break (level + slope change)
wald_full = model.f_test("""
post2001 = 0,
u_post = 0
""")

print("\nWald test (full structural break):")
print(wald_full)

# =========================================================
# 5. SIMPLE INTERPRETATION HELPER
# =========================================================

if wald_full.pvalue < 0.05:
    print("\nResult: Evidence of structural break after 2001")
else:
    print("\nResult: No statistically significant structural break detected")