import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

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

df["INFLATION_LAGED"] = df["INFLATION"].shift(1)

df = df.dropna()

X = df[["UNRATE_c", "INFLATION_LAGED","TRADE_GDP_c", "UNRATE_x_TRADE"]]
X = sm.add_constant(X)

y = df["INFLATION"]

model = sm.OLS(y, X).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 12}
)
print(model.summary())

u = df["UNRATE_c"]
g = df["TRADE_GDP_c"]
pi = df["INFLATION"]

# Interaction term
interaction = u * g

# Regression matrix
X = pd.DataFrame({
    "const": 1,
    "u": u,
    "g": g,
    "interaction": interaction
})

# OLS
model = sm.OLS(pi, X).fit()

# Globalization levels
g_low = g.quantile(0.25)
g_mid = g.quantile(0.50)
g_high = g.quantile(0.75)

# Unemployment grid
u_grid = np.linspace(u.min(), u.max(), 100)

# Predicted lines
def predict_line(g_level):
    return (
        model.params["const"]
        + model.params["u"] * u_grid
        + model.params["g"] * g_level
        + model.params["interaction"] * (u_grid * g_level)
    )

pi_low = predict_line(g_low)
pi_mid = predict_line(g_mid)
pi_high = predict_line(g_high)

# Plot
plt.figure(figsize=(9,6))

plt.scatter(u, pi, alpha=0.4)

plt.plot(u_grid, pi_low, linewidth=2,
         label="Low Globalization")

plt.plot(u_grid, pi_mid, linewidth=2,
         label="Median Globalization")

plt.plot(u_grid, pi_high, linewidth=2,
         label="High Globalization")

plt.xlabel("Unemployment (Centered)")
plt.ylabel("Inflation")
plt.title("Phillips Curve at Different Globalization Levels")

plt.legend()
plt.savefig("phillips_curve_interaction.png",
            dpi=300,
            bbox_inches='tight')
plt.show()