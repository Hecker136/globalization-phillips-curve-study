import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

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

# -----------------------------
# FILTER DATA
# -----------------------------
df = df[df["year"].between(input_1, input_2)].copy()

# -----------------------------
# CREATE VARIABLES
# -----------------------------
df["UNRATE_c"] = df["UNRATE"] - df["UNRATE"].mean()

df["TRADE_GDP_c"] = (
    df["TRADE_GDP"] - df["TRADE_GDP"].mean()
)

df["UNRATE_x_TRADE"] = (
    df["UNRATE_c"] * df["TRADE_GDP_c"]
)

df["INFLATION_LAGGED"] = (
    df["INFLATION"].shift(1)
)

df = df.dropna()

# -----------------------------
# REGRESSION
# -----------------------------
X = df[
    [
        "UNRATE_c",
        "INFLATION_LAGGED",
        "TRADE_GDP_c",
        "UNRATE_x_TRADE"
    ]
]

X = sm.add_constant(X)

y = df["INFLATION"]

model = sm.OLS(y, X).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 12}
)

print(model.summary())

# -----------------------------
# GRAPH FUNCTION
# -----------------------------
def plot_phillips_curve(start_year, end_year, filename):

    temp = df[
        df["year"].between(start_year, end_year)
    ].copy()

    # Recreate centered variables
    temp["UNRATE_c"] = (
        temp["UNRATE"] - temp["UNRATE"].mean()
    )

    temp["TRADE_GDP_c"] = (
        temp["TRADE_GDP"] - temp["TRADE_GDP"].mean()
    )

    u = temp["UNRATE_c"]
    g = temp["TRADE_GDP_c"]
    pi = temp["INFLATION"]

    # Interaction term
    interaction = u * g

    # Regression matrix
    X_graph = pd.DataFrame({
        "const": 1,
        "u": u,
        "g": g,
        "interaction": interaction
    })

    # OLS
    graph_model = sm.OLS(pi, X_graph).fit()

    # Globalization levels
    g_low = g.quantile(0.25)
    g_mid = g.quantile(0.50)
    g_high = g.quantile(0.75)

    # Unemployment grid
    u_grid = np.linspace(u.min(), u.max(), 100)

    # Prediction function
    def predict_line(g_level):
        return (
            graph_model.params["const"]
            + graph_model.params["u"] * u_grid
            + graph_model.params["g"] * g_level
            + graph_model.params["interaction"]
            * (u_grid * g_level)
        )

    pi_low = predict_line(g_low)
    pi_mid = predict_line(g_mid)
    pi_high = predict_line(g_high)

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure(figsize=(9, 6))

    plt.scatter(u, pi, alpha=0.4)

    plt.plot(
        u_grid,
        pi_low,
        linewidth=2,
        label="Low Globalization"
    )

    plt.plot(
        u_grid,
        pi_mid,
        linewidth=2,
        label="Median Globalization"
    )

    plt.plot(
        u_grid,
        pi_high,
        linewidth=2,
        label="High Globalization"
    )

    plt.xlabel("Unemployment (Centered)")
    plt.ylabel("Inflation")

    plt.title(
        f"Phillips Curve ({start_year}-{end_year})"
    )

    plt.legend()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# -----------------------------
# GRAPH 1
# -----------------------------
plot_phillips_curve(
    1954,
    2001,
    "phillips_curve_1954_2001.png"
)

# -----------------------------
# GRAPH 2
# -----------------------------
plot_phillips_curve(
    2001,
    2026,
    "phillips_curve_2001_2026.png"
)

plt.plot(df["year"], df["TRADE_GDP"])
plt.title("Trade Openness (1954-2026)")
plt.xlabel("Year")
plt.ylabel("Trade Openness (% of GDP)")
plt.gcf().autofmt_xdate()
plt.savefig(
    "trade_openness_1954_2026.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()