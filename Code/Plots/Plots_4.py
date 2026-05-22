import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Load the dataset

df = pd.read_csv("C:/phillips_project/data/processed/clean_data.csv")

x = df["UNRATE"]
y = df["INFLATION"]

# quadratic fit
coeffs = np.polyfit(x, y, 2)
quad = np.poly1d(coeffs)

# smooth curve
x_sorted = np.sort(x)

plt.scatter(x, y)
plt.plot(x_sorted, quad(x_sorted), color="red")

plt.xlabel("Unemployment")
plt.ylabel("Inflation")
plt.title("Quadratic Phillips Curve Fit, A = {:.2f}".format(coeffs[0]))
plt.grid(True)
plt.savefig("C:/phillips_project/Paper/Figures/unemployment_vs_inflation_quadratic.png")
plt.show()