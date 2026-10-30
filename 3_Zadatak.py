import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from poly_regression import PolynomialRegression

df = pd.read_csv("data.csv")
X, y = df[["X"]].values, df["Y"].values
X_new = np.array([[600], [900], [1300], [1600]])

results = {}
for M in [1, 2, 3, 5, 9]:
    m = PolynomialRegression(degree=M, solver="lstsq").fit(X, y)
    results[M] = m
    print(f"\n=== M = {M} ===")
    print("w        =", m.coef_)
    print("E(w)     =", m.error(X, y))
    print("MSE(tr.) =", m.score(X, y))
    print("predikcije za", X_new.ravel(), "->", m.predict(X_new).round(3))

for M in [5, 9]:
    m = PolynomialRegression(degree=M, solver="normal").fit(X, y)
    print(f"\n[normal] M={M}: MSE = {m.score(X, y):.3e}")

xs = np.linspace(500, 1650, 400).reshape(-1, 1)
plt.figure(figsize=(9, 5))
plt.scatter(X, y, c="k", label="podaci")
for M, m in results.items():
    plt.plot(xs, m.predict(xs), label=f"M={M}")
plt.ylim(0, 9)
plt.xlabel("Padavine X (mm)")
plt.ylabel("Biomasa Y (t/ha)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()