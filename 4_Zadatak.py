import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from poly_regression import PolynomialRegression

df = pd.read_csv("data.csv")
X, y = df[["X"]].values, df["Y"].values
X_new = np.array([[600], [900], [1300], [1600]])

alphas = [0, 1e-12, 1e-9, 1e-6, 1e-3]

for M in [1, 2, 3, 5, 9]:
    print(f"\n=== M = {M} ===")
    for a in alphas:
        m = PolynomialRegression(degree=M, alpha=a, solver="lstsq").fit(X, y)
        print(f"alpha={a:g}  MSE_tr={m.score(X, y):.4f}  "
              f"||w||={np.linalg.norm(m.coef_):.3e}  "
              f"pred={m.predict(X_new).round(2)}")

xs = np.linspace(500, 1650, 400).reshape(-1, 1)
plt.figure(figsize=(9, 5))
plt.scatter(X, y, c="k", label="podaci")
for a in [0, 1e-9, 1e-6, 1e-3]:
    m = PolynomialRegression(degree=9, alpha=a, solver="lstsq").fit(X, y)
    plt.plot(xs, m.predict(xs), label=f"M=9, alpha={a:g}")
plt.ylim(0, 9)
plt.legend()
plt.grid(alpha=0.3)
plt.xlabel("X (mm)")
plt.ylabel("Y (t/ha)")
plt.show()