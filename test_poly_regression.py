import pandas as pd
from poly_regression import PolynomialRegression

df = pd.read_csv("data.csv")
X = df[["X"]].values
y = df["Y"].values

model = PolynomialRegression(degree=2)
model.fit(X, y)

print("w =", model.coef_)
print("predikcije:", model.predict(X))
print("MSE =", model.score(X, y))