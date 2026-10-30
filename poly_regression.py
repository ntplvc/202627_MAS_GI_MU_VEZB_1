import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class PolynomialRegression(BaseEstimator, RegressorMixin):
    def __init__(self, degree=1, alpha=0.0, solver="normal"):
        self.degree = degree
        self.alpha = alpha
        self.solver = solver

    def _design_matrix(self, X):
        x = np.asarray(X, dtype=float).ravel()
        return np.vander(x, N=self.degree + 1, increasing=True)

    def fit(self, X, y):
        Phi = self._design_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        N, P = Phi.shape

        if self.solver == "normal":
            A = Phi.T @ Phi + N * self.alpha * np.eye(P)
            b = Phi.T @ y
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, b, rcond=None)[0]
        elif self.solver == "lstsq":
            if self.alpha == 0:
                w = np.linalg.lstsq(Phi, y, rcond=None)[0]
            else:
                Phi_aug = np.vstack([Phi, np.sqrt(N * self.alpha) * np.eye(P)])
                y_aug = np.concatenate([y, np.zeros(P)])
                w = np.linalg.lstsq(Phi_aug, y_aug, rcond=None)[0]
        else:
            raise ValueError("solver mora biti 'normal' ili 'lstsq'")

        self.coef_ = w
        return self

    def predict(self, X):
        return self._design_matrix(X) @ self.coef_

    def score(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        return np.mean((y - self.predict(X)) ** 2)

    def error(self, X, y):
        return 0.5 * self.score(X, y)