import numpy as np
from sklearn.linear_model import LinearRegression as LinR, Lasso as LasR

def sample_data(X_cov, M, noise_std, n_data, rng):
    X = rng.multivariate_normal(np.zeros_like(M), X_cov, n_data)
    X /= X.std(axis=0, keepdims=True)
    Y = X.dot(M) + rng.randn(n_data) * noise_std
    return X, Y

def fit_linear(X, Y):
    return LinR(fit_intercept=False).fit(X, Y).coef_

def fit_lasso(X, Y, alpha):
    return LasR(alpha=alpha, fit_intercept=False).fit(X, Y).coef_

def fit_lasso_linear(X, Y, alpha):
    las = fit_lasso(X, Y, alpha)
    if np.count_nonzero(las) < X.shape[1]:
        result = np.zeros_like(las)
        if np.count_nonzero(las) == 0:
            return result
        nz = np.nonzero(las)[0]
        Xp = X[:, nz]
        lin = fit_linear(Xp, Y)
        result[nz] = lin
        return result
    else:
        return fit_linear(X, Y)

def lin_cost(X, Y, M):
    if M.ndim > 1:
        n = M.shape[0]
        cost = np.zeros(n)
        for ii, Mi in enumerate(M):
            cost[ii] = np.mean((Y - X.dot(Mi))**2)
        return cost
    else:
        return np.mean((Y - X.dot(M))**2)

def abs_cost(M, alpha):
    if M.ndim > 1:
        n = M.shape[0]
        cost = np.zeros(n)
        for ii, Mi in enumerate(M):
            cost[ii] = alpha * np.sum(abs(Mi))
        return cost
    else:
        return alpha * np.sum(abs(Mi))

def las_cost(X, Y, M, alpha):
    return lin_cost(X, Y, M) + abs_cost(M, alpha)
