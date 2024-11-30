import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def loadData(data_path="data/daily_ret_clean.csv", date_idx="date"):
    df = pd.read_csv(data_path).set_index(date_idx)
    df.index = pd.DatetimeIndex(df.index)
    return df

def pca(returns_df, n=5):
    log_df = np.log(1 + returns_df)
    stocks = returns_df.columns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_df)

    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(scaled)

    combined = np.concatenate([pca.components_, pca.explained_variance_ratio_.reshape(-1,1)], axis=1)
    components = pd.DataFrame(
        pca.components_,
        columns=list(stocks),
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    return components

def factorReg(returns_df, n=5):
    components = pca(returns_df, n=n)
    weights = components/returns_df.std(axis=0)
    factor_returns = weights.dot(returns_df.T) / len(returns_df)

    X = factor_returns.T
    Y = returns_df

    model = LinearRegression()
    model.fit(X, Y)

    coefs = pd.DataFrame(model.coef_, index=returns_df.columns, columns=X.columns)
    coefs["alpha"] = model.intercept_

    return coefs
