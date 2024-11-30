import numpy as np
from pca_helpers import factorReg
import matplotlib.pyplot as plt

class Strategy():
    def __init__(self, pca_components=1):
        self.weights = None
        self.pca_components = pca_k

    def trainWeights(self, train_df):
        """
        Returns weights for an alpha portfolio
        """
        coefs = factorReg(train_df, n=self.pca_components)
        ordered_alpha = coefs.sort_values(ascending=False, by="PC1")

        num_stocks = len(ordered_alpha)
        quintile_num = int(num_stocks / 5)

        top_quintile = ordered_alpha.iloc[:quintile_num].index
        bottom_quintile = ordered_alpha.iloc[-quintile_num:].index

        self.weights = coefs["alpha"] / coefs["alpha"].abs().sum()

    def testWeights(self, test_df, plot=True):
        """
        Returns summary statistics for running the weights on test_df
        """
        market_returns = test_df.mean(axis=1)
        strat_returns = (self.weights * test_df).sum(axis=1)

        market_cumulative = (1 + market_returns).cumprod()
        strat_cumulative = (1 + strat_returns).cumprod()

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(market_cumulative, label="Market (Equal-Weighted)", linewidth=2)
            plt.plot(strat_cumulative, label="Market-Neutral Portfolio", linewidth=2)

            plt.title("Cumulative Returns Comparison", fontsize=14)
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Cumulative Returns", fontsize=12)
            plt.legend()
            plt.grid(True)

            plt.show()

        return {
            "Daily Sharpe": self.sharpe(strat_returns),
            "Annualized Sharpe": self.sharpe(strat_returns, annual=True),
            "Daily Information": self.information(strat_returns, market_returns),
            "Annualized Information": self.information(strat_returns, market_returns, annual=True)
        }

    def sharpe(self, rets, *, annual=False):
        """
        Calculate Sharpe ratio from simple returns
        """
        returns = np.log(1+rets)
        if annual:
            returns = returns.resample("YE").sum()

        return (np.e ** returns-1).mean() / (np.e ** returns-1).std()

    def information(self, rets, market_rets, *, annual=False):
        """
        Calculate Information ratio based on simple strategy and market returns
        """
        returns = np.log(1+rets)
        market_returns = np.log(1+market_rets)

        if annual:
            returns = returns.resample("YE").sum()
            market_returns = market_returns.resample("YE").sum()
        
        return (np.e ** returns - np.e ** market_returns).mean() / (np.e ** returns).std()

