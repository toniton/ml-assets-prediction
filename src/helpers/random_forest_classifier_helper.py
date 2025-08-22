import numpy as np
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split

from joblib import parallel_backend


class RandomForestClassifierHelper:

    def __init__(self):
        self.n_estimators = range(250, 800, 20)
        self.min_samples_split = range(75, 500, 15)
        self.max_depth = range(10, 50, 2)
        self.random_state = 5
        self.no_of_iterations = 150

    @staticmethod
    def _print_report(y_test, y_pred, best_params):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Print the best hyperparameters
        print('Best hyperparameters:', best_params)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("Classification report:", classification_report(y_test, y_pred))
        print("Predictions:", y_pred)

    @staticmethod
    def _backtest(df, y_pred, fee_per_trade=0.001):
        """
        Safe backtesting for long/flat strategy using next-day predictions.

        Parameters
        ----------
        df : DataFrame
            Must contain 'close' prices aligned with predictions.
        y_pred : array-like
            Model predictions for next-day direction (0/1), aligned to df.index.
        fee_per_trade : float
            Proportional cost per trade (0.001 = 0.1% per trade).

        Prints annualized return, equity curve, max drawdown, and Sharpe ratio.
        """
        df = df.copy().sort_values(by="timestamp", ascending=True)

        # 1. Compute next-day log returns
        df["log_ret_1d"] = np.log(df["close"] / df["close"].shift(1))

        # 2. Assign positions: 1 if predicting up, 0 otherwise
        df["pos"] = y_pred.astype(float)

        # 3. Strategy gross daily return in log space
        df["strat_log"] = df["pos"] * df["log_ret_1d"]

        # 4. Cost: apply when position changes
        df["cost"] = df["pos"].diff().abs().fillna(0) * fee_per_trade

        # 5. Net daily return
        df["net_ret"] = np.exp(df["strat_log"]) - 1 - df["cost"]

        # 6. Clip extreme values and fill NaNs
        df["net_ret"] = df["net_ret"].fillna(0).clip(lower=-0.99)

        # 7. Net log return for equity compounding
        df["net_log"] = np.log1p(df["net_ret"])

        # 8. Equity curve
        equity = np.exp(df["net_log"].cumsum())

        # 9. Annualized return
        ann_return = equity.iloc[-1] ** (365 / len(df)) - 1

        # 10. Max Drawdown
        roll_max = equity.cummax()
        drawdown = equity / roll_max - 1
        max_drawdown = drawdown.min()

        # 11. Sharpe Ratio (daily returns)
        daily_ret = df["net_log"]
        sharpe_ratio = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)

        # 12. Report
        print("Backtesting report")
        print(f"Annual return: {ann_return:.4f}")
        print(f"Equity curve (last 5 rows):\n{equity.tail()}")
        print(f"Max Drawdown: {max_drawdown:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    def train_model(self, data: DataFrame, target: Series) -> RandomForestClassifier:
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, shuffle=False,)

        random_forest_classifier = RandomForestClassifier(
            random_state=self.random_state,
            class_weight="balanced"
        )
        param_dist = {
            'n_estimators': self.n_estimators,
            'min_samples_split': self.min_samples_split,
            'max_depth': self.max_depth
        }
        rand_search = RandomizedSearchCV(
            estimator=random_forest_classifier,
            param_distributions=param_dist,
            n_iter=self.no_of_iterations,
            # cv=5,
            cv=TimeSeriesSplit(n_splits=10),
            scoring="recall",
            random_state=self.random_state
        )

        with parallel_backend('threading', n_jobs=12):
            print(X_train.shape, y_train.shape)
            rand_search.fit(X_train, y_train)

        best_rf = rand_search.best_estimator_
        best_params = rand_search.best_params_

        y_pred = best_rf.predict(X_test)
        self._print_report(y_test, y_pred, best_params)
        self._backtest(X_test, y_pred)

        return best_rf
