from strategy_lib import calculate_sharpe_ratio, calculate_max_drawdown

class QA:
    def __init__(self, logger=None):
        # Quality Assurance agent: validates performance, robustness, and stability.
        self.log = logger if logger is not None else print

    def validate_performance(self, final_metrics, train_metrics=None):
        # Compare training vs testing metrics to detect overfitting or risk issues.
        issues = []
        sharpe = final_metrics.get('sharpe', 0)
        drawdown = final_metrics.get('drawdown', 0)
        train_sharpe = train_metrics.get('sharpe', 0) if train_metrics else None
        train_drawdown = train_metrics.get('drawdown', 0) if train_metrics else None

        # Overfitting detection
        if train_metrics:
            if train_sharpe and sharpe < 0.8 * train_sharpe:
                issues.append("Sharpe ratio dropped significantly from training to validation.")
            if train_drawdown and drawdown > 1.5 * train_drawdown:
                issues.append("Drawdown increased excessively in validation phase.")

        # Absolute checks
        if sharpe < 0.5:
            issues.append("Low Sharpe ratio (<0.5): weak risk-adjusted returns.")
        if sharpe < 0:
            issues.append("Negative Sharpe ratio: losing relative to risk.")
        if drawdown > 0.3:
            issues.append("High max drawdown (>30%): potential capital risk.")

        if issues:
            for i in issues:
                self.log(f"QA Warning: {i}")
        else:
            self.log("QA: Performance metrics are within acceptable thresholds.")
        return issues

    def stability_analysis(self, equity, start_index_of_test=None):
        # Evaluate equity curve stability across sub-periods.
        import pandas as pd
        issues = []
        if equity is None or len(equity) == 0:
            return ["No equity data for stability analysis."]

        returns = equity.pct_change().fillna(0)

        if start_index_of_test is not None and start_index_of_test < len(equity):
            equity = equity.iloc[start_index_of_test:]
            returns = returns.iloc[start_index_of_test:]

        n = len(equity)
        if n < 20:
            issues.append("Too few data points for robust stability test.")
            return issues

        half = n // 2
        first_half, second_half = returns.iloc[:half], returns.iloc[half:]
        sharpe_first = calculate_sharpe_ratio(first_half)
        sharpe_second = calculate_sharpe_ratio(second_half)

        if sharpe_second < 0 and sharpe_first > 0:
            issues.append("Profitability reversed: early gains, later losses.")
        elif sharpe_second < sharpe_first * 0.5:
            issues.append("Performance declined in second half.")

        # Monthly consistency
        try:
            monthly = equity.resample('M').last().pct_change().dropna()
            positive_months = (monthly > 0).sum()
            total = len(monthly)
            pct_positive = positive_months / total * 100 if total > 0 else 0
            self.log(f"QA: {pct_positive:.1f}% positive months out of {total}.")
            if pct_positive < 50:
                issues.append("Less than half of months were profitable.")
        except Exception:
            pass

        if issues:
            for i in issues:
                self.log(f"QA Stability Warning: {i}")
        else:
            self.log("QA: Stability across periods appears satisfactory.")
        return issues
