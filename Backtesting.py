import pandas as pd
import json
import SignalGenerator
import RiskManagement
import matplotlib.pyplot as plt

# Load backtest settings from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)


def load_historical_data(filename="historical_nq.csv"):
    """
    Loads historical data (OHLCV) for backtesting.
    """
    df = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
    return df


def simulate_trades(data):
    """
    Runs backtesting simulation by applying the signal generation module.
    """
    results = []
    for index, row in data.iterrows():
        signal = SignalGenerator.process_market_data(row)
        if signal:
            trade = RiskManagement.adjust_trade_risk(signal)
            results.append(trade)

    return results


def calculate_performance_metrics(results):
    """
    Computes win rate, profit factor, and max drawdown.
    """
    wins = sum(1 for r in results if r["profit"] > 0)
    losses = sum(1 for r in results if r["profit"] < 0)

    profit_factor = sum(r["profit"] for r in results if r["profit"] > 0) / abs(
        sum(r["profit"] for r in results if r["profit"] < 0))
    win_rate = wins / (wins + losses) * 100

    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")


def visualize_results(results):
    """
    Plots the backtesting equity curve.
    """
    equity_curve = pd.Series([r["profit"] for r in results]).cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity Curve", color="blue")
    plt.title("Backtesting Performance")
    plt.xlabel("Trades")
    plt.ylabel("Cumulative P/L")
    plt.legend()
    plt.show()


# Run backtest
historical_data = load_historical_data()
trade_results = simulate_trades(historical_data)
calculate_performance_metrics(trade_results)
visualize_results(trade_results)
