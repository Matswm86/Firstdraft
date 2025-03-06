import pandas as pd
import json
from SignalGenerator import SignalGenerator
from RiskManagement import RiskManagement
import matplotlib.pyplot as plt

# Load backtest settings from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)


def load_historical_data(config):
    """
    Loads historical data for multiple timeframes based on the configuration.

    Args:
        config (dict): Configuration dictionary containing data paths and timeframe files.

    Returns:
        dict: Dictionary mapping timeframes to their respective DataFrames.
    """
    historical_data = {}
    data_dir = config['data_ingestion']['historical_data_path']
    for timeframe, filename in config['data_ingestion']['timeframe_files'].items():
        file_path = f"{data_dir}/{filename}"
        df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")
        historical_data[timeframe] = df
    return historical_data


def simulate_trades(signal_generator, risk_management, historical_data):
    """
    Runs backtesting simulation by generating signals and applying risk management.

    Args:
        signal_generator (SignalGenerator): Instance of SignalGenerator module.
        risk_management (RiskManagement): Instance of RiskManagement module.
        historical_data (dict): Historical data for multiple timeframes.

    Returns:
        list: List of trade results, each containing profit information.
    """
    results = []
    signal_generator.load_historical_data(historical_data)
    for tf in signal_generator.trading_timeframes:
        signals = signal_generator.generate_signal_for_tf(tf)
        for signal in signals:
            adjusted_signal = risk_management.evaluate_signal(signal)
            if adjusted_signal:
                # Simulate trade execution
                entry_price = adjusted_signal['entry_price']
                position_size = adjusted_signal['position_size']
                stop_loss = adjusted_signal['stop_loss']
                take_profit = adjusted_signal['take_profit']
                # Assume trade closes at take-profit or stop-loss (simplified)
                exit_price = take_profit if adjusted_signal['action'] == 'buy' else stop_loss
                profit = (exit_price - entry_price) * position_size if adjusted_signal['action'] == 'buy' else (
                                                                                                                           entry_price - exit_price) * position_size
                results.append({"profit": profit})
    return results


def calculate_performance_metrics(results):
    """
    Computes win rate and profit factor from trade results.

    Args:
        results (list): List of trade results with profit data.
    """
    wins = sum(1 for r in results if r["profit"] > 0)
    losses = sum(1 for r in results if r["profit"] < 0)
    if losses == 0:
        profit_factor = float('inf')
    else:
        profit_factor = sum(r["profit"] for r in results if r["profit"] > 0) / abs(
            sum(r["profit"] for r in results if r["profit"] < 0))
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")


def visualize_results(results):
    """
    Plots the backtesting equity curve.

    Args:
        results (list): List of trade results with profit data.
    """
    equity_curve = pd.Series([r["profit"] for r in results]).cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity Curve", color="blue")
    plt.title("Backtesting Performance")
    plt.xlabel("Trades")
    plt.ylabel("Cumulative P/L")
    plt.legend()
    plt.show()


# Initialize modules with configuration
signal_generator = SignalGenerator(config['signal_generation'])
risk_management = RiskManagement(config['risk_management'])

# Load historical data
historical_data = load_historical_data(config)

# Run backtest
trade_results = simulate_trades(signal_generator, risk_management, historical_data)
calculate_performance_metrics(trade_results)
visualize_results(trade_results)