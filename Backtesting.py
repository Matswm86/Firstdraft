import pandas as pd
import json
import logging
from SignalGenerator import SignalGenerator
from RiskManagement import RiskManagement
from TradeExecution import TradeExecution
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        try:
            df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")
            historical_data[timeframe] = df
            logger.info(f"Loaded historical data for {timeframe} from {file_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    return historical_data


def simulate_trades(signal_generator, risk_management, trade_execution, historical_data):
    """
    Runs backtesting simulation by generating signals, applying risk management, and simulating trades.

    Args:
        signal_generator (SignalGenerator): Instance of SignalGenerator module.
        risk_management (RiskManagement): Instance of RiskManagement module.
        trade_execution (TradeExecution): Instance of TradeExecution module for backtest simulation.
        historical_data (dict): Historical data for multiple timeframes.

    Returns:
        list: List of trade results, each containing profit information.
    """
    results = []
    signal_generator.load_historical_data(historical_data)

    for tf in signal_generator.trading_timeframes:
        try:
            signal_generator.generate_signal_for_tf(tf)
            if hasattr(signal_generator, 'last_signal') and signal_generator.last_signal:
                adjusted_signal = risk_management.evaluate_signal(signal_generator.last_signal)
                if adjusted_signal:
                    # Simulate trade execution using TradeExecution
                    trade_results = trade_execution.execute_backtest_trades()
                    if trade_results:
                        results.extend(trade_results)
                        logger.info(f"Simulated trades for {tf}: {len(trade_results)} trades")
        except Exception as e:
            logger.error(f"Error simulating trades for {tf}: {str(e)}")

    return results


def calculate_performance_metrics(results):
    """
    Computes win rate and profit factor from trade results.

    Args:
        results (list): List of trade results with profit data.
    """
    if not results:
        logger.warning("No trade results to analyze")
        print("Win Rate: N/A")
        print("Profit Factor: N/A")
        return

    wins = sum(1 for r in results if r["profit_loss"] > 0)
    losses = sum(1 for r in results if r["profit_loss"] < 0)
    total_trades = wins + losses

    if total_trades == 0:
        win_rate = 0
        profit_factor = 0
    else:
        win_rate = wins / total_trades * 100
        profit_factor = (sum(r["profit_loss"] for r in results if r["profit_loss"] > 0) /
                         abs(sum(r["profit_loss"] for r in results if r["profit_loss"] < 0))) if losses > 0 else float(
            'inf')

    logger.info(f"Backtest Results - Win Rate: {win_rate:.2f}%, Profit Factor: {profit_factor:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")


def visualize_results(results):
    """
    Plots the backtesting equity curve.

    Args:
        results (list): List of trade results with profit data.
    """
    if not results:
        logger.warning("No trade results to visualize")
        return

    equity_curve = pd.Series([r["profit_loss"] for r in results]).cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity Curve", color="blue")
    plt.title("Backtesting Performance")
    plt.xlabel("Trades")
    plt.ylabel("Cumulative P/L")
    plt.legend()
    plt.grid(True)
    plt.show()
    logger.info("Equity curve visualized")


# Initialize modules with configuration
signal_generator = SignalGenerator(config['signal_generation'])
risk_management = RiskManagement(config['risk_management'])
trade_execution = TradeExecution(config['ninja_trader_api'], None)  # No NinjaTraderAPI for backtesting

# Load historical data
historical_data = load_historical_data(config)

# Run backtest
trade_results = simulate_trades(signal_generator, risk_management, trade_execution, historical_data)
calculate_performance_metrics(trade_results)
visualize_results(trade_results)