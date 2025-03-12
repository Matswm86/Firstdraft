import backtrader as bt
import logging
from BacktraderStrategy import BacktraderStrategy
from DataIngestion import DataIngestion


class Backtesting:
    """
    Backtesting class for running a backtest on NASDAQ futures data using Backtrader.
    Note: This remains unchanged for backtesting and does not apply to live MT5 trading with The 5%ers.
    """
    def __init__(self, config):
        """
        Initialize Backtesting with configuration settings.

        Args:
            config (dict): Configuration dictionary with backtesting settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_ingestion = DataIngestion(config)
        self.symbols = config['backtesting']['symbols']  # e.g., ["NQ 03-25"]
        self.commission = config['backtesting'].get('commission', 0.001)  # Example commission
        self.slippage = config['backtesting'].get('slippage', 0.001)  # Example slippage
        self.enable_multi_asset = config['backtesting'].get('enable_multi_asset', False)

    def run(self):
        """Run the backtest using Backtrader."""
        try:
            # Initialize Cerebro
            cerebro = bt.Cerebro()
            cerebro.addstrategy(BacktraderStrategy)

            # Load historical data
            historical_data = self.data_ingestion.load_historical_data()
            if not historical_data:
                self.logger.error("No historical data available for backtesting")
                return

            # Add data feeds (assuming 15m timeframe for backtesting as per original intent)
            for (symbol, timeframe), df in historical_data.items():
                if df is not None and timeframe == '15m':  # Filter for 15m as per requirement
                    data = bt.feeds.PandasData(dataname=df)
                    cerebro.adddata(data, name=symbol)
                    self.logger.info(f"Added {symbol} {timeframe} data to backtest")

            # Set initial cash and broker settings
            cerebro.broker.set_cash(100000)  # Starting cash
            cerebro.broker.setcommission(commission=self.commission)
            cerebro.broker.set_slippage(self.slippage)

            # Run the backtest
            self.logger.info("Starting backtest for NASDAQ futures...")
            cerebro.run()
            self.logger.info(f"Backtest completed. Final portfolio value: {cerebro.broker.getvalue():.2f}")

            # Plot results (optional, comment out if not needed)
            cerebro.plot()

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "central_trading_bot": {"mode": "backtest"},
        "data_ingestion": {
            "historical_data_path": "C:\\Users\\matsw\\PycharmProjects\\Firstdraft\\data",
            "timeframe_files": {
                "1d": "backtrader_1d.csv",
                "4h": "backtrader_4h.csv",
                "1h": "backtrader_1h.csv",
                "30m": "backtrader_30m.csv",
                "15m": "backtrader_15m.csv",
                "5m": "backtrader_5m.csv",
                "1m": "backtrader_1m.csv"
            },
            "delimiter": ",",
            "column_names": ["datetime", "Open", "High", "Low", "Close", "Volume"]
        },
        "backtesting": {
            "symbols": ["NQ 03-25"],
            "commission": 0.001,
            "slippage": 0.001,
            "enable_multi_asset": False
        }
    }
    backtesting = Backtesting(config)
    backtesting.run()