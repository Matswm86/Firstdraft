import logging
import csv
from datetime import datetime


class TradeLogger:
    def __init__(self, config):
        """
        Initialize the TradeLogger module with configuration settings.

        Args:
            config (dict): Configuration dictionary with logging settings.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Extract logging settings from config
        self.log_file = config.get('trade_log_file', 'trades.csv')
        self.log_level = config.get('log_level', 'INFO').upper()
        self.real_time_log_level = config.get('real_time_log_level', 'DEBUG').upper()

        # Set up file logging for trades
        self.setup_trade_logging()

    def setup_trade_logging(self):
        """
        Set up CSV file for detailed trade logging if not already present.
        """
        try:
            # Check if file exists; if not, create it with headers
            write_headers = not self.log_file_exists()
            with open(self.log_file, 'a', newline='') as f:
                fieldnames = ['timestamp', 'symbol', 'action', 'entry_price',
                              'position_size', 'stop_loss', 'take_profit', 'profit_loss']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_headers:
                    writer.writeheader()
            self.logger.info(f"Trade log file set up at {self.log_file}")
        except Exception as e:
            self.logger.error(f"Failed to set up trade log file: {str(e)}")

    def log_file_exists(self):
        """
        Check if the trade log file already exists.

        Returns:
            bool: True if file exists, False otherwise.
        """
        import os
        return os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0

    def log_trade(self, trade):
        """
        Log a single trade to both the console/file logger and a CSV file.

        Args:
            trade (dict): Trade data with keys like 'symbol', 'action', 'entry_price', etc.
        """
        try:
            # Prepare trade data with timestamp
            trade_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': trade.get('symbol', 'Unknown'),
                'action': trade.get('action', 'Unknown'),
                'entry_price': trade.get('entry_price', 0.0),
                'position_size': trade.get('position_size', 0),
                'stop_loss': trade.get('stop_loss', 0.0),
                'take_profit': trade.get('take_profit', 0.0),
                'profit_loss': trade.get('profit_loss', 0.0)
            }

            # Log to console/file logger
            log_message = (f"Trade Logged - Symbol: {trade_data['symbol']}, "
                           f"Action: {trade_data['action']}, Entry: {trade_data['entry_price']}, "
                           f"Size: {trade_data['position_size']}, Profit/Loss: {trade_data['profit_loss']}")
            self.logger.info(log_message)

            # Append to CSV file
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trade_data.keys())
                writer.writerow(trade_data)

        except Exception as e:
            self.logger.error(f"Failed to log trade: {str(e)}")

    def log_trades(self, signals):
        """
        Log multiple trades or signals.

        Args:
            signals (list): List of trade/signal dictionaries to log.
        """
        if not signals:
            self.logger.info("No trades to log.")
            return

        self.logger.info(f"Logging {len(signals)} trades.")
        for signal in signals:
            self.log_trade(signal)


# Example usage (for testing purposes, commented out)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "trade_log_file": "trades.csv",
        "log_level": "INFO",
        "real_time_log_level": "DEBUG"
    }
    logger = TradeLogger(config)
    sample_trades = [
        {"symbol": "NQ", "action": "buy", "entry_price": 15000, "position_size": 2,
         "stop_loss": 14950, "take_profit": 15100, "profit_loss": 200},
        {"symbol": "NQ", "action": "sell", "entry_price": 15100, "position_size": 1,
         "stop_loss": 15150, "take_profit": 15000, "profit_loss": -50}
    ]
    logger.log_trades(sample_trades)