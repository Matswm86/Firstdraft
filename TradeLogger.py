import logging
import csv
from datetime import datetime
import os
import pytz

class TradeLogger:
    def __init__(self, config):
        """
        Initialize the TradeLogger module with configuration settings for The 5%ers MT5 trading.

        Args:
            config (dict): Configuration dictionary with logging settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract logging settings from config
        self.log_file = config.get('trade_log_file', 'logs/trades.csv')
        self.log_level = config.get('log_level', 'INFO').upper()
        self.real_time_log_level = config.get('real_time_log_level', 'DEBUG').upper()

        # Set up file logging for trades
        self.setup_trade_logging()

    def setup_trade_logging(self):
        """
        Set up CSV file for detailed trade logging if not already present.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_file) or '.', exist_ok=True)

            # Check if file exists; if not, create it with headers
            write_headers = not self.log_file_exists()
            with open(self.log_file, 'a', newline='') as f:
                fieldnames = ['timestamp', 'symbol', 'action', 'entry_price',
                              'position_size', 'stop_loss', 'take_profit', 'profit_loss', 'order_id']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_headers:
                    writer.writeheader()
            self.logger.info(f"Trade log file set up at {self.log_file}")
        except Exception as e:
            self.logger.error(f"Failed to set up trade log file: {str(e)}")

    def log_file_exists(self):
        """
        Check if the trade log file already exists and is non-empty.

        Returns:
            bool: True if file exists and has content, False otherwise.
        """
        return os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0

    def log_trade(self, trade):
        """
        Log a single trade to both the console/file logger and a CSV file.

        Args:
            trade (dict): Trade data with keys like 'symbol', 'action', 'entry_price', 'order_id', etc.
        """
        try:
            # Prepare trade data with timestamp and order_id
            trade_data = {
                'timestamp': trade.get('timestamp', datetime.utcnow().isoformat()),
                'symbol': trade.get('symbol', 'Unknown'),
                'action': trade.get('action', 'Unknown'),
                'entry_price': float(trade.get('entry_price', 0.0)),
                'position_size': float(trade.get('position_size', trade.get('volume', 0))),  # Updated for MT5 volume
                'stop_loss': float(trade.get('stop_loss', 0.0)),
                'take_profit': float(trade.get('take_profit', 0.0)),
                'profit_loss': float(trade.get('profit_loss', 0.0)),
                'order_id': trade.get('order_id', 'N/A')
            }

            # Log to console/file logger
            log_message = (f"Trade Logged - Symbol: {trade_data['symbol']}, "
                           f"Action: {trade_data['action']}, Entry: {trade_data['entry_price']}, "
                           f"Size: {trade_data['position_size']}, Profit/Loss: {trade_data['profit_loss']}, "
                           f"Order ID: {trade_data['order_id']}")
            self.logger.info(log_message)

            # Append to CSV file
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trade_data.keys())
                writer.writerow(trade_data)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid trade data format: {str(e)} - Trade: {trade}")
        except Exception as e:
            self.logger.error(f"Failed to log trade: {str(e)}")

    def log_trades(self, signals):
        """
        Log multiple trades or signals.

        Args:
            signals (list): List of trade/signal dictionaries to log.
        """
        if not signals:
            self.logger.info("No trades to log")
            return

        self.logger.info(f"Logging {len(signals)} trades")
        for signal in signals:
            self.log_trade(signal)


# Example usage (for testing purposes, commented out)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "trade_log_file": "logs/trades.csv",
        "log_level": "INFO",
        "real_time_log_level": "DEBUG"
    }
    logger = TradeLogger(config)
    sample_trades = [
        {"symbol": "EURUSD", "action": "buy", "entry_price": 1.0900, "position_size": 0.1,
         "stop_loss": 1.0890, "take_profit": 1.0920, "profit_loss": 20.0, "order_id": "12345"},
        {"symbol": "GBPJPY", "action": "sell", "entry_price": 150.00, "position_size": 0.2,
         "stop_loss": 150.50, "take_profit": 149.00, "profit_loss": -50.0, "order_id": "12346"}
    ]
    logger.log_trades(sample_trades)