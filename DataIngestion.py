import pandas as pd
import os
import logging

class DataIngestion:
    def __init__(self, config, ninja_trader_api=None):
        """
        Initialize the DataIngestion module.

        Args:
            config (dict): Configuration for data ingestion.
            ninja_trader_api (NinjaTraderAPI, optional): Instance for live data in live mode.
        """
        self.data_dir = config.get('historical_data_path', '.')
        self.timeframes = config.get('timeframe_files', {})
        self.delimiter = config.get('delimiter', ',')
        self.column_names = config.get('column_names', ["datetime", "Open", "High", "Low", "Close", "Volume"])
        self.ninja_trader_api = ninja_trader_api  # For live trading, optional
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_historical_data(self):
        """
        Load historical data from CSV files based on the configuration.

        Returns:
            dict: A dictionary mapping timeframes to their corresponding DataFrames.
        """
        historical_data = {}
        for timeframe, filename in self.timeframes.items():
            file_path = os.path.join(self.data_dir, filename)
            try:
                df = pd.read_csv(file_path, delimiter=self.delimiter, header=0,
                                 names=self.column_names, index_col='datetime', parse_dates=True)
                historical_data[timeframe] = df
                self.logger.info(f"Loaded cleaned {timeframe} data from {file_path}")
            except FileNotFoundError:
                self.logger.error(f"File not found: {file_path}")
                raise
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")
                raise
        return historical_data

    def start_websocket(self, subscriptions):
        """
        Start the WebSocket connection for live data streaming.

        Args:
            subscriptions (list): List of subscription requests (e.g., symbols and data types).
        """
        if self.ninja_trader_api:
            try:
                self.ninja_trader_api.start_websocket(subscriptions)
                self.logger.info("Started WebSocket connection for live data.")
            except Exception as e:
                self.logger.error(f"Failed to start WebSocket: {str(e)}")
        else:
            self.logger.error("NinjaTraderAPI not provided; cannot start WebSocket.")

    def fetch_live_data(self):
        """
        Fetch live market data from NinjaTraderAPI.

        Returns:
            dict or None: Latest tick data as a dictionary, or None if not available.
        """
        if self.ninja_trader_api:
            try:
                live_data = self.ninja_trader_api.get_live_data()
                if live_data:
                    self.logger.debug(f"Fetched live tick: {live_data}")
                    # Return raw tick dictionary as expected by SignalGenerator
                    return live_data
                else:
                    self.logger.debug("No new live data available yet.")
                    return None
            except Exception as e:
                self.logger.error(f"Failed to fetch live data: {str(e)}")
                return None
        else:
            self.logger.info("NinjaTraderAPI not provided; cannot fetch live data.")
            return None

# Example usage (for testing, commented out)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "historical_data_path": r"C:\Users\matsw\PycharmProjects\Firstdraft\data",
        "timeframe_files": {
            "1d": "backtrader_1d.csv",
            "4h": "backtrader_4h.csv",
            "1h": "backtrader_1h.csv",
            "30m": "backtrader_30m.csv",
            "15m": "backtrader_15m.csv",
            "5m": "backtrader_5m.csv"
        },
        "delimiter": ",",
        "column_names": ["datetime", "Open", "High", "Low", "Close", "Volume"]
    }
    di = DataIngestion(config)
    data = di.load_historical_data()
    print(data['5m'].head())