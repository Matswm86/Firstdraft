import logging
import pandas as pd
import os

class DataIngestion:
    def __init__(self, config, ninja_trader_api=None):
        """
        Initialize DataIngestion with configuration and optional NinjaTraderAPI instance.

        Args:
            config (dict): Configuration dictionary with data settings.
            ninja_trader_api (NinjaTraderAPI, optional): Instance for live data.
        """
        self.config = config
        self.data_dir = config.get('historical_data_path', '.')
        self.timeframes = config.get('timeframe_files', {})
        self.delimiter = config.get('delimiter', ',')
        self.column_names = config.get('column_names', ["datetime", "Open", "High", "Low", "Close", "Volume"])
        self.ninja_trader_api = ninja_trader_api
        self.live_mode = self.ninja_trader_api is not None
        self.subscriptions = config.get('live_trading', {}).get('subscriptions', [])
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_historical_data(self):
        """
        Load historical data from CSV files for all configured timeframes.

        Returns:
            dict: A dictionary mapping timeframes to their corresponding DataFrames.
        """
        if self.live_mode:
            self.logger.warning("Cannot load historical data in live mode")
            return {}

        historical_data = {}
        for timeframe, filename in self.timeframes.items():
            file_path = os.path.join(self.data_dir, filename)
            try:
                df = pd.read_csv(
                    file_path,
                    delimiter=self.delimiter,
                    header=0,
                    names=self.column_names,
                    index_col='datetime',
                    parse_dates=True
                )
                historical_data[timeframe] = df
                self.logger.info(f"Loaded cleaned {timeframe} data from {file_path}")
            except FileNotFoundError:
                self.logger.error(f"File not found: {file_path}")
                historical_data[timeframe] = None
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")
                historical_data[timeframe] = None
        return historical_data

    def start_websocket(self):
        """
        Start the WebSocket connection for live data if in live mode.
        """
        if self.live_mode and self.ninja_trader_api:
            try:
                self.ninja_trader_api.start_websocket(self.subscriptions)
                self.logger.info("WebSocket started for live data ingestion")
            except Exception as e:
                self.logger.error(f"Failed to start WebSocket: {str(e)}")
        else:
            self.logger.debug("WebSocket not started: not in live mode or no NinjaTraderAPI")

    def fetch_live_data(self):
        """
        Fetch live data from NinjaTraderAPI (WebSocket or REST quote).

        Returns:
            dict or None: Latest data point or None if unavailable.
        """
        if self.live_mode and self.ninja_trader_api:
            # Try WebSocket first for tick data
            tick = self.ninja_trader_api.get_live_data()
            if tick:
                self.logger.debug(f"Live tick data: {tick}")
                return tick
            # Fallback to REST quote for aggregated data
            quote = self.ninja_trader_api.get_quote("NQ 03-25", "Sim101")
            if quote:
                self.logger.debug(f"Live quote data: {quote}")
                return quote
            self.logger.debug("No live data available at this moment")
            return None
        self.logger.info("Not in live mode or NinjaTraderAPI not provided; cannot fetch live data")
        return None

# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "ninja_trader_api": {
            "ws_url": "ws://127.0.0.1:8088",
            "rest_url": "https://app.crosstrade.io/v1/api",
            "webhook_url": "https://app.crosstrade.io/v1/send/Ih0yVAcJ/F4xqDBYyh-R-w8_6FaYUCA",
            "api_key": "A_HhRVALPcyadtJ61U0YM-20LG_aqh7IQPQQhpZWw-Q"
        },
        "live_trading": {"subscriptions": [{"command": "subscribe", "symbol": "NQ 03-25"}]},
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
    }
    from NinjaTraderAPI import NinjaTraderAPI
    ninja_api = NinjaTraderAPI(config["ninja_trader_api"])
    data_ingestion = DataIngestion(config, ninja_api)
    data_ingestion.start_websocket()
    live_data = data_ingestion.fetch_live_data()
    print("Live data:", live_data)
    historical_data = data_ingestion.load_historical_data()
    for timeframe, df in historical_data.items():
        if df is not None:
            print(f"{timeframe} data head:\n{df.head()}")