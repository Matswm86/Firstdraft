import logging
import pandas as pd
import os
from MT5API import MT5API


class DataIngestion:
    def __init__(self, config):
        """
        Initialize DataIngestion with configuration for The 5%ers MT5 trading and backtesting.

        Args:
            config (dict): Configuration dictionary with data settings.
        """
        self.config = config
        self.data_dir = config.get('data_ingestion', {}).get('historical_data_path', '.')
        self.timeframes = config.get('data_ingestion', {}).get('timeframe_files', {})
        self.delimiter = config.get('data_ingestion', {}).get('delimiter', ',')
        self.column_names = config.get('data_ingestion', {}).get('column_names',
                                                                ["datetime", "Open", "High", "Low", "Close", "Volume"])
        self.live_mode = config['central_trading_bot']['mode'] == 'live'
        self.symbols = config['symbols']  # e.g., ["EURUSD", "GBPJPY"] for live trading
        self.logger = logging.getLogger(__name__)

        # Initialize MT5API only for live mode
        if self.live_mode:
            self.mt5_api = MT5API(
                config['mt5_settings']['server'],
                config['mt5_settings']['login'],
                config['mt5_settings']['password']
            )
        else:
            self.mt5_api = None

    def load_historical_data(self):
        """
        Load historical data from CSV files for backtesting (unchanged for NASDAQ futures).

        Returns:
            dict: A dictionary mapping (symbol, timeframe) to their corresponding DataFrames.
        """
        if self.live_mode:
            self.logger.warning("Cannot load historical data in live mode; use MT5API for live data")
            return {}

        historical_data = {}
        # Assuming backtesting uses a single symbol (e.g., "NQ") as per original config
        backtest_symbol = self.config['backtesting']['symbols'][0]  # e.g., "NQ 03-25"
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
                historical_data[(backtest_symbol, timeframe)] = df
                self.logger.info(f"Loaded historical {backtest_symbol} {timeframe} data from {file_path}")
            except FileNotFoundError:
                self.logger.error(f"File not found: {file_path}")
                historical_data[(backtest_symbol, timeframe)] = None
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")
                historical_data[(backtest_symbol, timeframe)] = None
        return historical_data

    def fetch_live_data(self, symbol):
        """
        Fetch live data from MT5 for a given symbol in live trading mode.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD', 'GBPJPY').

        Returns:
            dict or None: Latest tick data or None if unavailable or not in live mode.
        """
        if not self.live_mode or not self.mt5_api:
            self.logger.info("Not in live mode or MT5API not initialized; cannot fetch live data")
            return None

        try:
            tick = self.mt5_api.get_tick_data(symbol)
            if tick:
                self.logger.debug(f"Live tick data for {symbol}: {tick}")
                return tick
            self.logger.debug(f"No live data available for {symbol} at this moment")
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch live data for {symbol}: {str(e)}")
            return None

    def get_ohlc(self, symbol, timeframe=mt5.TIMEFRAME_M15, count=100):
        """
        Fetch OHLC data from MT5 for a given symbol and timeframe in live trading mode.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD', 'GBPJPY').
            timeframe (int): MT5 timeframe constant (default: 15 minutes).
            count (int): Number of bars to retrieve (default: 100).

        Returns:
            list or None: List of OHLC data dictionaries, or None if failed or not in live mode.
        """
        if not self.live_mode or not self.mt5_api:
            self.logger.info("Not in live mode or MT5API not initialized; cannot fetch OHLC data")
            return None

        try:
            ohlc_data = self.mt5_api.get_ohlc_data(symbol, timeframe, count)
            if ohlc_data:
                self.logger.debug(f"Fetched {len(ohlc_data)} OHLC bars for {symbol}")
                return ohlc_data
            self.logger.debug(f"No OHLC data available for {symbol}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLC data for {symbol}: {str(e)}")
            return None

    def get_dom(self, symbol):
        """
        Fetch Depth of Market (DOM) data from MT5 for a given symbol in live trading mode.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD', 'GBPJPY').

        Returns:
            list or None: List of DOM entries, or None if failed or not in live mode.
        """
        if not self.live_mode or not self.mt5_api:
            self.logger.info("Not in live mode or MT5API not initialized; cannot fetch DOM data")
            return None

        try:
            dom_data = self.mt5_api.get_dom(symbol)
            if dom_data:
                self.logger.debug(f"Fetched DOM data for {symbol}: {len(dom_data)} entries")
                return dom_data
            self.logger.debug(f"No DOM data available for {symbol}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch DOM data for {symbol}: {str(e)}")
            return None


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "central_trading_bot": {"mode": "live"},
        "mt5_settings": {
            "server": "The5ers-Server",
            "login": "your_login",
            "password": "your_password"
        },
        "symbols": ["EURUSD", "GBPJPY"],
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
        "backtesting": {"symbols": ["NQ 03-25"]}
    }
    data_ingestion = DataIngestion(config)
    live_data = data_ingestion.fetch_live_data("EURUSD")
    print("Live data:", live_data)
    ohlc_data = data_ingestion.get_ohlc("EURUSD")
    print("OHLC data:", ohlc_data[:5] if ohlc_data else None)
    dom_data = data_ingestion.get_dom("EURUSD")
    print("DOM data:", dom_data)
    if data_ingestion.mt5_api:
        data_ingestion.mt5_api.shutdown()