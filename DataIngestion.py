import pandas as pd
import os
import logging


class DataIngestion:
    def __init__(self, config):

        """
        Initialize the DataIngestion module.

        Args:
            config (dict): Configuration for data ingestion.
        """
        self.data_dir = config.get('historical_data_path', '.')
        # Note: we expect the key "timeframe_files" for file mappings
        self.timeframes = config.get('timeframe_files', {})
        self.delimiter = config.get('delimiter', ',')
        self.column_names = config.get('column_names', ["datetime", "Open", "High", "Low", "Close", "Volume"])
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

    def fetch_live_data(self):
        """
        Placeholder method for fetching live market data.

        Returns:
            DataFrame or None: Live data in DataFrame format.
        """
        self.logger.info("Fetching live data is not implemented yet.")
        # TODO: Implement live data fetching logic based on your data source.
        return None


if __name__ == '__main__':
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