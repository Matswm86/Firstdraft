import logging
import pandas as pd
import os
import pytz
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from functools import lru_cache


class DataIngestion:
    """
    Data Ingestion module for both historical and live market data.
    Handles data loading, caching, and preprocessing for all market data sources.
    """

    def __init__(self, config, mt5_api=None):
        """
        Initialize DataIngestion with configuration settings.

        Args:
            config (dict): Configuration dictionary with data settings
            mt5_api (MT5API, optional): Instance of MT5API for live trading
        """
        self.config = config
        self._validate_config()

        # Extract configuration parameters
        self.data_dir = config.get('data_ingestion', {}).get('historical_data_path', '.')
        self.timeframes = config.get('data_ingestion', {}).get('timeframe_files', {})
        self.delimiter = config.get('data_ingestion', {}).get('delimiter', ',')
        self.column_names = config.get('data_ingestion', {}).get('column_names',
                                                                 ["datetime", "Open", "High", "Low", "Close", "Volume"])
        self.live_mode = config['central_trading_bot']['mode'] == 'live'
        self.symbols = config['symbols']
        self.logger = logging.getLogger(__name__)
        self.mt5_api = mt5_api

        # Initialize data cache with tuned expiry
        self.data_cache = {}
        self.cache_expiry = {}
        self.max_cache_age = {
            'tick': timedelta(seconds=5),  # Tuned to 5 seconds for tick data
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=1),
            '15min': timedelta(minutes=2),
            '30min': timedelta(minutes=5),
            '1h': timedelta(minutes=10),
            '4h': timedelta(minutes=30),
            'daily': timedelta(hours=1)
        }

        # Map string timeframes to MT5 timeframe constants
        self.timeframe_map = {
            '1min': mt5.TIMEFRAME_M1,
            '5min': mt5.TIMEFRAME_M5,
            '15min': mt5.TIMEFRAME_M15,
            '30min': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            'daily': mt5.TIMEFRAME_D1
        }

        # Log initialization
        if self.live_mode:
            self.logger.info(f"DataIngestion initialized in live mode for symbols: {', '.join(self.symbols)}")
        else:
            self.logger.info(f"DataIngestion initialized in backtest mode with data from: {self.data_dir}")

    def _validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['central_trading_bot', 'symbols']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        if not isinstance(self.config.get('symbols', []), list) or not self.config.get('symbols'):
            raise ValueError("Configuration must include a non-empty 'symbols' list")

        if self.config['central_trading_bot'].get('mode') not in ['live', 'backtest']:
            raise ValueError("Trading mode must be either 'live' or 'backtest'")

    def load_historical_data(self):
        """
        Load historical data from CSV files for backtesting.

        Returns:
            dict: A dictionary mapping (symbol, timeframe) to their corresponding DataFrames
        """
        if self.live_mode:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Historical data loading skipped in live mode")
            return {}

        historical_data = {}
        backtest_symbols = self.config['backtesting'].get('symbols', [])

        # Handle case with no backtest symbols specified
        if not backtest_symbols:
            self.logger.warning("No symbols specified for backtesting")
            return {}

        # Log the start of data loading
        self.logger.info(
            f"Loading historical data for {len(backtest_symbols)} symbols and {len(self.timeframes)} timeframes")

        for symbol in backtest_symbols:
            for tf, filename in self.timeframes.items():
                file_path = os.path.join(self.data_dir, filename)
                try:
                    # Check if file exists
                    if not os.path.exists(file_path):
                        self.logger.error(f"File not found: {file_path}")
                        historical_data[(symbol, tf)] = None
                        continue

                    # Check file size
                    file_size = os.path.getsize(file_path)
                    if file_size == 0:
                        self.logger.error(f"File is empty: {file_path}")
                        historical_data[(symbol, tf)] = None
                        continue

                    # Load CSV data
                    df = pd.read_csv(
                        file_path,
                        delimiter=self.delimiter,
                        header=0,
                        names=self.column_names,
                        parse_dates=['datetime']
                    )

                    # Validate loaded data
                    if df.empty:
                        self.logger.error(f"No data loaded from {file_path}")
                        historical_data[(symbol, tf)] = None
                        continue

                    # Set datetime as index without dropping the column
                    df_indexed = df.set_index('datetime', drop=False)

                    # Ensure all required columns exist
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_columns:
                        if col not in df_indexed.columns:
                            self.logger.error(f"Missing required column: {col} in {file_path}")
                            historical_data[(symbol, tf)] = None
                            continue

                    # Validate data ranges
                    if not self._validate_dataframe(df_indexed, symbol):
                        self.logger.warning(f"Data validation issues in {file_path}")
                        # Continue with warnings, don't return None

                    # Standardize column names to lowercase for consistency
                    df_indexed.columns = [col.lower() for col in df_indexed.columns]

                    # Store in historical data dictionary
                    historical_data[(symbol, tf)] = df_indexed
                    self.logger.info(f"Loaded historical {symbol} {tf} data from {file_path}: {len(df_indexed)} rows")
                except pd.errors.EmptyDataError:
                    self.logger.error(f"Empty data file: {file_path}")
                    historical_data[(symbol, tf)] = None
                except pd.errors.ParserError as e:
                    self.logger.error(f"Error parsing {file_path}: {str(e)}")
                    historical_data[(symbol, tf)] = None
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {str(e)}")
                    historical_data[(symbol, tf)] = None

        # Check if we have data for all combinations
        total_possible = len(backtest_symbols) * len(self.timeframes)
        loaded_count = sum(1 for df in historical_data.values() if df is not None)

        if loaded_count < total_possible:
            self.logger.warning(f"Only loaded {loaded_count}/{total_possible} data sets")
        else:
            self.logger.info(f"Successfully loaded all {total_possible} data sets")

        return historical_data

    def _validate_dataframe(self, df, symbol):
        """
        Validate dataframe contents to ensure data quality.

        Args:
            df (pd.DataFrame): DataFrame to validate
            symbol (str): Symbol being validated

        Returns:
            bool: True if validation passes, False otherwise
        """
        # Return False if DataFrame is empty to avoid invalid operations
        if df.empty:
            self.logger.warning(f"Empty DataFrame for {symbol}, skipping validation")
            return False

        # Get validation parameters from config's symbol_settings
        min_price = self.config['symbol_settings'].get(symbol, {}).get('min_price', 0.0001)
        max_price = self.config['symbol_settings'].get(symbol, {}).get('max_price', 1000000.0)
        max_volume = self.config['symbol_settings'].get(symbol, {}).get('max_volume', 1000000.0)

        validation_passed = True

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            self.logger.warning(f"Found {missing_values} missing values in {symbol} data")
            validation_passed = False

        # Check price ranges
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # Ensure Series operations return a Series, not a scalar
                below_min = df[col].lt(min_price).sum()
                above_max = df[col].gt(max_price).sum()

                if below_min > 0:
                    self.logger.warning(f"Found {below_min} values below minimum price in {symbol} {col}")
                    validation_passed = False

                if above_max > 0:
                    self.logger.warning(f"Found {above_max} values above maximum price in {symbol} {col}")
                    validation_passed = False

        # Check for negative volumes
        if 'volume' in df.columns:
            negative_volumes = df['volume'].lt(0).sum()
            if negative_volumes > 0:
                self.logger.warning(f"Found {negative_volumes} negative volume values in {symbol} data")
                validation_passed = False

            # Check for excessive volumes
            excessive_volumes = df['volume'].gt(max_volume).sum()
            if excessive_volumes > 0:
                self.logger.warning(f"Found {excessive_volumes} excessive volume values in {symbol} data")
                validation_passed = False

        # Check for OHLC anomalies
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High < Low
            invalid_hl = df['high'].lt(df['low']).sum()
            if invalid_hl > 0:
                self.logger.warning(f"Found {invalid_hl} bars where High < Low in {symbol} data")
                validation_passed = False

            # High not highest
            invalid_high = (df['high'].lt(df['open']) | df['high'].lt(df['close'])).sum()
            if invalid_high > 0:
                self.logger.warning(f"Found {invalid_high} bars where High is not highest in {symbol} data")
                validation_passed = False

            # Low not lowest
            invalid_low = (df['low'].gt(df['open']) | df['low'].gt(df['close'])).sum()
            if invalid_low > 0:
                self.logger.warning(f"Found {invalid_low} bars where Low is not lowest in {symbol} data")
                validation_passed = False

        return validation_passed

    def fetch_live_data(self, symbol):
        """
        Fetch live tick data from MT5 for a given symbol with tuned cache expiry.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD', 'GBPJPY')

        Returns:
            dict or None: Latest tick data or None if unavailable or not in live mode
        """
        if not self.live_mode or not self.mt5_api:
            return None

        if symbol not in self.symbols:
            self.logger.warning(f"Symbol {symbol} not in configured symbols list")
            return None

        cache_key = f"tick_{symbol}"
        if cache_key in self.data_cache and cache_key in self.cache_expiry:
            if datetime.now(pytz.UTC) < self.cache_expiry[cache_key]:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Using cached tick data for {symbol}")
                return self.data_cache[cache_key]

        try:
            tick = self.mt5_api.get_tick_data(symbol)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Raw tick data from MT5 for {symbol}: {tick}")
            if tick:
                if not self._validate_tick(tick, symbol):
                    self.logger.warning(f"Invalid tick data for {symbol}: {tick}")
                    return None

                # Ensure 'price' key is always present
                if 'price' not in tick or tick['price'] == 0.0 or tick['price'] is None:
                    if 'bid' in tick and 'ask' in tick and tick['bid'] is not None and tick['ask'] is not None:
                        tick['price'] = (float(tick['bid']) + float(tick['ask'])) / 2
                        self.logger.debug(f"Computed 'price' from bid/ask for {symbol}: {tick['price']}")
                    else:
                        self.logger.warning(f"Cannot compute 'price' for {symbol}: missing or invalid bid/ask in {tick}")
                        tick['price'] = 0.0  # Fallback to 0 if all else fails

                self.data_cache[cache_key] = tick
                self.cache_expiry[cache_key] = datetime.now(pytz.UTC) + self.max_cache_age['tick']

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Fetched live tick data for {symbol}: bid={tick['bid']}, ask={tick['ask']}, price={tick['price']}")
                return tick
            else:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"No tick data available for {symbol}")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching tick data for {symbol}: {str(e)}")
            return None

    def _validate_tick(self, tick, symbol):
        """
        Validate tick data.

        Args:
            tick (dict): Tick data to validate
            symbol (str): Symbol name

        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = ['symbol', 'bid', 'ask', 'timestamp']
        for field in required_fields:
            if field not in tick:
                self.logger.warning(f"Missing required field {field} in tick data for {symbol}")
                return False

        if tick['symbol'] != symbol:
            self.logger.warning(f"Symbol mismatch in tick data: expected {symbol}, got {tick['symbol']}")
            return False

        # Use config values instead of hardcoded params
        min_price = self.config['symbol_settings'].get(symbol, {}).get('min_price', 0.0001)
        max_price = self.config['symbol_settings'].get(symbol, {}).get('max_price', 1000000.0)
        max_spread_base = self.config['symbol_settings'].get(symbol, {}).get('max_spread_pips', 5.0)
        point = self.config['symbol_settings'].get(symbol, {}).get('point', 0.00001)
        max_spread_multiplier = self.config.get('trade_execution', {}).get('max_spread_multiplier', 1.5)
        max_spread = max_spread_base * max_spread_multiplier  # Apply multiplier here

        if tick['bid'] <= 0 or tick['ask'] <= 0:
            self.logger.warning(f"Non-positive bid/ask in tick data for {symbol}")
            return False

        if tick['bid'] < min_price or tick['bid'] > max_price or tick['ask'] < min_price or tick['ask'] > max_price:
            self.logger.warning(f"Bid/ask out of range for {symbol}: bid={tick['bid']}, ask={tick['ask']}")
            return False

        # Calculate spread in pips using symbol-specific point
        spread_pips = (tick['ask'] - tick['bid']) / point
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Tick spread for {symbol}: {spread_pips:.2f} pips, max allowed: {max_spread:.2f}")
        if spread_pips > max_spread:
            self.logger.warning(f"Excessive spread ({spread_pips:.2f} pips) in tick data for {symbol}, max: {max_spread:.2f}")
            return False

        try:
            tick_time = pd.to_datetime(tick['timestamp'])
            now = datetime.now(pytz.UTC)
            if tick_time.tzinfo is None:
                tick_time = tick_time.replace(tzinfo=pytz.UTC)

            time_diff = abs((tick_time - now).total_seconds())
            if time_diff > 10800:  # 3 hours
                self.logger.warning(f"Invalid timestamp in tick data for {symbol}: {tick_time} (diff: {time_diff}s)")
                return False
        except Exception as e:
            self.logger.warning(f"Could not parse timestamp in tick data for {symbol}: {str(e)}")
            return False

        return True

    def get_ohlc(self, symbol, timeframe=None, count=100, use_cache=True):
        """
        Fetch OHLC data from MT5 for a given symbol and timeframe.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD', 'GBPJPY')
            timeframe (str): Timeframe string (e.g., '1min', '15min', '1h')
            count (int): Number of bars to retrieve
            use_cache (bool): Whether to use cached data if available

        Returns:
            list or None: List of OHLC data dictionaries, or None if failed
        """
        if not self.live_mode or not self.mt5_api:
            return None

        if symbol not in self.symbols:
            self.logger.warning(f"Symbol {symbol} not in configured symbols list")
            return None

        if timeframe is None:
            timeframe = '15min'

        mt5_tf = self.timeframe_map.get(timeframe)
        if mt5_tf is None:
            self.logger.error(f"Invalid timeframe: {timeframe}")
            return None

        if use_cache:
            cache_key = f"ohlc_{symbol}_{timeframe}_{count}"
            if cache_key in self.data_cache and cache_key in self.cache_expiry:
                if datetime.now(pytz.UTC) < self.cache_expiry[cache_key]:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"Using cached OHLC data for {symbol} {timeframe}")
                    return self.data_cache[cache_key]

        try:
            ohlc_data = self.mt5_api.get_ohlc_data(symbol, mt5_tf, count)
            if ohlc_data:
                if not self._validate_ohlc(ohlc_data, symbol):
                    self.logger.warning(f"Validation issues in OHLC data for {symbol} {timeframe}")
                    # Continue with data despite warnings

                if use_cache:
                    cache_key = f"ohlc_{symbol}_{timeframe}_{count}"
                    self.data_cache[cache_key] = ohlc_data
                    self.cache_expiry[cache_key] = datetime.now(pytz.UTC) + self.max_cache_age.get(timeframe,
                                                                                                   timedelta(minutes=5))

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Fetched {len(ohlc_data)} OHLC bars for {symbol} on {timeframe}")
                return ohlc_data
            else:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"No OHLC data available for {symbol} on {timeframe}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching OHLC data for {symbol} on {timeframe}: {str(e)}")
            return []

    def _validate_ohlc(self, ohlc_data, symbol):
        """
        Validate OHLC data.

        Args:
            ohlc_data (list): List of OHLC dictionaries to validate
            symbol (str): Symbol name

        Returns:
            bool: True if valid, False otherwise
        """
        if not ohlc_data:
            return False

        min_price = self.config['symbol_settings'].get(symbol, {}).get('min_price', 0.0001)
        max_price = self.config['symbol_settings'].get(symbol, {}).get('max_price', 1000000.0)

        validation_passed = True
        anomalies = 0

        for bar in ohlc_data:
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(field in bar for field in required_fields):
                missing = [field for field in required_fields if field not in bar]
                self.logger.warning(f"Missing fields {missing} in OHLC data for {symbol}")
                anomalies += 1
                continue

            if (bar['open'] < min_price or bar['open'] > max_price or
                    bar['high'] < min_price or bar['high'] > max_price or
                    bar['low'] < min_price or bar['low'] > max_price or
                    bar['close'] < min_price or bar['close'] > max_price):
                anomalies += 1
                continue

            if bar['high'] < bar['low'] or bar['high'] < bar['open'] or bar['high'] < bar['close'] or bar['low'] > bar[
                'open'] or bar['low'] > bar['close']:
                anomalies += 1
                continue

            if bar['volume'] < 0:
                anomalies += 1
                continue

        max_anomalies = max(1, int(len(ohlc_data) * 0.02))  # Allow up to 2% anomalies
        if anomalies > max_anomalies:
            self.logger.warning(f"Found {anomalies} anomalies in OHLC data for {symbol} (max allowed: {max_anomalies})")
            validation_passed = False

        return validation_passed

    def get_dom(self, symbol):
        """
        Fetch Depth of Market data from MT5.

        Args:
            symbol (str): Trading symbol

        Returns:
            list or None: List of DOM entries, or None if unavailable
        """
        if not self.live_mode or not self.mt5_api:
            return None

        if symbol not in self.symbols:
            self.logger.warning(f"Symbol {symbol} not in configured symbols list")
            return None

        cache_key = f"dom_{symbol}"
        if cache_key in self.data_cache and cache_key in self.cache_expiry:
            if datetime.now(pytz.UTC) < self.cache_expiry[cache_key]:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Using cached DOM data for {symbol}")
                return self.data_cache[cache_key]

        try:
            dom_data = self.mt5_api.get_dom(symbol)
            if dom_data:
                if not self._validate_dom(dom_data, symbol):
                    self.logger.warning(f"Validation issues in DOM data for {symbol}")
                    # Continue with data despite warnings

                self.data_cache[cache_key] = dom_data
                self.cache_expiry[cache_key] = datetime.now(pytz.UTC) + timedelta(seconds=2)  # DOM data changes quickly

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Fetched DOM data for {symbol}: {len(dom_data)} entries")
                return dom_data
            else:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"No DOM data available for {symbol}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching DOM data for {symbol}: {str(e)}")
            return []

    def _validate_dom(self, dom_data, symbol):
        """
        Validate DOM data.

        Args:
            dom_data (list): List of DOM entries to validate
            symbol (str): Symbol name

        Returns:
            bool: True if valid, False otherwise
        """
        if not dom_data:
            return False

        min_price = self.config['symbol_settings'].get(symbol, {}).get('min_price', 0.0001)
        max_price = self.config['symbol_settings'].get(symbol, {}).get('max_price', 1000000.0)

        validation_passed = True
        anomalies = 0

        for entry in dom_data:
            required_fields = ['type', 'price', 'volume']
            if not all(field in entry for field in required_fields):
                missing = [field for field in required_fields if field not in entry]
                self.logger.warning(f"Missing fields {missing} in DOM data for {symbol}")
                anomalies += 1
                continue

            if entry['type'] not in ['bid', 'ask']:
                self.logger.warning(f"Invalid DOM entry type: {entry['type']}")
                anomalies += 1
                continue

            if entry['price'] < min_price or entry['price'] > max_price:
                anomalies += 1
                continue

            if entry['volume'] <= 0:
                anomalies += 1
                continue

        max_anomalies = max(1, int(len(dom_data) * 0.05))  # Allow up to 5% anomalies
        if anomalies > max_anomalies:
            self.logger.warning(f"Found {anomalies} anomalies in DOM data for {symbol} (max allowed: {max_anomalies})")
            validation_passed = False

        return validation_passed

    @lru_cache(maxsize=32)
    def get_timeframe_map(self):
        """
        Get mapping between string timeframes and MT5 constants.

        Returns:
            dict: Dictionary mapping timeframe strings to MT5 constants
        """
        return self.timeframe_map.copy()

    def clear_cache(self, symbol=None, timeframe=None):
        """
        Clear data cache, optionally for a specific symbol/timeframe.

        Args:
            symbol (str, optional): Symbol to clear cache for
            timeframe (str, optional): Timeframe to clear cache for
        """
        if symbol is None and timeframe is None:
            self.data_cache.clear()
            self.cache_expiry.clear()
            self.logger.info("Cleared all data cache")
        elif symbol is not None and timeframe is None:
            keys_to_clear = [k for k in self.data_cache.keys() if k.split('_')[1] == symbol]
            for key in keys_to_clear:
                if key in self.data_cache:
                    del self.data_cache[key]
                if key in self.cache_expiry:
                    del self.cache_expiry[key]
            self.logger.info(f"Cleared cache for {symbol}")
        elif symbol is None and timeframe is not None:
            keys_to_clear = [k for k in self.data_cache.keys() if k.split('_')[2] == timeframe]
            for key in keys_to_clear:
                if key in self.data_cache:
                    del self.data_cache[key]
                if key in self.cache_expiry:
                    del self.cache_expiry[key]
            self.logger.info(f"Cleared cache for {timeframe}")
        else:
            key = f"ohlc_{symbol}_{timeframe}"
            if key in self.data_cache:
                del self.data_cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]
            self.logger.info(f"Cleared cache for {symbol} {timeframe}")

    def synchronize_timeframes(self, symbol, base_timeframe='1min'):
        """
        Ensure data from different timeframes is properly synchronized.

        Args:
            symbol (str): Symbol to synchronize
            base_timeframe (str): Base timeframe to use (typically the smallest)

        Returns:
            dict: Dictionary of synchronized DataFrames by timeframe
        """
        if not self.live_mode or not self.mt5_api:
            self.logger.warning("Cannot synchronize timeframes in backtest mode")
            return {}

        result = {}

        try:
            base_data = self.get_ohlc(symbol, base_timeframe, count=10000, use_cache=False)
            if not base_data:
                self.logger.error(f"Could not fetch base timeframe data for synchronization: {symbol} {base_timeframe}")
                return {}

            base_df = pd.DataFrame(base_data)
            base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])
            base_df.set_index('timestamp', inplace=True)

            expected_ticks = pd.date_range(
                start=base_df.index.min(),
                end=base_df.index.max(),
                freq=self._get_frequency_string(base_timeframe)
            )
            missing_ticks = len(expected_ticks) - len(base_df)
            if missing_ticks > 0:
                self.logger.warning(f"Found {missing_ticks} missing bars in base timeframe for {symbol}")

            result[base_timeframe] = base_df

            for tf in self.timeframe_map.keys():
                if tf == base_timeframe:
                    continue

                tf_data = self.get_ohlc(symbol, tf, count=1000, use_cache=False)
                if not tf_data:
                    self.logger.warning(f"Could not fetch data for synchronization: {symbol} {tf}")
                    continue

                tf_df = pd.DataFrame(tf_data)
                tf_df['timestamp'] = pd.to_datetime(tf_df['timestamp'])
                tf_df.set_index('timestamp', inplace=True)

                result[tf] = tf_df

                if len(tf_df) > 0:
                    latest_base = base_df.index.max()
                    latest_tf = tf_df.index.max()
                    time_diff = abs((latest_base - latest_tf).total_seconds())

                    expected_diff = {
                        '5min': 240,
                        '15min': 840,
                        '30min': 1740,
                        '1h': 3540,
                        '4h': 14340,
                        'daily': 86340
                    }.get(tf, 0)

                    if time_diff > expected_diff + 60:
                        self.logger.warning(
                            f"Timeframe {tf} not aligned with base timeframe for {symbol}. Difference: {time_diff} seconds")

            return result
        except Exception as e:
            self.logger.error(f"Error synchronizing timeframes for {symbol}: {str(e)}")
            return {}

    def get_multi_timeframe_data(self, symbol):
        """
        Get data for a symbol across all configured timeframes.

        Args:
            symbol (str): Symbol to get data for

        Returns:
            dict: Dictionary of DataFrames by timeframe
        """
        if not self.live_mode or not self.mt5_api:
            self.logger.warning("Cannot get multi-timeframe data in backtest mode")
            return {}

        result = {}

        try:
            for tf in self.timeframe_map.keys():
                count = {
                    '1min': 1000,
                    '5min': 1000,
                    '15min': 500,
                    '30min': 500,
                    '1h': 300,
                    '4h': 200,
                    'daily': 100
                }.get(tf, 100)

                tf_data = self.get_ohlc(symbol, tf, count=count)
                if not tf_data:
                    self.logger.warning(f"Could not fetch data for {symbol} {tf}")
                    continue

                tf_df = pd.DataFrame(tf_data)
                tf_df['timestamp'] = pd.to_datetime(tf_df['timestamp'])

                tf_df.columns = [col.lower() for col in tf_df.columns]

                result[tf] = tf_df.set_index('timestamp', drop=False)

            return result
        except Exception as e:
            self.logger.error(f"Error getting multi-timeframe data for {symbol}: {str(e)}")
            return {}

    def get_market_hours_status(self, symbol):
        """
        Check if the market is currently open for a symbol.

        Args:
            symbol (str): Symbol to check

        Returns:
            dict: Market hours status information
        """
        if not self.live_mode or not self.mt5_api:
            return {"is_open": False, "reason": "Not in live mode"}

        try:
            tick = self.fetch_live_data(symbol)
            if not tick:
                return {"is_open": False, "reason": "No tick data available"}

            tick_time = pd.to_datetime(tick['timestamp'])
            if tick_time.tzinfo is None:
                tick_time = tick_time.replace(tzinfo=pytz.UTC)

            now = datetime.now(pytz.UTC)
            if (now - tick_time).total_seconds() > 300:  # 5 minutes
                return {
                    "is_open": False,
                    "reason": "Stale tick data",
                    "last_tick_time": tick_time.isoformat()
                }

            return {
                "is_open": True,
                "last_tick_time": tick_time.isoformat(),
                "bid": tick['bid'],
                "ask": tick['ask'],
                "spread": (tick['ask'] - tick['bid']) / self.config['symbol_settings'].get(symbol, {}).get('point', 0.00001)
            }
        except Exception as e:
            self.logger.error(f"Error checking market hours for {symbol}: {str(e)}")
            return {"is_open": False, "reason": f"Error: {str(e)}"}

    def _get_frequency_string(self, timeframe):
        """
        Get pandas frequency string for a timeframe.

        Args:
            timeframe (str): Timeframe string

        Returns:
            str: Pandas frequency string
        """
        freq_map = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '4h': '4h',
            'daily': 'D'
        }
        return freq_map.get(timeframe, '1min')