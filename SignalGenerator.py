import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import pytz
import logging
import MetaTrader5 as mt5
import numpy as np
import ta.volatility as volatility
import ta.momentum as momentum
from typing import Dict, Optional, List, Any
from hurst import compute_Hc
from scipy.signal import argrelextrema
import ta.trend as trend

class SignalGenerator:
    def __init__(self, config: Dict, mt5_api=None):
        """
        Initialize the SignalGenerator with configuration settings.

        Args:
            config (dict): Configuration dictionary with signal generation settings
            mt5_api (MT5API, optional): Instance of MT5API for market data
        """
        self.config = config
        self._validate_config()

        self.trading_timeframes = self.config['signal_generation']['trading_timeframes']
        self.thresholds = self.config['signal_generation']['thresholds']
        self.timeframes = self.config['signal_generation']['timeframes']
        self.max_bars = self.config['signal_generation']['max_bars']
        self.symbols = self.config['symbols']
        self.mt5_api = mt5_api
        self.logger = logging.getLogger(__name__)

        # Initialize data structures
        self.histories = {symbol: {tf: deque(maxlen=self.max_bars.get(tf, 100)) for tf in self.timeframes}
                          for symbol in self.symbols}
        self.indicator_histories = {symbol: {tf: pd.DataFrame() for tf in self.timeframes}
                                    for symbol in self.symbols}
        self.current_bars = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}
        self.last_signals = {symbol: {tf: None for tf in self.trading_timeframes} for symbol in self.symbols}

        # Real-time data tracking
        self.real_time_data = {
            symbol: {
                'cumulative_delta': 0,
                'delta_history': deque(maxlen=1000),
                'bid_ask_imbalance': 0,
                'composite_breadth_score': 0,
                'tick': 0,
                'cumulative_tick': 0,
                'last_price': None,
                'last_volume': 0
            } for symbol in self.symbols
        }

        # Map timeframe strings to timedelta objects
        self.timeframe_intervals = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            'daily': timedelta(days=1)
        }

        # Map timeframe strings to pandas frequency strings
        self.freq_map = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '4h': '4h',
            'daily': 'D'
        }

        # Map timeframe strings to MT5 timeframe constants
        self.mt5_timeframe_map = {
            '1min': mt5.TIMEFRAME_M1,
            '5min': mt5.TIMEFRAME_M5,
            '15min': mt5.TIMEFRAME_M15,
            '30min': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            'daily': mt5.TIMEFRAME_D1
        }

        # Hurst exponent caching
        self.hurst_cache = {symbol: {tf: None for tf in ['15min', '30min', '1h', '4h', 'daily']} for symbol in self.symbols}
        self.hurst_last_update = {symbol: {tf: None for tf in ['15min', '30min', '1h', '4h', 'daily']} for symbol in self.symbols}

        # Initialize missing variables
        self.atr_period = 14
        self.rsi_threshold = 70

        # Load initial history if MT5API is available
        if self.mt5_api and self.config['central_trading_bot']['mode'] == 'live':
            self._load_initial_history()

    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        required_keys = ['signal_generation', 'symbols']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        signal_gen = self.config.get('signal_generation', {})
        required_signal_keys = ['trading_timeframes', 'thresholds', 'timeframes', 'max_bars']
        for key in required_signal_keys:
            if key not in signal_gen:
                raise ValueError(f"Missing required signal_generation key: {key}")

        if not isinstance(self.config.get('symbols', []), list) or not self.config.get('symbols'):
            raise ValueError("Configuration must include a non-empty 'symbols' list")

    def _load_initial_history(self) -> None:
        """Load initial historical data from MT5"""
        for symbol in self.symbols:
            for tf in self.timeframes:
                try:
                    mt5_tf = self.mt5_timeframe_map.get(tf)
                    if mt5_tf is None:
                        self.logger.error(f"Invalid timeframe: {tf}")
                        continue

                    rates = self.mt5_api.get_ohlc_data(symbol, mt5_tf, 100)
                    if not rates:
                        self.logger.warning(f"No historical data available for {symbol} on {tf}")
                        continue

                    for rate in rates:
                        bar = {
                            'timestamp': pd.to_datetime(rate['timestamp']),
                            'open': float(rate['open']),
                            'high': float(rate['high']),
                            'low': float(rate['low']),
                            'close': float(rate['close']),
                            'volume': float(rate['volume']),
                            'symbol': symbol  # Add symbol to bar
                        }
                        self.histories[symbol][tf].append(bar)

                    if self.histories[symbol][tf]:
                        df = pd.DataFrame(list(self.histories[symbol][tf]))
                        if len(df) >= 14:
                            df = self.calculate_indicators(df, tf)
                            self.indicator_histories[symbol][tf] = df
                            self.logger.info(f"Loaded initial history for {symbol} on {tf}: {len(df)} bars")
                except Exception as e:
                    self.logger.error(f"Error loading history for {symbol} on {tf}: {str(e)}")

    def preprocess_tick(self, raw_tick: Dict) -> Optional[Dict[str, Any]]:
        """
        Preprocess tick data from MT5.

        Args:
            raw_tick (dict): Raw tick data from MT5API

        Returns:
            Optional[Dict[str, Any]]: Preprocessed tick data or None if invalid
        """
        try:
            required_fields = ['timestamp', 'symbol', 'bid', 'ask']
            for field in required_fields:
                if field not in raw_tick:
                    self.logger.error(f"Missing required field in tick data: {field}")
                    return None

            price = raw_tick.get('last', 0)
            if price == 0.0 and 'bid' in raw_tick and 'ask' in raw_tick:
                price = (raw_tick['bid'] + raw_tick['ask']) / 2
            elif price == 0.0:
                self.logger.error("Invalid tick data: No valid price available")
                return None

            return {
                'timestamp': raw_tick['timestamp'],
                'price': float(price),
                'volume': float(raw_tick.get('volume', 0)),
                'bid': float(raw_tick.get('bid', price)),
                'ask': float(raw_tick.get('ask', price)),
                'symbol': raw_tick['symbol']
            }
        except Exception as e:
            self.logger.error(f"Error preprocessing tick data: {str(e)}")
            return None

    def process_tick(self, raw_tick: Dict) -> Optional[Dict[str, Any]]:
        """
        Process a new tick and potentially generate signals.

        Args:
            raw_tick (dict): Raw tick data from MT5API

        Returns:
            Optional[Dict[str, Any]]: Generated signal if applicable, None otherwise
        """
        tick = self.preprocess_tick(raw_tick)
        if not tick:
            return None

        symbol = tick['symbol']
        if symbol not in self.symbols:
            self.logger.warning(f"Received tick for unknown symbol: {symbol}")
            return None

        try:
            if isinstance(tick['timestamp'], str):
                tick_time = pd.to_datetime(tick['timestamp'])
            else:
                tick_time = tick['timestamp']

            if tick_time.tzinfo is None:
                tick_time = tick_time.replace(tzinfo=pytz.UTC)

            price = tick['price']
            volume = tick.get('volume', 0)
            self.logger.debug(f"Processing tick for {symbol}: price={price}, volume={volume}, time={tick_time}")
        except Exception as e:
            self.logger.error(f"Error processing tick values for {symbol}: {str(e)}")
            return None

        self.update_real_time_data(tick)

        for tf in self.timeframes:
            bar = self._aggregate_tick(tf, tick_time, price, volume, symbol)
            if bar:
                self.add_bar(tf, bar, symbol)

            if tf in self.trading_timeframes:
                signal = self.generate_signal_for_tf(tf, symbol, tick_time)
                if signal:
                    self.last_signals[symbol][tf] = signal
                    return signal
                else:
                    self.logger.debug(f"No signal generated for {symbol} on {tf}")

        return None

    def _aggregate_tick(self, timeframe: str, tick_time: datetime, price: float, volume: float, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Aggregate tick data into OHLC bars.

        Args:
            timeframe (str): Timeframe string (e.g., '1min', '5min')
            tick_time (datetime): Timestamp of the tick
            price (float): Current price
            volume (float): Current volume
            symbol (str): Symbol being processed

        Returns:
            Optional[Dict[str, Any]]: Completed bar if available, None otherwise
        """
        try:
            interval = self.timeframe_intervals.get(timeframe)
            freq = self.freq_map.get(timeframe)

            if not interval or not freq:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None

            tick_timestamp = pd.Timestamp(tick_time)
            floored_time = tick_timestamp.replace(second=0, microsecond=0).floor(freq)
            current_bar = self.current_bars[symbol][timeframe]

            if current_bar is None or floored_time > current_bar['timestamp']:
                completed_bar = current_bar.copy() if current_bar else None
                self.current_bars[symbol][timeframe] = {
                    'timestamp': floored_time,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'symbol': symbol  # Add symbol to bar
                }
                if completed_bar:
                    self.histories[symbol][timeframe].append(completed_bar)
                self.logger.debug(f"Aggregated bar for {symbol} on {timeframe}: {completed_bar}")
                return completed_bar
            else:
                current_bar['high'] = max(current_bar['high'], price)
                current_bar['low'] = min(current_bar['low'], price)
                current_bar['close'] = price
                current_bar['volume'] += volume
                return None
        except Exception as e:
            self.logger.error(f"Error aggregating tick for {symbol} on {timeframe}: {str(e)}")
            return None

    def add_bar(self, timeframe: str, bar: Dict[str, Any], symbol: str) -> None:
        """
        Add a completed bar to history and recalculate indicators.

        Args:
            timeframe (str): Timeframe string
            bar (dict): Bar data dictionary
            symbol (str): Symbol being processed
        """
        try:
            if bar:
                self.histories[symbol][timeframe].append(bar)

            bars_list = list(self.histories[symbol][timeframe])
            if not bars_list:
                return

            df = pd.DataFrame(bars_list)
            if len(df) >= 14:
                df = self.calculate_indicators(df, timeframe)
                self.indicator_histories[symbol][timeframe] = df
                self.logger.debug(f"Updated indicator history for {symbol} on {timeframe}: {len(df)} bars")
        except Exception as e:
            self.logger.error(f"Error adding bar for {symbol} on {timeframe}: {str(e)}")

    def update_real_time_data(self, tick: Dict[str, Any]) -> None:
        """
        Update real-time market data metrics based on a new tick.

        Args:
            tick (dict): Preprocessed tick data
        """
        try:
            symbol = tick['symbol']
            price = tick['price']
            volume = tick.get('volume', 0)
            bid = tick.get('bid', price)
            ask = tick.get('ask', price)

            rtd = self.real_time_data.get(symbol)
            if not rtd:
                self.logger.warning(f"Real-time data not initialized for {symbol}")
                return

            if rtd['last_price'] is None:
                rtd['last_price'] = price
                rtd['last_volume'] = volume
                rtd['tick'] = 0
                return

            tick_value = 1 if price > rtd['last_price'] else -1 if price < rtd['last_price'] else 0
            rtd['tick'] = tick_value
            rtd['cumulative_tick'] += tick_value

            price_change = price - rtd['last_price']
            delta = 0
            if price_change > 0:
                delta = volume
            elif price_change < 0:
                delta = -volume

            rtd['cumulative_delta'] += delta
            rtd['delta_history'].append(delta)

            if bid != ask and (bid + ask) > 0:
                rtd['bid_ask_imbalance'] = (bid - ask) / (bid + ask)

            rtd['composite_breadth_score'] = (
                rtd['composite_breadth_score'] * 0.9 +
                rtd['tick'] * 0.1
            )

            rtd['last_price'] = price
            rtd['last_volume'] = volume
            self.logger.debug(f"Updated RTD for {symbol}: tick={tick_value}, cumulative_tick={rtd['cumulative_tick']}")
        except Exception as e:
            self.logger.error(f"Error updating real-time data: {str(e)}")

    def calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calculate technical indicators for a DataFrame of price data.

        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            timeframe (str): Timeframe string

        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column for indicators: {col}")
                    return df

            # Use dynamic atr_period
            df['atr'] = volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'],
                window=self.atr_period, fillna=True
            ).average_true_range().fillna(0.001)

            lookback = 30
            if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]):
                lookback = max(1, min(100, int(df['atr'].iloc[-1] * 1000)))

            df['adaptive_ma'] = df['close'].rolling(window=lookback, min_periods=1).mean()

            df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi().fillna(50)

            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            else:
                df['vwap'] = df['close']

            if 'vwap' in df.columns:
                df['vwap_slope'] = df['vwap'].diff(periods=5) / 5

            mean = df['close'].rolling(window=20, min_periods=1).mean()
            std = df['close'].rolling(window=20, min_periods=1).std()
            df['zscore'] = np.where(std > 0, (df['close'] - mean) / std, 0)

            # Hurst calculation (only for key timeframes)
            if timeframe in ['15min', '30min', '1h', '4h', 'daily'] and len(df) > 100:
                H, _, _ = compute_Hc(df['close'].values, kind='price')
                df['hurst'] = H

            if 'adaptive_ma' in df.columns:
                df['sma_slope'] = (df['adaptive_ma'] - df['adaptive_ma'].shift(5)) / 5

            if 'atr' in df.columns:
                df['avg_atr'] = df['atr'].rolling(window=20, min_periods=1).mean()

            if 'vwap' in df.columns:
                df['vwap_std'] = (df['close'] - df['vwap']).rolling(window=20, min_periods=1).std()
                df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
                df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
                df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
                df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']

            self.logger.debug(f"Calculated indicators for {timeframe}: close={df['close'].iloc[-1]}, adaptive_ma={df['adaptive_ma'].iloc[-1]}, sma_slope={df['sma_slope'].iloc[-1] if 'sma_slope' in df.columns else 'N/A'}")
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df

    def compute_hurst(self, symbol: str) -> Dict[str, float]:
        """Compute Hurst exponent dynamically with caching for key timeframes."""
        timeframes = ['15min', '30min', '1h', '4h', 'daily']
        hurst_values = {}
        current_time = datetime.now(pytz.UTC)
        for tf in timeframes:
            # Check cache and update interval (e.g., 5 minutes)
            last_update = self.hurst_last_update[symbol][tf]
            if last_update and (current_time - last_update).total_seconds() < 300:  # 5 minutes
                hurst_values[tf] = self.hurst_cache[symbol][tf]
                continue

            df = self.indicator_histories.get(symbol, {}).get(tf, pd.DataFrame())
            if df.empty or len(df) < 50:  # Reduced window size for performance
                continue
            series = df['close'].tail(50)  # Smaller window for performance
            H, _, _ = compute_Hc(series, kind='price')
            hurst_values[tf] = H
            self.hurst_cache[symbol][tf] = H
            self.hurst_last_update[symbol][tf] = current_time
        return hurst_values

    def check_confluence_persistence(self, df: pd.DataFrame, timeframe: str, current_time: datetime) -> bool:
        """
        Check if signal confluence persists across multiple bars, enhanced with Hurst.

        Args:
            df (pd.DataFrame): DataFrame with indicator data
            timeframe (str): Timeframe string
            current_time (datetime): Current tick time from MT5

        Returns:
            bool: True if confluence persists, False otherwise
        """
        try:
            required_bars = 3
            min_duration = timedelta(minutes=2)

            if len(df) < required_bars + 1:
                self.logger.debug(f"Insufficient bars for {timeframe}: {len(df)} < {required_bars + 1}")
                return False

            if 'timestamp' not in df.columns:
                self.logger.error("DataFrame missing timestamp column")
                return False

            latest_bar_time = pd.Timestamp(df['timestamp'].iloc[-2])
            if latest_bar_time.tzinfo is None:
                latest_bar_time = latest_bar_time.tz_localize(pytz.UTC)

            for i in range(-2, -required_bars - 1, -1):
                score = self.calculate_confluence_score(df.iloc[i], symbol=df['symbol'].iloc[0])
                threshold = self.thresholds.get(timeframe, 1)
                if score < threshold:
                    self.logger.debug(f"Confluence score below threshold for {timeframe}: {score} < {threshold}")
                    return False

            earliest_bar_time = pd.Timestamp(df['timestamp'].iloc[-required_bars - 1])
            if earliest_bar_time.tzinfo is None:
                earliest_bar_time = earliest_bar_time.tz_localize(pytz.UTC)

            duration = (current_time - earliest_bar_time).total_seconds()
            min_required_duration = min_duration.total_seconds()
            if duration < min_required_duration:
                self.logger.debug(f"Confluence duration too short for {timeframe}: {duration} < {min_required_duration}")
                return False

            # Enhanced: Hurst check with caching
            symbol = df['symbol'].iloc[0]
            hurst_values = self.compute_hurst(symbol)
            if hurst_values and all(h > 0.5 for h in hurst_values.values()):
                self.logger.debug(f"Hurst confluence passed for {timeframe}: {hurst_values}")
            else:
                self.logger.debug(f"Hurst confluence failed for {timeframe}: {hurst_values}")
                return False

            self.logger.debug(f"Confluence persists for {timeframe}, duration={duration}")
            return True
        except Exception as e:
            self.logger.error(f"Error checking confluence persistence: {str(e)}")
            return False

    def detect_williams_fractal(self, data: pd.DataFrame, direction: str = 'high') -> List[int]:
        """Detect Williams' fractal patterns (5-bar structure)."""
        order = 2  # Two bars on either side
        if direction == 'high':
            return list(argrelextrema(data['high'].values, np.greater, order=order)[0])
        return list(argrelextrema(data['low'].values, np.less, order=order)[0])

    def calculate_alligator_mas(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Alligator Moving Averages (Jaw, Teeth, Lips)."""
        return {
            'jaw': trend.SMAIndicator(data['close'], window=13).sma_indicator().shift(8),
            'teeth': trend.SMAIndicator(data['close'], window=8).sma_indicator().shift(5),
            'lips': trend.SMAIndicator(data['close'], window=5).sma_indicator().shift(3)
        }

    def _check_break_of_structure(self, symbol: str) -> bool:
        """
        Check for a break of market structure, enhanced with fractals and Alligator.

        Args:
            symbol (str): Symbol to check

        Returns:
            bool: True if break of structure detected, False otherwise
        """
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty or len(df) < 20 or 'high' not in df.columns:
                return False

            current_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]) else 0
            avg_atr = df['avg_atr'].iloc[-1] if 'avg_atr' in df.columns and not pd.isna(df['avg_atr'].iloc[-1]) else 0

            lookback = 20 if current_atr == 0 or avg_atr == 0 else (5 if current_atr > 1.5 * avg_atr else 20)
            recent_highs = df['high'].iloc[-lookback:]
            basic_bos = df['high'].iloc[-1] > recent_highs[:-1].max()

            # Enhanced: Fractal and Alligator confirmation
            fractal_highs = self.detect_williams_fractal(df)
            if fractal_highs and fractal_highs[-1] < len(df) - 1:
                fractal_bos = df['high'].iloc[-1] > df['high'].iloc[fractal_highs[-1]]
            else:
                fractal_bos = False

            alligator = self.calculate_alligator_mas(df)
            price_above_jaw = df['close'].iloc[-1] > alligator['jaw'].iloc[-1]

            return basic_bos and (fractal_bos or price_above_jaw)
        except Exception as e:
            self.logger.error(f"Error checking break of structure for {symbol}: {str(e)}")
            return False

    def _check_change_of_character(self, symbol: str) -> bool:
        """
        Check for a change of market character, enhanced with fractals and Alligator.

        Args:
            symbol (str): Symbol to check

        Returns:
            bool: True if change of character detected, False otherwise
        """
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty or len(df) < 20 or 'low' not in df.columns:
                return False

            current_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]) else 0
            avg_atr = df['avg_atr'].iloc[-1] if 'avg_atr' in df.columns and not pd.isna(df['avg_atr'].iloc[-1]) else 0

            lookback = 20 if current_atr == 0 or avg_atr == 0 else (5 if current_atr > 1.5 * avg_atr else 20)
            recent_lows = df['low'].iloc[-lookback:]
            basic_choch = df['low'].iloc[-1] < recent_lows[:-1].min()

            # Enhanced: Fractal and Alligator confirmation
            fractal_lows = self.detect_williams_fractal(df, direction='low')
            if fractal_lows and fractal_lows[-1] < len(df) - 1:
                fractal_choch = df['low'].iloc[-1] < df['low'].iloc[fractal_lows[-1]] - current_atr * 0.5
            else:
                fractal_choch = False

            alligator = self.calculate_alligator_mas(df)
            price_below_lips = df['close'].iloc[-1] < alligator['lips'].iloc[-1]

            return basic_choch and (fractal_choch or price_below_lips)
        except Exception as e:
            self.logger.error(f"Error checking change of character for {symbol}: {str(e)}")
            return False

    def check_bid_ask_imbalance(self, symbol: str) -> bool:
        """
        Check for bid-ask imbalance with dynamic threshold.

        Args:
            symbol (str): Symbol to check

        Returns:
            bool: True if imbalance detected, False otherwise
        """
        try:
            rtd = self.real_time_data.get(symbol)
            if not rtd:
                return False

            imbalance = rtd['bid_ask_imbalance']
            delta_history = list(rtd['delta_history'])
            if len(delta_history) < 20:
                return False

            rolling_avg = np.mean([abs(d) for d in delta_history[-20:]])
            dynamic_threshold = max(self.config.get('bid_ask_imbalance_threshold', 0.1), rolling_avg * 0.1)
            return abs(imbalance) > dynamic_threshold
        except Exception as e:
            self.logger.error(f"Error checking bid-ask imbalance for {symbol}: {str(e)}")
            return False

    def check_delta_divergence(self, symbol: str) -> bool:
        """
        Check for delta divergence with dynamic threshold.

        Args:
            symbol (str): Symbol to check

        Returns:
            bool: True if delta divergence detected, False otherwise
        """
        try:
            rtd = self.real_time_data.get(symbol)
            if not rtd or not rtd['delta_history']:
                return False

            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty:
                return False

            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
            threshold = max(self.config.get('delta_divergence_threshold', 500), atr * 1000)
            recent_delta = sum(rtd['delta_history'][-min(100, len(rtd['delta_history'])):])
            price_change = df['close'].iloc[-1] - df['close'].iloc[-5] if len(df) >= 5 else 0

            return abs(recent_delta) > threshold and abs(price_change) < atr * 0.5
        except Exception as e:
            self.logger.error(f"Error checking delta divergence for {symbol}: {str(e)}")
            return False

    def check_liquidity_zone(self, symbol: str) -> bool:
        """
        Check if price is in a liquidity zone, enhanced with price stagnation.

        Args:
            symbol (str): Symbol to check

        Returns:
            bool: True if in liquidity zone, False otherwise
        """
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty or len(df) < 20 or 'volume' not in df.columns:
                return False

            current_vol = df['volume'].iloc[-1]
            avg_vol = df['volume'].rolling(window=min(20, len(df))).mean().iloc[-1]
            if avg_vol == 0:
                return False

            multiplier = self.config.get('liquidity_zone_volume_multiplier', 1.5)
            volume_condition = current_vol > multiplier * avg_vol

            # Enhanced: Price stagnation and VWAP deviation
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
            price_stagnation = abs(df['close'].iloc[-1] - df['close'].iloc[-5]) < atr * 0.5 if len(df) >= 5 else False
            vwap_deviation = abs(df['close'].iloc[-1] - df['vwap'].iloc[-1]) < df['vwap_std'].iloc[-1] if 'vwap_std' in df.columns else True

            return volume_condition and price_stagnation and vwap_deviation
        except Exception as e:
            self.logger.error(f"Error checking liquidity zone for {symbol}: {str(e)}")
            return False

    def check_effort_vs_result(self, symbol: str) -> bool:
        """Filter signals where high volume doesn't lead to price movement."""
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty or len(df) < 5:
                return True  # Default to True if insufficient data

            effort = df['volume'].iloc[-1] > df['volume'].rolling(window=20).mean().iloc[-1] * 1.5
            result = abs(df['close'].iloc[-1] - df['close'].iloc[-5]) < df['atr'].iloc[-1]
            return not (effort and result)
        except Exception as e:
            self.logger.error(f"Error checking effort vs result for {symbol}: {str(e)}")
            return True

    def _check_multi_tf_trend_alignment(self, symbol: str, trading_tf: str) -> bool:
        """
        Check if the trend is aligned across multiple timeframes, enhanced with VWAP.

        Args:
            symbol (str): Symbol to check
            trading_tf (str): Timeframe for which the signal is being generated

        Returns:
            bool: True if trend is aligned, False otherwise
        """
        try:
            alignment_tfs = ['1min', '5min', '15min'] if trading_tf == '1min' else ['5min', '15min', '30min']
            if trading_tf not in alignment_tfs:
                alignment_tfs.append(trading_tf)

            trend_direction = None
            for tf in alignment_tfs:
                df = self.indicator_histories.get(symbol, {}).get(tf, pd.DataFrame())
                if df.empty:
                    return False
                trend = self.check_trend(tf, symbol)
                if trend is None:
                    return False
                if trend_direction is None:
                    trend_direction = trend
                elif trend != trend_direction:
                    return False

                # Enhanced: VWAP slope alignment
                if 'vwap_slope' in df.columns:
                    vwap_slope = df['vwap_slope'].iloc[-1]
                    if (trend_direction == "uptrend" and vwap_slope <= 0) or (trend_direction == "downtrend" and vwap_slope >= 0):
                        return False

            return True
        except Exception as e:
            self.logger.error(f"Error checking multi-timeframe trend alignment for {symbol}: {str(e)}")
            return False

    def check_supply_demand_zone(self, symbol: str) -> bool:
        """Detect supply/demand zones based on volume clusters and price rejection."""
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty or len(df) < 50:
                return False

            volume_cluster = df['volume'].iloc[-1] > np.percentile(df['volume'].iloc[-50:], 90)
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
            rejection = abs(df['close'].iloc[-1] - df['high'].iloc[-1]) < atr * 0.2 or abs(df['close'].iloc[-1] - df['low'].iloc[-1]) < atr * 0.2
            return volume_cluster and rejection
        except Exception as e:
            self.logger.error(f"Error checking supply/demand zone for {symbol}: {str(e)}")
            return False

    def detect_swing_points(self, symbol: str, timeframe: str = '1h', direction: str = 'high') -> List[int]:
        """Detect swing points with ATR-adjusted window and volume confirmation."""
        try:
            df = self.indicator_histories.get(symbol, {}).get(timeframe, pd.DataFrame())
            if df.empty or len(df) < 20:
                return []

            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
            window = max(5, min(20, int(atr / df['close'].iloc[-1] * 100)))
            if direction == 'high':
                highs = list(argrelextrema(df['high'].values, np.greater, order=window)[0])
                return [h for h in highs if df['volume'].iloc[h] > df['volume'].mean()]
            else:
                lows = list(argrelextrema(df['low'].values, np.less, order=window)[0])
                return [l for l in lows if df['volume'].iloc[l] > df['volume'].mean()]
        except Exception as e:
            self.logger.error(f"Error detecting swing points for {symbol}: {str(e)}")
            return []

    def update_parameters(self, symbol: str) -> None:
        """Dynamically adjust ATR period and thresholds based on volatility."""
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty:
                return

            volatility = df['close'].pct_change().std()
            self.atr_period = max(10, min(20, int(14 * (1 + volatility * 10))))
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            self.rsi_threshold = 70 if rsi > 50 else 30
            self.logger.debug(f"Updated parameters for {symbol}: ATR Period={self.atr_period}, RSI Threshold={self.rsi_threshold}")
        except Exception as e:
            self.logger.error(f"Error updating parameters for {symbol}: {str(e)}")

    def estimate_slippage(self, symbol: str, order_size: float = 1.0) -> float:
        """Estimate slippage based on order size and market depth."""
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty:
                return 0.0

            avg_volume = df['volume'].iloc[-20:].mean()
            impact = order_size / avg_volume * 0.01 if avg_volume > 0 else 0
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
            return atr * impact
        except Exception as e:
            self.logger.error(f"Error estimating slippage for {symbol}: {str(e)}")
            return 0.0

    def calculate_confluence_score(self, bar: pd.Series, symbol: str) -> int:
        """
        Calculate signal confluence score for a bar, enhanced with new checks.

        Args:
            bar (pd.Series): Bar data with indicators
            symbol (str): Symbol for real-time checks

        Returns:
            int: Confluence score
        """
        try:
            score = 0
            weights = self.config.get('confluence_weights', {
                'hurst_exponent': 3,
                'adaptive_moving_average': 2,
                'bos': 2,
                'choch': 1,
                'delta_divergence': 2,
                'bid_ask_imbalance': 2,
                'cumulative_delta': 3,
                'tick_filter': 1,
                'price_near_vwap': 2,
                'liquidity_zone': 2,
                'zscore_deviation': 2,
                'vwap_slope': 1,
                'supply_demand_zone': 2
            })

            required_columns = ['close', 'adaptive_ma', 'sma_slope', 'vwap_slope']
            for col in required_columns:
                if col not in bar or pd.isna(bar[col]):
                    self.logger.warning(f"Missing or NaN required column for confluence score: {col}")
                    return 0

            signal_direction = "uptrend" if bar['close'] > bar['adaptive_ma'] else "downtrend"
            sma_slope = bar['sma_slope']
            vwap_slope = bar['vwap_slope']

            if signal_direction == "uptrend" and sma_slope > 0.00002:
                score += weights.get('adaptive_moving_average', 2)
            elif signal_direction == "downtrend" and sma_slope < -0.00002:
                score += weights.get('adaptive_moving_average', 2)

            if not pd.isna(vwap_slope):
                if vwap_slope > 0 and signal_direction == "uptrend":
                    score += weights.get('vwap_slope', 1)
                elif vwap_slope < 0 and signal_direction == "downtrend":
                    score += weights.get('vwap_slope', 1)

            if self._check_break_of_structure(symbol):
                score += weights.get('bos', 2)
            if self._check_change_of_character(symbol):
                score += weights.get('choch', 1)
            if self.check_delta_divergence(symbol):
                score += weights.get('delta_divergence', 2)
            if self.check_bid_ask_imbalance(symbol):
                score += weights.get('bid_ask_imbalance', 2)
            if self.check_liquidity_zone(symbol):
                score += weights.get('liquidity_zone', 2)
            if self._check_price_near_vwap(symbol):
                score += weights.get('price_near_vwap', 2)
            if self._check_zscore_deviation(symbol):
                score += weights.get('zscore_deviation', 2)
            if self.check_supply_demand_zone(symbol):
                score += weights.get('supply_demand_zone', 2)
            if self.check_tick_filter(symbol):
                score += weights.get('tick_filter', 1)

            self.logger.debug(f"Confluence score for {symbol}: {score}")
            return score
        except Exception as e:
            self.logger.error(f"Error calculating confluence score: {str(e)}")
            return 0

    def generate_signal_for_tf(self, trading_tf: str, symbol: str, current_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal for a specific timeframe, enhanced with new checks.

        Args:
            trading_tf (str): Timeframe to generate signal for
            symbol (str): Symbol to generate signal for
            current_time (datetime): Current tick time from MT5

        Returns:
            Optional[Dict[str, Any]]: Generated signal if applicable, None otherwise
        """
        try:
            rtd = self.real_time_data.get(symbol)
            if not rtd or rtd['last_price'] is None:
                self.logger.debug(f"No real-time data for {symbol}")
                return None

            df = self.indicator_histories.get(symbol, {}).get(trading_tf)
            if df is None or df.empty or len(df) < 14:
                self.logger.debug(f"Insufficient data for {symbol} on {trading_tf}: {len(df) if df is not None else 'None'} bars")
                return None

            # Enhanced: Dynamic parameter updates
            self.update_parameters(symbol)

            if not self._check_multi_tf_trend_alignment(symbol, trading_tf):
                self.logger.debug(f"Multi-timeframe trend alignment failed for {symbol} on {trading_tf}")
                return None

            if not self.check_confluence_persistence(df, trading_tf, current_time):
                self.logger.debug(f"Confluence persistence failed for {symbol} on {trading_tf}")
                return None

            # Enhanced: Effort vs. result check
            if not self.check_effort_vs_result(symbol):
                self.logger.debug(f"Effort vs. result check failed for {symbol}")
                return None

            signal_direction = "uptrend" if df['close'].iloc[-1] > df['adaptive_ma'].iloc[-1] else "downtrend"
            score = self.calculate_confluence_score(df.iloc[-1], symbol)
            threshold = self.thresholds.get(trading_tf, 1)
            threshold_source = "config" if trading_tf in self.thresholds else "default"
            self.logger.debug(f"Signal check for {symbol} on {trading_tf}: score={score}, threshold={threshold} (from {threshold_source}), direction={signal_direction}")
            if score >= threshold:
                entry_price = df['close'].iloc[-1]
                slippage = self.estimate_slippage(symbol)
                signal = {
                    'action': 'buy' if signal_direction == "uptrend" else 'sell',
                    'timeframe': trading_tf,
                    'score': float(score),
                    'timestamp': current_time.isoformat(),
                    'entry_price': float(entry_price + slippage if signal_direction == "uptrend" else entry_price - slippage),
                    'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]) else 0.001,
                    'symbol': symbol
                }
                self.logger.info(f"Generated signal for {symbol} on {trading_tf}: {signal}")
                return signal
            return None
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol} on {trading_tf}: {str(e)}")
            return None

    def calculate_ad_line(self, df: pd.DataFrame, bars: int) -> List[float]:
        """
        Calculate Accumulation/Distribution Line.

        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            bars (int): Number of bars to include

        Returns:
            List[float]: A/D line values
        """
        try:
            ad = [0] * len(df)
            for i in range(1, len(df)):
                if i < bars:
                    ad[i] = ad[i - 1]
                else:
                    high = df['high'].iloc[i]
                    low = df['low'].iloc[i]
                    close = df['close'].iloc[i]
                    volume = df['volume'].iloc[i]
                    if high == low:
                        mf = 0
                    else:
                        mf = ((close - low) - (high - close)) / (high - low)
                    ad[i] = ad[i - 1] + mf * volume
            return ad
        except Exception as e:
            self.logger.error(f"Error calculating A/D line: {str(e)}")
            return [0] * len(df)

    def calculate_volume_delta_oscillator(self, symbol: str, tf: str, window: int = 20) -> float:
        """
        Calculate Volume Delta Oscillator.

        Args:
            symbol (str): Symbol to calculate for
            tf (str): Timeframe string
            window (int): Lookback window

        Returns:
            float: Oscillator value
        """
        try:
            df = self.indicator_histories.get(symbol, {}).get(tf, pd.DataFrame())
            if df.empty or len(df) < window:
                return 0

            delta = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']
            actual_window = min(window, len(df))
            cumulative_delta = delta.rolling(window=actual_window).sum()
            total_volume = df['volume'].rolling(window=actual_window).sum()

            if total_volume.iloc[-1] > 0:
                oscillator = cumulative_delta / total_volume
                return float(oscillator.iloc[-1])
            return 0
        except Exception as e:
            self.logger.error(f"Error calculating volume delta oscillator: {str(e)}")
            return 0

    def check_trend(self, timeframe: str, symbol: str) -> Optional[str]:
        """
        Check the current trend for a symbol on a timeframe.

        Args:
            timeframe (str): Timeframe to check
            symbol (str): Symbol to check

        Returns:
            Optional[str]: "uptrend", "downtrend", "sideways", or None if data unavailable
        """
        try:
            df = self.indicator_histories.get(symbol, {}).get(timeframe, pd.DataFrame())
            if df.empty or 'close' not in df.columns or 'adaptive_ma' not in df.columns:
                return None

            last_close = df['close'].iloc[-1]
            last_ma = df['adaptive_ma'].iloc[-1]

            if last_close > last_ma * 1.005:
                return "uptrend"
            elif last_close < last_ma * 0.995:
                return "downtrend"
            else:
                return "sideways"
        except Exception as e:
            self.logger.error(f"Error checking trend for {symbol} on {timeframe}: {str(e)}")
            return None

    def check_tick_filter(self, symbol: str) -> bool:
        """
        Check tick filter for trading signal.

        Args:
            symbol (str): Symbol to check

        Returns:
            bool: True if tick filter triggered, False otherwise
        """
        try:
            tick_threshold = self.config.get('tick_threshold', 800)

            rtd = self.real_time_data.get(symbol)
            if not rtd:
                return False

            return rtd['cumulative_tick'] > tick_threshold
        except Exception as e:
            self.logger.error(f"Error checking tick filter for {symbol}: {str(e)}")
            return False

    def _check_price_near_vwap(self, symbol: str) -> bool:
        """
        Check if price is near VWAP.

        Args:
            symbol (str): Symbol to check

        Returns:
            bool: True if price is near VWAP, False otherwise
        """
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty or 'close' not in df.columns or 'vwap' not in df.columns:
                return False

            close = df['close'].iloc[-1]
            vwap = df['vwap'].iloc[-1]

            if vwap == 0:
                return False

            return abs(close - vwap) / vwap < 0.005
        except Exception as e:
            self.logger.error(f"Error checking price near VWAP for {symbol}: {str(e)}")
            return False

    def _check_zscore_deviation(self, symbol: str) -> bool:
        """
        Check for significant Z-score deviation.

        Args:
            symbol (str): Symbol to check

        Returns:
            bool: True if significant deviation detected, False otherwise
        """
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h', pd.DataFrame())
            if df.empty or 'zscore' not in df.columns:
                return False

            zscore = df['zscore'].iloc[-1]
            return abs(zscore) > 2
        except Exception as e:
            self.logger.error(f"Error checking Z-score deviation for {symbol}: {str(e)}")
            return False