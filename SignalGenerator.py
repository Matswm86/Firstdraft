import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import numpy as np
from hurst import compute_Hc
import ta  # Technical Analysis library for indicator calculations


class SignalGenerator:
    def __init__(self, config):
        """
        Initialize the SignalGenerator with configuration parameters.

        Args:
            config (dict): Configuration including timeframes, thresholds, and max bars.
        """
        self.trading_timeframes = config['trading_timeframes']  # e.g., ['1min', '5min']
        self.thresholds = config['thresholds']  # e.g., {'1min': 10, '5min': 8}
        self.timeframes = config['timeframes']  # e.g., ['1min', '5min', '15min', '1h', '4h', 'daily']
        self.max_bars = config['max_bars']  # e.g., {'1min': 10000, '5min': 5000, ..., 'daily': 16}

        # History storage: deque of bar dictionaries
        self.histories = {tf: deque(maxlen=self.max_bars[tf]) for tf in self.timeframes}
        # Indicator histories: DataFrames with precomputed indicators
        self.indicator_histories = {tf: pd.DataFrame() for tf in self.timeframes}
        # Current bars for live aggregation
        self.current_bars = {tf: None for tf in self.timeframes}

        # Real-time data
        self.real_time_data = {
            'cumulative_delta': 0,
            'delta_history': deque(maxlen=1000),
            'bid_ask_imbalance': 0,
            'composite_breadth_score': 0,
            'tick': 0,
            'last_price': None,
            'last_volume': 0
        }

        # Timeframe intervals
        self.timeframe_intervals = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            'daily': timedelta(days=1)
        }

    def process_tick(self, tick):
        """
        Process an incoming tick, aggregate into bars, and generate signals if applicable.

        Args:
            tick (dict): Tick data with 'timestamp', 'price', 'volume', 'bid', 'ask', etc.
        """
        tick_time = pd.to_datetime(tick['timestamp'])
        price = tick['price']
        volume = tick.get('volume', 0)

        # Update real-time metrics
        self.update_real_time_data(tick)

        # Aggregate tick into bars for each timeframe
        for tf in self.timeframes:
            bar = self._aggregate_tick(tf, tick_time, price, volume)
            if bar:
                self.add_bar(tf, bar)
                if tf in self.trading_timeframes:
                    self.generate_signal_for_tf(tf)

    def _aggregate_tick(self, timeframe, tick_time, price, volume):
        """
        Aggregate tick data into a bar for the given timeframe.

        Returns:
            dict: Completed bar if finalized, None otherwise.
        """
        interval = self.timeframe_intervals[timeframe]
        if self.current_bars[timeframe] is None:
            self.current_bars[timeframe] = {
                'timestamp': tick_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
            return None
        else:
            current_bar = self.current_bars[timeframe]
            bar_end_time = current_bar['timestamp'] + interval
            if tick_time < bar_end_time:
                current_bar['high'] = max(current_bar['high'], price)
                current_bar['low'] = min(current_bar['low'], price)
                current_bar['close'] = price
                current_bar['volume'] += volume
                return None
            else:
                completed_bar = current_bar.copy()
                self.current_bars[timeframe] = {
                    'timestamp': tick_time,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                }
                return completed_bar

    def add_bar(self, timeframe, bar):
        """
        Add a completed bar to the history and update indicators.

        Args:
            timeframe (str): Timeframe of the bar (e.g., '1min').
            bar (dict): Bar data with OHLCV and timestamp.
        """
        self.histories[timeframe].append(bar)
        # Update indicator history if enough data
        df = pd.DataFrame(list(self.histories[timeframe]))
        if len(df) >= 50:  # Minimum bars for meaningful indicators
            df = self.calculate_indicators(df, timeframe)
            self.indicator_histories[timeframe] = df

    def update_real_time_data(self, tick):
        """
        Update real-time metrics with each tick.

        Args:
            tick (dict): Tick data with 'price', 'volume', 'bid', 'ask', etc.
        """
        price = tick['price']
        volume = tick.get('volume', 0)
        bid = tick.get('bid', price)
        ask = tick.get('ask', price)

        # Cumulative Delta (simplified: assumes volume direction based on price movement)
        if self.real_time_data['last_price'] is not None:
            price_change = price - self.real_time_data['last_price']
            delta = volume if price_change > 0 else -volume if price_change < 0 else 0
            self.real_time_data['cumulative_delta'] += delta
            self.real_time_data['delta_history'].append(delta)

        # Bid-Ask Imbalance
        if bid != ask:
            self.real_time_data['bid_ask_imbalance'] = (bid - ask) / (bid + ask)

        # TICK (simplified: price movement direction)
        self.real_time_data['tick'] = 1 if price > self.real_time_data['last_price'] else -1 if price < \
                                                                                                self.real_time_data[
                                                                                                    'last_price'] else 0

        # Composite Breadth Score (simplified: running average of tick)
        self.real_time_data['composite_breadth_score'] = (self.real_time_data['composite_breadth_score'] * 0.9) + (
                    self.real_time_data['tick'] * 0.1)

        # Update last price and volume
        self.real_time_data['last_price'] = price
        self.real_time_data['last_volume'] = volume

    def calculate_indicators(self, df, timeframe):
        """
        Compute technical indicators for the given DataFrame.

        Args:
            df (pd.DataFrame): Bar data with OHLCV.
            timeframe (str): Timeframe of the data.

        Returns:
            pd.DataFrame: Updated DataFrame with indicators.
        """
        # ATR
        df['atr'] = ta.volatility.atr(df['high'], df['low'], df['close'], window=14)

        # Adaptive Moving Average (based on ATR)
        lookback = max(10, min(50, int(df['atr'].iloc[-1] * 10) if not pd.isna(df['atr'].iloc[-1]) else 20))
        df['adaptive_ma'] = df['close'].rolling(window=lookback).mean()

        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # VWAP (simplified daily reset)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_slope'] = df['vwap'].diff(periods=5) / 5

        # Z-Score
        df['zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()

        # Hurst Exponent (for daily timeframe)
        if timeframe == 'daily' and len(df) > 100:
            H, _, _ = compute_Hc(df['close'], kind='price')
            df['hurst'] = H

        return df

    def generate_signal_for_tf(self, trading_tf):
        """
        Generate a trading signal for the given timeframe based on confluence checks.

        Args:
            trading_tf (str): Trading timeframe (e.g., '1min').
        """
        if not self.indicator_histories[trading_tf].empty:
            score = 0

            # Perform confluence checks
            if self.check_hurst_daily():
                score += 3
            if self.check_adaptive_ma_4h():
                score += 2
            if self.check_bos():
                score += 2
            if self.check_choch():
                score += 2
            if self.check_cumulative_delta():
                score += 1
            if self.check_bid_ask_imbalance():
                score += 1
            if self.check_composite_breadth():
                score += 1
            if self.check_tick_filter():
                score += 1
            if self.check_price_near_vwap():
                score += 1
            if self.check_zscore_deviation():
                score += 1
            if self.check_vwap_slope():
                score += 1
            if self.check_proximity_liquidity():
                score += 1
            if self.check_reversal_patterns():
                score += 1

            # Generate signal if threshold met
            if score >= self.thresholds[trading_tf]:
                signal = {
                    'action': 'buy',  # Simplified; could be 'sell' based on additional logic
                    'timeframe': trading_tf,
                    'score': score,
                    'timestamp': datetime.now()
                }
                self.send_signal(signal)

    # Confluence Check Methods
    def check_hurst_daily(self):
        df = self.indicator_histories['daily']
        return 'hurst' in df.columns and not df.empty and df['hurst'].iloc[-1] > 0.5

    def check_adaptive_ma_4h(self):
        df = self.indicator_histories['4h']
        return not df.empty and df['close'].iloc[-1] > df['adaptive_ma'].iloc[-1]

    def check_bos(self):
        df = self.indicator_histories['1h']  # Example timeframe
        return not df.empty and df['high'].iloc[-1] > df['high'].iloc[-2]  # Break of Structure

    def check_choch(self):
        df = self.indicator_histories['1h']  # Example timeframe
        return not df.empty and df['low'].iloc[-1] < df['low'].iloc[-2]  # Change of Character

    def check_cumulative_delta(self):
        return self.real_time_data['cumulative_delta'] > 0  # Positive momentum

    def check_bid_ask_imbalance(self):
        return abs(self.real_time_data['bid_ask_imbalance']) > 0.1  # Significant imbalance

    def check_composite_breadth(self):
        return self.real_time_data['composite_breadth_score'] > 0.5  # Bullish breadth

    def check_tick_filter(self):
        return self.real_time_data['tick'] > 0  # Positive tick

    def check_price_near_vwap(self):
        df = self.indicator_histories['1h']  # Example timeframe
        return not df.empty and abs(df['close'].iloc[-1] - df['vwap'].iloc[-1]) / df['vwap'].iloc[-1] < 0.01

    def check_zscore_deviation(self):
        df = self.indicator_histories['1h']  # Example timeframe
        return not df.empty and abs(df['zscore'].iloc[-1]) > 2  # Significant deviation

    def check_vwap_slope(self):
        df = self.indicator_histories['1h']  # Example timeframe
        return not df.empty and df['vwap_slope'].iloc[-1] > 0  # Upward slope

    def check_proximity_liquidity(self):
        df = self.indicator_histories['1h']  # Example timeframe
        return not df.empty and df['volume'].iloc[-1] > df['volume'].rolling(window=20).mean().iloc[-1]

    def check_reversal_patterns(self):
        df = self.indicator_histories['1h']  # Example timeframe
        return not df.empty and df['rsi'].iloc[-1] < 30  # Oversold condition

    def load_historical_data(self, historical_data):
        """
        Load historical bar data into the histories for backtesting.

        Args:
            historical_data (dict): Dictionary of DataFrames keyed by timeframe.
        """
        for tf in self.timeframes:
            if tf in historical_data:
                bars = historical_data[tf].to_dict('records')
                self.histories[tf].extend(bars[-self.max_bars[tf]:])
                # Compute indicators
                df = pd.DataFrame(list(self.histories[tf]))
                if len(df) >= 50:
                    df = self.calculate_indicators(df, tf)
                    self.indicator_histories[tf] = df

    def send_signal(self, signal):
        """
        Send the generated signal (placeholder for actual implementation).

        Args:
            signal (dict): Signal data with action, timeframe, score, and timestamp.
        """
        print(f"Signal Generated: {signal}")  # Replace with actual signal sending logic