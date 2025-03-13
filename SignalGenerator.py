import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import pytz
from hurst import compute_Hc
import ta
import logging
import MetaTrader5 as mt5
import math

class SignalGenerator:
    def __init__(self, config, mt5_api=None):
        self.config = config
        self.trading_timeframes = config['signal_generation']['trading_timeframes']
        self.thresholds = config['signal_generation']['thresholds']
        self.timeframes = config['signal_generation']['timeframes']
        self.max_bars = config['signal_generation']['max_bars']
        self.symbols = config['symbols']
        self.mt5_api = mt5_api
        self.logger = logging.getLogger(__name__)

        self.histories = {symbol: {tf: deque(maxlen=self.max_bars[tf]) for tf in self.timeframes}
                          for symbol in self.symbols}
        self.indicator_histories = {symbol: {tf: pd.DataFrame() for tf in self.timeframes}
                                    for symbol in self.symbols}
        self.current_bars = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}
        self.last_signals = {symbol: None for symbol in self.symbols}

        self.real_time_data = {symbol: {
            'cumulative_delta': 0, 'delta_history': deque(maxlen=1000), 'bid_ask_imbalance': 0,
            'composite_breadth_score': 0, 'tick': 0, 'cumulative_tick': 0, 'last_price': None, 'last_volume': 0
        } for symbol in self.symbols}

        self.timeframe_intervals = {'1min': timedelta(minutes=1), '5min': timedelta(minutes=5),
                                    '15min': timedelta(minutes=15), '30min': timedelta(minutes=30),
                                    '1h': timedelta(hours=1), '4h': timedelta(hours=4), 'daily': timedelta(days=1)}

        self.freq_map = {'1min': 'min', '5min': '5min', '15min': '15min', '30min': '30min',
                         '1h': 'h', '4h': '4h', 'daily': 'D'}

        # Swing tracking for BOS and CHOCH
        self.swing_highs = {symbol: {tf: deque(maxlen=10) for tf in self.timeframes} for symbol in self.symbols}
        self.swing_lows = {symbol: {tf: deque(maxlen=10) for tf in self.timeframes} for symbol in self.symbols}
        self.last_bos = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}

        if self.mt5_api:
            self._load_initial_history()

    def _load_initial_history(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                mt5_tf = {'1min': mt5.TIMEFRAME_M1, '5min': mt5.TIMEFRAME_M5, '15min': mt5.TIMEFRAME_M15,
                          '30min': mt5.TIMEFRAME_M30, '1h': mt5.TIMEFRAME_H1, '4h': mt5.TIMEFRAME_H4,
                          'daily': mt5.TIMEFRAME_D1}[tf]
                rates = self.mt5_api.get_ohlc_data(symbol, mt5_tf, 100)
                if rates:
                    for rate in rates:
                        bar = {'timestamp': pd.to_datetime(rate['timestamp'], utc=True),
                               'open': rate['open'], 'high': rate['high'], 'low': rate['low'],
                               'close': rate['close'], 'volume': rate['volume']}
                        self.histories[symbol][tf].append(bar)
                    df = pd.DataFrame([dict(b) for b in self.histories[symbol][tf]])
                    if len(df) >= 14:
                        df = self.calculate_indicators(df, tf)
                        self.indicator_histories[symbol][tf] = df
                        self.logger.info(f"Loaded initial history for {symbol} on {tf}: {len(df)} bars")

    def preprocess_tick(self, raw_tick):
        try:
            price = raw_tick['last'] if raw_tick['last'] != 0.0 else (raw_tick['bid'] + raw_tick['ask']) / 2
            return {
                'timestamp': raw_tick['timestamp'],
                'price': price,
                'volume': raw_tick.get('volume', 0),
                'bid': raw_tick.get('bid', price),
                'ask': raw_tick.get('ask', price),
                'symbol': raw_tick['symbol']
            }
        except KeyError as e:
            self.logger.error(f"Invalid tick data: missing {e}")
            return None

    def process_tick(self, raw_tick):
        tick = self.preprocess_tick(raw_tick)
        if not tick:
            return None
        symbol = tick['symbol']
        try:
            tick_time = pd.to_datetime(tick['timestamp'], utc=True)
            price = tick['price']
            volume = tick.get('volume', 0)
        except Exception as e:
            self.logger.error(f"Error processing tick for {symbol}: {e}")
            return None

        self.update_real_time_data(tick)
        for tf in self.timeframes:
            bar = self._aggregate_tick(tf, tick_time, price, volume, symbol)
            if bar:
                self.add_bar(tf, bar, symbol)
            if tf in self.trading_timeframes:
                self.generate_signal_for_tf(tf, symbol)
        return self.last_signals[symbol]

    def _aggregate_tick(self, timeframe, tick_time, price, volume, symbol):
        freq = self.freq_map[timeframe]
        floored_time = tick_time.floor(freq=freq)
        current_bar = self.current_bars[symbol][timeframe]

        if current_bar is None or floored_time > current_bar['timestamp']:
            if current_bar:
                completed_bar = current_bar.copy()
                self.histories[symbol][timeframe].append(completed_bar)
                self.logger.debug(f"Completed bar for {symbol} on {timeframe}: {completed_bar}")
            else:
                completed_bar = None
            self.current_bars[symbol][timeframe] = {
                'timestamp': floored_time, 'open': price, 'high': price,
                'low': price, 'close': price, 'volume': volume
            }
            return completed_bar
        else:
            current_bar['high'] = max(current_bar['high'], price)
            current_bar['low'] = min(current_bar['low'], price)
            current_bar['close'] = price
            current_bar['volume'] += volume
            return None

    def add_bar(self, timeframe, bar, symbol):
        if bar:
            self.histories[symbol][timeframe].append(bar)
        df = pd.DataFrame([dict(b) for b in self.histories[symbol][timeframe]])
        if len(df) >= 14:
            df = self.calculate_indicators(df, timeframe)
            self.indicator_histories[symbol][timeframe] = df
            self.logger.debug(f"Updated indicator history for {symbol} on {timeframe}: {len(df)} bars")

    def update_real_time_data(self, tick):
        symbol = tick['symbol']
        price = tick['price']
        volume = tick.get('volume', 0)
        bid = tick.get('bid', price)
        ask = tick.get('ask', price)
        rtd = self.real_time_data[symbol]

        if rtd['last_price'] is None:
            rtd['last_price'] = price
            rtd['last_volume'] = volume
            rtd['tick'] = 0
        else:
            tick_value = 1 if price > rtd['last_price'] else -1 if price < rtd['last_price'] else 0
            rtd['tick'] = tick_value
            rtd['cumulative_tick'] += tick_value
            price_change = price - rtd['last_price']
            delta = volume if price_change > 0 else -volume if price_change < 0 else 0
            rtd['cumulative_delta'] += delta
            rtd['delta_history'].append(delta)

        if bid != ask:
            rtd['bid_ask_imbalance'] = (bid - ask) / (bid + ask)
        rtd['composite_breadth_score'] = rtd['composite_breadth_score'] * 0.9 + rtd['tick'] * 0.1
        rtd['last_price'] = price
        rtd['last_volume'] = volume

    def calculate_indicators(self, df, timeframe):
        """Calculate all indicators, including BOS, CHOCH, and Z-Score."""
        # Existing indicators
        atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'],
                                                       window=14, fillna=True)
        df['atr'] = atr_indicator.average_true_range()
        lookback = max(1, min(100, int(df['atr'].iloc[-1] * 10) if not pd.isna(df['atr'].iloc[-1]) else 30))
        df['adaptive_ma'] = df['close'].rolling(window=lookback).mean()
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        vwap_type = self.config.get('vwap_type', 'daily')
        if vwap_type == 'daily':
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_slope'] = df['vwap'].diff(periods=5) / 5

        # Z-Score calculation
        df['zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()

        # Additional indicators
        if timeframe == 'daily' and len(df) > 100:
            H, _, _ = compute_Hc(df['close'], kind='price')
            df['hurst'] = H
        df['sma_slope'] = (df['adaptive_ma'] - df['adaptive_ma'].shift(5)) / 5
        df['avg_atr'] = df['atr'].rolling(window=20).mean()
        df['vwap_std'] = (df['close'] - df['vwap']).rolling(window=20).std()
        df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
        df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
        df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
        df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']

        # BOS and CHOCH detection
        df = self.detect_bos_and_choch(df, timeframe, df['symbol'].iloc[0] if 'symbol' in df.columns else self.symbols[0])

        return df

    def detect_bos_and_choch(self, df, timeframe, symbol):
        """Separate logic for detecting BOS and CHOCH."""
        # Swing high/low detection (5-bar window, centered)
        df['swing_high'] = df['high'].rolling(window=5, center=True).max()
        df['swing_low'] = df['low'].rolling(window=5, center=True).min()

        # Initialize columns
        df['bos_detected'] = False
        df['bos_direction'] = None
        df['choch_detected'] = False

        for i in range(1, len(df)):
            # BOS Detection
            if df['close'].iloc[i] > df['swing_high'].iloc[i-1] and not pd.isna(df['swing_high'].iloc[i-1]):
                df.at[df.index[i], 'bos_detected'] = True
                df.at[df.index[i], 'bos_direction'] = 'uptrend'
                self.last_bos[symbol][timeframe] = {'index': i, 'direction': 'uptrend'}
            elif df['close'].iloc[i] < df['swing_low'].iloc[i-1] and not pd.isna(df['swing_low'].iloc[i-1]):
                df.at[df.index[i], 'bos_detected'] = True
                df.at[df.index[i], 'bos_direction'] = 'downtrend'
                self.last_bos[symbol][timeframe] = {'index': i, 'direction': 'downtrend'}

            # CHOCH Detection (no new high/low within 5 bars after BOS)
            if self.last_bos[symbol][timeframe]:
                bos_index = self.last_bos[symbol][timeframe]['index']
                bos_direction = self.last_bos[symbol][timeframe]['direction']
                if bos_direction == 'uptrend' and i > bos_index + 5:
                    if df['high'].iloc[bos_index:i].max() <= df['high'].iloc[bos_index]:
                        df.at[df.index[i], 'choch_detected'] = True
                elif bos_direction == 'downtrend' and i > bos_index + 5:
                    if df['low'].iloc[bos_index:i].min() >= df['low'].iloc[bos_index]:
                        df.at[df.index[i], 'choch_detected'] = True

        return df

    def get_required_bars(self, timeframe):
        tf_minutes = {'1min': 1, '5min': 5, '15min': 15, '30min': 30, '1h': 60, '4h': 240, 'daily': 1440}[timeframe]
        min_duration_minutes = 15
        required_bars = max(3, math.ceil(min_duration_minutes / tf_minutes))
        return required_bars

    def check_confluence_persistence(self, df, timeframe, current_time):
        required_bars = self.get_required_bars(timeframe)
        if len(df) < required_bars:
            self.logger.debug(f"Not enough completed bars for {timeframe}: {len(df)} < {required_bars}")
            return False

        for i in range(-1, -required_bars - 1, -1):
            score = self.calculate_confluence_score(df.iloc[i])
            if score < self.thresholds.get(timeframe, 5):
                self.logger.debug(f"Confluence not persistent at bar {i}: Score {score}")
                return False

        earliest_bar_time = df['timestamp'].iloc[-required_bars]
        min_duration = timedelta(minutes=15)
        if (current_time - earliest_bar_time) < min_duration:
            self.logger.debug(f"Confluence duration too short: {(current_time - earliest_bar_time).total_seconds() / 60:.2f} minutes")
            return False
        return True

    def calculate_confluence_score(self, bar):
        """Calculate confluence score including BOS, CHOCH, and Z-Score."""
        score = 0
        weights = {
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
            'vwap_slope': 1
        }

        signal_direction = "uptrend" if bar['close'] > bar['adaptive_ma'] else "downtrend"

        # Adaptive Moving Average
        if signal_direction == "uptrend" and bar['sma_slope'] > 0.0001:
            score += weights['adaptive_moving_average']
        elif signal_direction == "downtrend" and bar['sma_slope'] < -0.0001:
            score += weights['adaptive_moving_average']

        # VWAP Slope
        if signal_direction == "uptrend" and bar['vwap_slope'] > 0:
            score += weights['vwap_slope']
        elif signal_direction == "downtrend" and bar['vwap_slope'] < 0:
            score += weights['vwap_slope']

        # Hurst Exponent
        if 'hurst' in bar and bar['hurst'] > 0.5:
            score += weights['hurst_exponent']

        # BOS
        if 'bos_detected' in bar and bar['bos_detected'] and bar['bos_direction'] == signal_direction:
            score += weights['bos']

        # CHOCH
        if 'choch_detected' in bar and bar['choch_detected']:
            score += weights['choch']

        # Delta Divergence (example)
        if 'cumulative_delta' in bar:
            if (signal_direction == "uptrend" and bar['cumulative_delta'] < 0) or \
               (signal_direction == "downtrend" and bar['cumulative_delta'] > 0):
                score += weights['delta_divergence']

        # Bid-Ask Imbalance (example)
        if 'bid_ask_imbalance' in bar and abs(bar['bid_ask_imbalance']) > 500:
            score += weights['bid_ask_imbalance']

        # Cumulative Delta (example)
        if 'cumulative_delta' in bar and abs(bar['cumulative_delta']) > 1000:
            score += weights['cumulative_delta']

        # Tick Filter (example)
        if 'tick_volume' in bar and bar['tick_volume'] > 100:
            score += weights['tick_filter']

        # Price Near VWAP
        if 'vwap' in bar and abs(bar['close'] - bar['vwap']) < 0.01 * bar['vwap']:
            score += weights['price_near_vwap']

        # Liquidity Zone
        if 'vwap_upper_1' in bar and 'vwap_lower_1' in bar and \
           bar['vwap_lower_1'] < bar['close'] < bar['vwap_upper_1']:
            score += weights['liquidity_zone']

        # Z-Score Deviation
        if 'zscore' in bar and abs(bar['zscore']) > 2:
            score += weights['zscore_deviation']

        return score

    def generate_signal_for_tf(self, trading_tf, symbol):
        df = self.indicator_histories[symbol][trading_tf]
        if df.empty or len(df) < 14:
            self.logger.debug(f"Insufficient indicator history for {symbol} on {trading_tf}: {len(df)} bars")
            return

        current_time = datetime.now(pytz.UTC)
        if not self.check_confluence_persistence(df, trading_tf, current_time):
            self.logger.debug(f"Confluence not persistent for {symbol} on {trading_tf}")
            return

        signal_direction = "uptrend" if df['close'].iloc[-1] > df['adaptive_ma'].iloc[-1] else "downtrend"
        score = self.calculate_confluence_score(df.iloc[-1])
        threshold = self.thresholds.get(trading_tf, 5)

        if score >= threshold:
            signal = {
                'action': 'buy' if signal_direction == "uptrend" else 'sell',
                'timeframe': trading_tf, 'score': score, 'timestamp': current_time.isoformat(),
                'entry_price': df['close'].iloc[-1], 'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 0.001,
                'symbol': symbol
            }
            self.last_signals[symbol] = signal
            self.logger.info(f"Generated signal for {symbol} on {trading_tf}: {signal}")