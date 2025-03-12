import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from hurst import compute_Hc
import ta  # Technical Analysis library
import logging


class SignalGenerator:
    def __init__(self, config, trade_execution=None):
        """
        Initialize SignalGenerator for The 5%ers MT5 trading with EURUSD and GBPJPY.

        Args:
            config (dict): Configuration dictionary with signal generation settings.
            trade_execution (TradeExecution, optional): Instance for live trading (not used in backtest).
        """
        self.config = config
        self.trading_timeframes = config['signal_generation']['trading_timeframes']  # e.g., ["15min", "5min", "1min"]
        self.thresholds = config['signal_generation']['thresholds']
        self.timeframes = config['signal_generation']['timeframes']
        self.max_bars = config['signal_generation']['max_bars']
        self.symbols = config['symbols']  # e.g., ["EURUSD", "GBPJPY"]
        self.trade_execution = trade_execution
        self.logger = logging.getLogger(__name__)

        # Initialize data structures per symbol and timeframe
        self.histories = {symbol: {tf: deque(maxlen=self.max_bars[tf]) for tf in self.timeframes}
                          for symbol in self.symbols}
        self.indicator_histories = {symbol: {tf: pd.DataFrame() for tf in self.timeframes}
                                    for symbol in self.symbols}
        self.current_bars = {symbol: {tf: None for tf in self.timeframes} for symbol in self.symbols}
        self.last_signals = {symbol: None for symbol in self.symbols}

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

        # Timeframe intervals for aggregation
        self.timeframe_intervals = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            'daily': timedelta(days=1)
        }

    def preprocess_tick(self, raw_tick):
        """
        Preprocess raw tick data into a standardized format.

        Args:
            raw_tick (dict): Raw tick data from MT5API.

        Returns:
            dict or None: Processed tick data, or None if invalid.
        """
        try:
            return {
                'timestamp': raw_tick['timestamp'],
                'price': raw_tick['price'],
                'volume': raw_tick.get('volume', 0),
                'bid': raw_tick.get('bid', raw_tick['price']),
                'ask': raw_tick.get('ask', raw_tick['price']),
                'symbol': raw_tick['symbol']
            }
        except KeyError as e:
            self.logger.error(f"Invalid tick data: missing {e}")
            return None

    def process_tick(self, raw_tick):
        """
        Process a single tick to update histories and generate signals.

        Args:
            raw_tick (dict): Raw tick data from MT5API.

        Returns:
            dict or None: Generated signal, or None if no signal.
        """
        tick = self.preprocess_tick(raw_tick)
        if not tick:
            return None
        symbol = tick['symbol']
        try:
            tick_time = pd.to_datetime(tick['timestamp'])
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
        """
        Aggregate tick data into bars for a given timeframe.

        Args:
            timeframe (str): Timeframe (e.g., '15min').
            tick_time (datetime): Timestamp of the tick.
            price (float): Price of the tick.
            volume (float): Volume of the tick.
            symbol (str): Trading symbol.

        Returns:
            dict or None: Completed bar, or None if still aggregating.
        """
        interval = self.timeframe_intervals[timeframe]
        if self.current_bars[symbol][timeframe] is None:
            self.current_bars[symbol][timeframe] = {
                'timestamp': tick_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
            return None
        else:
            current_bar = self.current_bars[symbol][timeframe]
            bar_end_time = current_bar['timestamp'] + interval
            if tick_time < bar_end_time:
                current_bar['high'] = max(current_bar['high'], price)
                current_bar['low'] = min(current_bar['low'], price)
                current_bar['close'] = price
                current_bar['volume'] += volume
                return None
            else:
                completed_bar = current_bar.copy()
                self.current_bars[symbol][timeframe] = {
                    'timestamp': tick_time,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                }
                return completed_bar

    def add_bar(self, timeframe, bar, symbol):
        """
        Add a completed bar to the history and calculate indicators.

        Args:
            timeframe (str): Timeframe of the bar.
            bar (dict): Bar data with OHLC and volume.
            symbol (str): Trading symbol.
        """
        self.histories[symbol][timeframe].append(bar)
        df = pd.DataFrame([dict(b) for b in self.histories[symbol][timeframe]])
        if len(df) >= 50:
            df = self.calculate_indicators(df, timeframe)
            if self.config.get('ad_line_bars'):
                df['ad_line'] = self.calculate_ad_line(df, self.config['ad_line_bars'])
            self.indicator_histories[symbol][timeframe] = df

    def update_real_time_data(self, tick):
        """
        Update real-time data metrics for a symbol.

        Args:
            tick (dict): Processed tick data.
        """
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
        rtd['composite_breadth_score'] = (
                rtd['composite_breadth_score'] * 0.9 +
                rtd['tick'] * 0.1
        )
        rtd['last_price'] = price
        rtd['last_volume'] = volume

    def calculate_indicators(self, df, timeframe):
        """
        Calculate technical indicators for a DataFrame of bars.

        Args:
            df (pd.DataFrame): Bar data with OHLC and volume.
            timeframe (str): Timeframe of the data.

        Returns:
            pd.DataFrame: DataFrame with added indicators.
        """
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'],
            window=14, fillna=True
        )
        df['atr'] = atr_indicator.average_true_range()
        lookback = max(20, min(100, int(df['atr'].iloc[-1] * 10) if not pd.isna(df['atr'].iloc[-1]) else 30))
        df['adaptive_ma'] = df['close'].rolling(window=lookback).mean()
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        vwap_type = self.config.get('vwap_type', 'daily')
        if vwap_type == 'daily':
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_slope'] = df['vwap'].diff(periods=5) / 5
        df['zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        if timeframe == 'daily' and len(df) > 100:
            H, _, _ = compute_Hc(df['close'], kind='price')
            df['hurst'] = H
        return df

    def calculate_ad_line(self, df, bars):
        """
        Calculate the Advance/Decline line for a DataFrame.

        Args:
            df (pd.DataFrame): Bar data with OHLC and volume.
            bars (int): Number of bars for calculation.

        Returns:
            list: A/D line values.
        """
        ad = [0] * len(df)
        for i in range(1, len(df)):
            if i < bars:
                ad[i] = ad[i - 1]
            else:
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                mf = ((df['close'].iloc[i] - low) - (high - df['close'].iloc[i])) / (high - low) if high != low else 0
                ad[i] = ad[i - 1] + mf * df['volume'].iloc[i]
        return ad

    def generate_signal_for_tf(self, trading_tf, symbol):
        """
        Generate a trading signal for a specific timeframe and symbol.

        Args:
            trading_tf (str): Trading timeframe (e.g., '15min', '5min', '1min').
            symbol (str): Trading symbol (e.g., 'EURUSD', 'GBPJPY').
        """
        if self.indicator_histories[symbol][trading_tf].empty:
            return

        score = 0
        df = self.indicator_histories[symbol][trading_tf]
        weights = self.config.get('confluence_weights', {
            'hurst_exponent': 3,
            'adaptive_moving_average': 2,
            'bos': 2,
            'choch': 1,
            'delta_divergence': 2,
            'bid_ask_imbalance': 2,
            'cumulative_delta': 3,
            'tick_filter': 1,
            'smt_confirmation': 1,
            'price_near_vwap': 2,
            'liquidity_zone': 2,
            'zscore_deviation': 2,
            'vwap_slope': 1
        })

        # Confluence checks
        if self.check_trend('daily', symbol) == "uptrend":
            score += weights.get('hurst_exponent', 3)
        if not self.indicator_histories[symbol].get('4h', pd.DataFrame()).empty:
            if df['close'].iloc[-1] > df['adaptive_ma'].iloc[-1]:
                score += weights.get('adaptive_moving_average', 2)
        if self._check_break_of_structure(symbol):
            score += weights.get('bos', 2)
        if self._check_change_of_character(symbol):
            score += weights.get('choch', 1)
        if self.check_delta_divergence(symbol):
            score += weights.get('delta_divergence', 2)
        if self.check_bid_ask_imbalance(symbol):
            score += weights.get('bid_ask_imbalance', 2)
        if self.real_time_data[symbol]['cumulative_delta'] > 0:
            score += weights.get('cumulative_delta', 3)
        if self.check_tick_filter(symbol):
            score += weights.get('tick_filter', 1)
        if self.check_smt(symbol):
            score += weights.get('smt_confirmation', 1)
        if self._check_price_near_vwap(symbol):
            score += weights.get('price_near_vwap', 2)
        if self.check_liquidity_zone(symbol):
            score += weights.get('liquidity_zone', 2)
        if self._check_zscore_deviation(symbol):
            score += weights.get('zscore_deviation', 2)
        if self._check_vwap_slope(symbol):
            score += weights.get('vwap_slope', 1)

        signal_direction = "uptrend" if df['close'].iloc[-1] > df['adaptive_ma'].iloc[-1] else "downtrend"
        threshold = self.thresholds.get(trading_tf, 0)
        if score >= threshold:
            signal = {
                'action': 'buy' if signal_direction == "uptrend" else 'sell',
                'timeframe': trading_tf,
                'score': score,
                'timestamp': datetime.utcnow().isoformat(),
                'entry_price': df['close'].iloc[-1],
                'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 1.0
            }
            self.last_signals[symbol] = signal
            self.logger.info(f"Generated signal for {symbol} on {trading_tf}: {signal}")

    def check_trend(self, timeframe, symbol):
        """
        Check the trend direction for a given timeframe.

        Args:
            timeframe (str): Timeframe to check (e.g., 'daily').
            symbol (str): Trading symbol.

        Returns:
            str or None: Trend direction ('uptrend', 'downtrend', 'sideways'), or None if data insufficient.
        """
        df = self.indicator_histories[symbol].get(timeframe, pd.DataFrame())
        if df.empty:
            return None
        last_close = df['close'].iloc[-1]
        last_ma = df['adaptive_ma'].iloc[-1]
        if last_close > last_ma * 1.005:
            return "uptrend"
        elif last_close < last_ma * 0.995:
            return "downtrend"
        else:
            return "sideways"

    def _check_break_of_structure(self, symbol):
        """Check for a break of structure on 1h timeframe."""
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty or len(df) < 2:
            return False
        return df['high'].iloc[-1] > df['high'].iloc[-2]

    def _check_change_of_character(self, symbol):
        """Check for a change of character on 1h timeframe."""
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty or len(df) < 2:
            return False
        return df['low'].iloc[-1] < df['low'].iloc[-2]

    def check_delta_divergence(self, symbol):
        """Check for delta divergence based on recent delta history."""
        threshold = self.config.get('delta_divergence_threshold', 500)
        recent_delta = sum(list(self.real_time_data[symbol]['delta_history'])[-100:])
        return abs(recent_delta) > threshold

    def check_tick_filter(self, symbol):
        """Check if cumulative tick exceeds threshold."""
        tick_threshold = self.config.get('tick_threshold', 800)
        return self.real_time_data[symbol]['cumulative_tick'] > tick_threshold

    def check_bid_ask_imbalance(self, symbol):
        """Check if bid/ask imbalance exceeds threshold."""
        threshold = self.config.get('bid_ask_imbalance_threshold', 0.1)
        return abs(self.real_time_data[symbol]['bid_ask_imbalance']) > threshold

    def check_liquidity_zone(self, symbol):
        """Check for high volume liquidity zone on 1h timeframe."""
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty or len(df) < 20:
            return False
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
        multiplier = self.config.get('liquidity_zone_volume_multiplier', 1.5)
        return current_vol > multiplier * avg_vol

    def _check_price_near_vwap(self, symbol):
        """Check if price is near VWAP on 1h timeframe."""
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty:
            return False
        return abs(df['close'].iloc[-1] - df['vwap'].iloc[-1]) / df['vwap'].iloc[-1] < 0.005

    def _check_zscore_deviation(self, symbol):
        """Check if z-score deviation exceeds threshold on 1h timeframe."""
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty:
            return False
        return abs(df['zscore'].iloc[-1]) > 2

    def _check_vwap_slope(self, symbol):
        """Check if VWAP slope is positive on 1h timeframe."""
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty:
            return False
        return df['vwap_slope'].iloc[-1] > 0

    def check_smt(self, symbol):
        """Check for SMT confirmation (placeholder)."""
        return self.config.get('smt_confirmation', False)


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "symbols": ["EURUSD", "GBPJPY"],
        "signal_generation": {
            "timeframes": ["1min", "5min", "15min", "30min", "1h", "4h", "daily"],
            "trading_timeframes": ["15min", "5min", "1min"],
            "thresholds": {"15min": 15, "5min": 15, "1min": 18},
            "max_bars": {"1min": 20160, "5min": 4032, "15min": 1344, "30min": 672, "1h": 336, "4h": 84, "daily": 14}
        }
    }
    sg = SignalGenerator(config)
    tick = {"timestamp": "2025-03-12T10:00:00", "price": 1.0900, "volume": 10, "symbol": "EURUSD"}
    signal = sg.process_tick(tick)
    print(signal)