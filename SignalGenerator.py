import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import pytz
from hurst import compute_Hc
import ta
import logging
import MetaTrader5 as mt5


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

        self.timeframe_intervals = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            'daily': timedelta(days=1)
        }

        # Bootstrap historical data
        if self.mt5_api:
            self._load_initial_history()

    def _load_initial_history(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                mt5_tf = {
                    '1min': mt5.TIMEFRAME_M1,
                    '5min': mt5.TIMEFRAME_M5,
                    '15min': mt5.TIMEFRAME_M15,
                    '30min': mt5.TIMEFRAME_M30,
                    '1h': mt5.TIMEFRAME_H1,
                    '4h': mt5.TIMEFRAME_H4,
                    'daily': mt5.TIMEFRAME_D1
                }[tf]
                rates = self.mt5_api.get_ohlc_data(symbol, mt5_tf, 100)  # Load 100 bars
                if rates:
                    for rate in rates:
                        bar = {
                            'timestamp': pd.to_datetime(rate['timestamp']),
                            'open': rate['open'],
                            'high': rate['high'],
                            'low': rate['low'],
                            'close': rate['close'],
                            'volume': rate['volume']
                        }
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
        interval = self.timeframe_intervals[timeframe]
        current_bar = self.current_bars[symbol][timeframe]
        if current_bar is None or tick_time >= current_bar['timestamp'] + interval:
            if current_bar:
                completed_bar = current_bar.copy()
                self.histories[symbol][timeframe].append(completed_bar)
            else:
                completed_bar = None
            self.current_bars[symbol][timeframe] = {
                'timestamp': tick_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
            self.logger.debug(f"Aggregated bar for {symbol} on {timeframe}: {self.current_bars[symbol][timeframe]}")
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
            if self.config.get('ad_line_bars'):
                df['ad_line'] = self.calculate_ad_line(df, self.config['ad_line_bars'])
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
        rtd['composite_breadth_score'] = (
                rtd['composite_breadth_score'] * 0.9 +
                rtd['tick'] * 0.1
        )
        rtd['last_price'] = price
        rtd['last_volume'] = volume

    def calculate_indicators(self, df, timeframe):
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'],
            window=14, fillna=True
        )
        df['atr'] = atr_indicator.average_true_range()
        lookback = max(1, min(100, int(df['atr'].iloc[-1] * 10) if not pd.isna(df['atr'].iloc[-1]) else 30))
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

        df['sma_slope'] = (df['adaptive_ma'] - df['adaptive_ma'].shift(5)) / 5
        df['avg_atr'] = df['atr'].rolling(window=20).mean()
        df['vwap_std'] = (df['close'] - df['vwap']).rolling(window=20).std()
        df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
        df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
        df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
        df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']

        return df

    def calculate_ad_line(self, df, bars):
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

    def calculate_volume_delta_oscillator(self, symbol, tf, window=20):
        df = self.indicator_histories[symbol].get(tf, pd.DataFrame())
        if df.empty or len(df) < window:
            return 0
        delta = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']
        cumulative_delta = delta.rolling(window=min(window, len(df))).sum()
        total_volume = df['volume'].rolling(window=min(window, len(df))).sum()
        oscillator = cumulative_delta / total_volume if total_volume.iloc[-1] != 0 else 0
        return oscillator.iloc[-1]

    def generate_signal_for_tf(self, trading_tf, symbol):
        rtd = self.real_time_data[symbol]
        if rtd['last_price'] is None:
            self.logger.debug(f"No price data yet for {symbol} on {trading_tf}")
            return

        df = self.indicator_histories[symbol][trading_tf]
        if df.empty or len(df) < 14:
            self.logger.debug(f"Insufficient indicator history for {symbol} on {trading_tf}: {len(df)} bars")
            return

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

        signal_direction = "uptrend" if df['close'].iloc[-1] > df['adaptive_ma'].iloc[-1] else "downtrend"
        self.logger.debug(f"Signal direction for {symbol} on {trading_tf}: {signal_direction}")

        hurst = None
        if 'daily' in self.indicator_histories[symbol] and 'hurst' in self.indicator_histories[symbol]['daily'].columns:
            hurst = self.indicator_histories[symbol]['daily']['hurst'].iloc[-1]

        trend_following = ['adaptive_moving_average', 'bos', 'choch', 'vwap_slope', 'cumulative_delta']
        mean_reverting = ['price_near_vwap', 'liquidity_zone', 'zscore_deviation']

        if hurst is not None:
            if hurst > 0.5:
                adjusted_weights = {k: v * 1.5 if k in trend_following else v for k, v in weights.items()}
            elif hurst < 0.5:
                adjusted_weights = {k: v * 1.5 if k in mean_reverting else v for k, v in weights.items()}
            else:
                adjusted_weights = weights
        else:
            adjusted_weights = weights

        score = 0

        higher_tf_uptrend = 0
        higher_tf_downtrend = 0
        for higher_tf in ['1h', '4h']:
            if higher_tf in self.indicator_histories[symbol] and not self.indicator_histories[symbol][higher_tf].empty:
                sma_slope = self.indicator_histories[symbol][higher_tf]['sma_slope'].iloc[-1]
                if sma_slope > 0.0001:
                    higher_tf_uptrend += 1
                elif sma_slope < -0.0001:
                    higher_tf_downtrend += 1

        if signal_direction == "uptrend":
            score += higher_tf_uptrend * 2
        elif signal_direction == "downtrend":
            score += higher_tf_downtrend * 2

        if self.check_trend('daily', symbol) == signal_direction:
            score += adjusted_weights.get('hurst_exponent', 3)
        if (df['close'].iloc[-1] > df['adaptive_ma'].iloc[-1] and signal_direction == "uptrend") or \
                (df['close'].iloc[-1] < df['adaptive_ma'].iloc[-1] and signal_direction == "downtrend"):
            score += adjusted_weights.get('adaptive_moving_average', 2)
        if self._check_break_of_structure(symbol):
            score += adjusted_weights.get('bos', 2)
        if self._check_change_of_character(symbol):
            score += adjusted_weights.get('choch', 1)
        if self.check_delta_divergence(symbol):
            score += adjusted_weights.get('delta_divergence', 2)
        if self.check_bid_ask_imbalance(symbol):
            score += adjusted_weights.get('bid_ask_imbalance', 2)
        if (self.real_time_data[symbol]['cumulative_delta'] > 0 and signal_direction == "uptrend") or \
                (self.real_time_data[symbol]['cumulative_delta'] < 0 and signal_direction == "downtrend"):
            score += adjusted_weights.get('cumulative_delta', 3)
        if self.check_tick_filter(symbol):
            score += adjusted_weights.get('tick_filter', 1)
        if self.check_smt(symbol):
            score += adjusted_weights.get('smt_confirmation', 1)
        if self._check_price_near_vwap(symbol):
            score += adjusted_weights.get('price_near_vwap', 2)
        if self.check_liquidity_zone(symbol):
            score += adjusted_weights.get('liquidity_zone', 2)
        if self._check_zscore_deviation(symbol):
            score += adjusted_weights.get('zscore_deviation', 2)
        if (df['vwap_slope'].iloc[-1] > 0 and signal_direction == "uptrend") or \
                (df['vwap_slope'].iloc[-1] < 0 and signal_direction == "downtrend"):
            score += adjusted_weights.get('vwap_slope', 1)

        oscillator = self.calculate_volume_delta_oscillator(symbol, trading_tf)
        if (signal_direction == "uptrend" and oscillator > 0.5) or \
                (signal_direction == "downtrend" and oscillator < -0.5):
            score += 2

        threshold = self.thresholds.get(trading_tf, 0)
        self.logger.debug(f"Signal score for {symbol} on {trading_tf}: {score} (Threshold: {threshold})")

        test_threshold = 5
        if score >= test_threshold:
            signal = {
                'action': 'buy' if signal_direction == "uptrend" else 'sell',
                'timeframe': trading_tf,
                'score': score,
                'timestamp': datetime.now(pytz.UTC).isoformat(),
                'entry_price': df['close'].iloc[-1],
                'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 0.001,
                'symbol': symbol
            }
            self.last_signals[symbol] = signal
            self.logger.info(f"Generated signal for {symbol} on {trading_tf}: {signal}")
        else:
            self.logger.debug(f"Signal rejected for {symbol} on {trading_tf}: Score {score} < {test_threshold}")

    def check_trend(self, timeframe, symbol):
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
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty or len(df) < 20:
            return False
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['avg_atr'].iloc[-1]
        lookback = 5 if current_atr > 1.5 * avg_atr else 20
        if len(df) < lookback:
            return False
        recent_highs = df['high'].iloc[-lookback:]
        return df['high'].iloc[-1] > max(recent_highs[:-1])

    def _check_change_of_character(self, symbol):
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty or len(df) < 20:
            return False
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['avg_atr'].iloc[-1]
        lookback = 5 if current_atr > 1.5 * avg_atr else 20
        if len(df) < lookback:
            return False
        recent_lows = df['low'].iloc[-lookback:]
        return df['low'].iloc[-1] < min(recent_lows[:-1])

    def check_delta_divergence(self, symbol):
        threshold = self.config.get('delta_divergence_threshold', 500)
        recent_delta = sum(list(self.real_time_data[symbol]['delta_history'])[-100:])
        return abs(recent_delta) > threshold

    def check_tick_filter(self, symbol):
        tick_threshold = self.config.get('tick_threshold', 800)
        return self.real_time_data[symbol]['cumulative_tick'] > tick_threshold

    def check_bid_ask_imbalance(self, symbol):
        threshold = self.config.get('bid_ask_imbalance_threshold', 0.1)
        return abs(self.real_time_data[symbol]['bid_ask_imbalance']) > threshold

    def check_liquidity_zone(self, symbol):
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty or len(df) < 20:
            return False
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
        multiplier = self.config.get('liquidity_zone_volume_multiplier', 1.5)
        return current_vol > multiplier * avg_vol

    def _check_price_near_vwap(self, symbol):
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty:
            return False
        return abs(df['close'].iloc[-1] - df['vwap'].iloc[-1]) / df['vwap'].iloc[-1] < 0.005

    def _check_zscore_deviation(self, symbol):
        df = self.indicator_histories[symbol].get('1h', pd.DataFrame())
        if df.empty:
            return False
        return abs(df['zscore'].iloc[-1]) > 2

    def check_smt(self, symbol):
        return self.config.get('smt_confirmation', False)