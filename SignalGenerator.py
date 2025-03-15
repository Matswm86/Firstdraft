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
import traceback

# Import specialized modules
from MarketStructure import MarketStructure
from OrderFlow import OrderFlow


class SignalGenerator:
    """
    Signal Generator module for generating trading signals based on technical analysis.
    Coordinates Market Structure and Order Flow analysis for high-probability setups.
    """

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

        # Initialize specialized modules
        self.market_structure = MarketStructure(config, self)
        self.order_flow = OrderFlow(config, self)

        # Historical performance tracking
        self.signal_performance = {symbol: {'win_count': 0, 'loss_count': 0, 'total_profit': 0, 'total_loss': 0}
                                   for symbol in self.symbols}

        # Initialize core analysis parameters
        self.atr_period = 14
        self.rsi_period = 14

        # Load initial history if MT5API is available
        if self.mt5_api and self.config['central_trading_bot']['mode'] == 'live':
            self._load_initial_history()

        self.logger.info("SignalGenerator initialized with MarketStructure and OrderFlow modules")

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
                            df = self.calculate_indicators(df)
                            self.indicator_histories[symbol][tf] = df
                            self.logger.info(f"Loaded initial history for {symbol} on {tf}: {len(df)} bars")
                except Exception as e:
                    self.logger.error(f"Error loading history for {symbol} on {tf}: {str(e)}")
                    self.logger.debug(traceback.format_exc())

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

    def _aggregate_tick(self, timeframe: str, tick_time: datetime, price: float, volume: float, symbol: str) -> \
    Optional[Dict[str, Any]]:
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
                    return None
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
                df = self.calculate_indicators(df)
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

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators for a DataFrame of price data.

        Args:
            df (pd.DataFrame): DataFrame with OHLC data

        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column for indicators: {col}")
                    return df

            # Calculate ATR
            df['atr'] = volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'],
                window=self.atr_period, fillna=True
            ).average_true_range().fillna(0.001)

            # Calculate average ATR
            df['avg_atr'] = df['atr'].rolling(window=20, min_periods=1).mean()

            # Calculate adaptive moving average
            lookback = 30
            if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]):
                lookback = max(1, min(100, int(df['atr'].iloc[-1] * 1000)))
            df['adaptive_ma'] = df['close'].rolling(window=lookback, min_periods=1).mean()

            # Calculate RSI
            df['rsi'] = momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi().fillna(50)

            # Calculate VWAP
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            else:
                df['vwap'] = df['close']

            # Calculate VWAP slope
            if 'vwap' in df.columns:
                df['vwap_slope'] = df['vwap'].diff(periods=5) / 5

            # Calculate Z-score
            mean = df['close'].rolling(window=20, min_periods=1).mean()
            std = df['close'].rolling(window=20, min_periods=1).std()
            df['zscore'] = np.where(std > 0, (df['close'] - mean) / std, 0)

            # Calculate slope of adaptive MA
            if 'adaptive_ma' in df.columns:
                df['sma_slope'] = (df['adaptive_ma'] - df['adaptive_ma'].shift(5)) / 5

            # Calculate VWAP bands
            if 'vwap' in df.columns:
                df['vwap_std'] = (df['close'] - df['vwap']).rolling(window=20, min_periods=1).std()
                df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
                df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
                df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
                df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']

            self.logger.debug(
                f"Calculated indicators: close={df['close'].iloc[-1]}, adaptive_ma={df['adaptive_ma'].iloc[-1]}")
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df

    def generate_signal_for_tf(self, trading_tf: str, symbol: str, current_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal for a specific timeframe by combining
        market structure and order flow analysis.

        Args:
            trading_tf (str): Timeframe to generate signal for
            symbol (str): Symbol to generate signal for
            current_time (datetime): Current tick time from MT5

        Returns:
            Optional[Dict[str, Any]]: Generated signal if applicable, None otherwise
        """
        try:
            # Check for minimum data requirements
            rtd = self.real_time_data.get(symbol)
            if not rtd or rtd['last_price'] is None:
                self.logger.debug(f"No real-time data for {symbol}")
                return None

            df = self.indicator_histories.get(symbol, {}).get(trading_tf)
            if df is None or df.empty or len(df) < 14:
                self.logger.debug(
                    f"Insufficient data for {symbol} on {trading_tf}: {len(df) if df is not None else 'None'} bars")
                return None

            # Get market structure analysis
            structure_analysis = self.market_structure.analyze(symbol, trading_tf, df, current_time)

            # If structure analysis is invalid, no signal is generated
            if not structure_analysis.get('valid', False):
                self.logger.debug(f"Invalid market structure analysis for {symbol} on {trading_tf}")
                return None

            # Get order flow analysis
            flow_analysis = self.order_flow.analyze(symbol, trading_tf, df, rtd)

            # If order flow analysis is invalid, no signal is generated
            if not flow_analysis.get('valid', False):
                self.logger.debug(f"Invalid order flow analysis for {symbol} on {trading_tf}")
                return None

            # Combine analyses for signal decision
            signal_decision = self._evaluate_combined_analysis(
                symbol, trading_tf, structure_analysis, flow_analysis
            )

            if not signal_decision['generate_signal']:
                self.logger.debug(f"Signal criteria not met for {symbol} on {trading_tf}: {signal_decision['reason']}")
                return None

            # Create signal with combined parameters
            entry_price = df['close'].iloc[-1]
            signal = self._construct_signal(
                symbol, trading_tf, current_time, entry_price,
                structure_analysis, flow_analysis, signal_decision
            )

            self.logger.info(
                f"Generated signal for {symbol} on {trading_tf}: {signal['action']} with score {signal['score']}")
            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol} on {trading_tf}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

    def _evaluate_combined_analysis(self, symbol: str, timeframe: str,
                                    structure_analysis: Dict, flow_analysis: Dict) -> Dict:
        """
        Evaluate combined market structure and order flow analysis for signal decision.

        Args:
            symbol (str): Symbol being analyzed
            timeframe (str): Timeframe being analyzed
            structure_analysis (Dict): Market structure analysis results
            flow_analysis (Dict): Order flow analysis results

        Returns:
            Dict: Signal decision with reason and direction
        """
        try:
            # Get signal direction from market structure
            # Default to None if no clear direction
            direction = structure_analysis.get('direction', None)

            # Get scores from both analyses
            structure_score = structure_analysis.get('structure_score', 0)
            flow_score = flow_analysis.get('flow_score', 0)

            # Calculate combined score (weighted sum)
            # Default weights: 60% structure, 40% order flow
            structure_weight = self.config.get('structure_weight', 0.6)
            flow_weight = self.config.get('flow_weight', 0.4)

            combined_score = (structure_score * structure_weight) + (flow_score * flow_weight)

            # Get threshold for this timeframe
            threshold = self.thresholds.get(timeframe, 5)

            # Check if order flow confirms structure direction
            flow_direction = flow_analysis.get('direction', 'neutral')
            directional_agreement = (
                    (direction == 'uptrend' and flow_direction == 'up') or
                    (direction == 'downtrend' and flow_direction == 'down') or
                    (flow_direction == 'neutral')
            )

            # Initial decision (assumes no signal)
            decision = {
                'generate_signal': False,
                'reason': 'Unknown',
                'direction': direction,
                'action': 'buy' if direction == 'uptrend' else 'sell' if direction == 'downtrend' else None,
                'combined_score': combined_score
            }

            # Apply decision logic
            if not direction or direction == 'sideways':
                decision['reason'] = 'No clear trend direction'
            elif not directional_agreement:
                decision['reason'] = f'Order flow ({flow_direction}) contradicts price direction ({direction})'
            elif combined_score < threshold:
                decision['reason'] = f'Combined score ({combined_score:.2f}) below threshold ({threshold})'
            else:
                # All criteria met, generate signal
                decision['generate_signal'] = True
                decision['reason'] = f'Strong {direction} with confirming order flow'

            return decision

        except Exception as e:
            self.logger.error(f"Error evaluating combined analysis: {str(e)}")
            return {'generate_signal': False, 'reason': f'Error: {str(e)}'}

    def _construct_signal(self, symbol: str, timeframe: str, current_time: datetime,
                          entry_price: float, structure_analysis: Dict,
                          flow_analysis: Dict, signal_decision: Dict) -> Dict[str, Any]:
        """
        Construct a complete trading signal from analysis components.

        Args:
            symbol (str): Symbol to trade
            timeframe (str): Signal timeframe
            current_time (datetime): Current market time
            entry_price (float): Entry price for the signal
            structure_analysis (Dict): Market structure analysis
            flow_analysis (Dict): Order flow analysis
            signal_decision (Dict): Combined decision logic results

        Returns:
            Dict[str, Any]: Complete trading signal
        """
        try:
            # Get ATR for volatility calculations
            df = self.indicator_histories.get(symbol, {}).get(timeframe)
            atr = df['atr'].iloc[-1] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]) else 0.001

            # Determine action from decision
            action = signal_decision['action']

            # Get optimal stop loss and take profit levels
            sl_tp = structure_analysis.get('optimal_levels', {})
            stop_loss = sl_tp.get('stop_loss')
            take_profit = sl_tp.get('take_profit')

            # If structure analysis didn't provide levels, use default ATR-based calculation
            if stop_loss is None or take_profit is None:
                if action == 'buy':
                    stop_loss = entry_price - (atr * 2)
                    take_profit = entry_price + (atr * 3)
                else:  # 'sell'
                    stop_loss = entry_price + (atr * 2)
                    take_profit = entry_price - (atr * 3)

            # Calculate risk-reward ratio
            if action == 'buy':
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # 'sell'
                risk = stop_loss - entry_price
                reward = entry_price - take_profit

            risk_reward_ratio = reward / risk if risk > 0 else 0

            # Create comprehensive signal
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'action': action,
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'timestamp': current_time.isoformat(),
                'score': float(signal_decision['combined_score']),
                'risk_reward_ratio': float(risk_reward_ratio),
                'atr': float(atr),

                # Include key analysis components
                'regime': structure_analysis.get('regime', 'undefined'),
                'patterns': structure_analysis.get('patterns', []),
                'order_flow': flow_analysis.get('direction', 'neutral'),
                'liquidity': flow_analysis.get('liquidity', {}),

                # Add performance metrics if available
                'win_rate': self._get_historical_win_rate(symbol, timeframe, action)
            }

            return signal

        except Exception as e:
            self.logger.error(f"Error constructing signal: {str(e)}")
            # Return basic signal with error
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'action': signal_decision.get('action', 'buy'),
                'entry_price': float(entry_price),
                'timestamp': current_time.isoformat(),
                'score': float(signal_decision.get('combined_score', 0)),
                'atr': float(atr),
                'error': str(e)
            }

    def _get_historical_win_rate(self, symbol: str, timeframe: str, action: str) -> float:
        """
        Get historical win rate for similar signals.

        Args:
            symbol (str): Trading symbol
            timeframe (str): Signal timeframe
            action (str): Signal action (buy/sell)

        Returns:
            float: Historical win rate (0-1) or 0.5 if no history
        """
        try:
            if symbol not in self.signal_performance:
                return 0.5

            perf = self.signal_performance[symbol]
            total_trades = perf['win_count'] + perf['loss_count']

            if total_trades == 0:
                return 0.5

            return perf['win_count'] / total_trades

        except Exception as e:
            self.logger.error(f"Error getting historical win rate: {str(e)}")
            return 0.5

    def update_signal_performance(self, signal: Dict[str, Any], profit_loss: float) -> None:
        """
        Update historical performance tracking for signals.

        Args:
            signal (Dict[str, Any]): The original signal
            profit_loss (float): The profit/loss result
        """
        try:
            if not signal or 'symbol' not in signal:
                return

            symbol = signal['symbol']

            if symbol not in self.signal_performance:
                self.signal_performance[symbol] = {
                    'win_count': 0,
                    'loss_count': 0,
                    'total_profit': 0,
                    'total_loss': 0
                }

            if profit_loss > 0:
                self.signal_performance[symbol]['win_count'] += 1
                self.signal_performance[symbol]['total_profit'] += profit_loss
            else:
                self.signal_performance[symbol]['loss_count'] += 1
                self.signal_performance[symbol]['total_loss'] += profit_loss

        except Exception as e:
            self.logger.error(f"Error updating signal performance: {str(e)}")

    def estimate_slippage(self, symbol: str, order_size: float = 1.0) -> float:
        """
        Estimate slippage based on order size and market conditions.

        Args:
            symbol (str): Symbol to estimate for
            order_size (float): Order size in lots

        Returns:
            float: Estimated slippage in price points
        """
        try:
            df = self.indicator_histories.get(symbol, {}).get('1h')
            if df is None or df.empty:
                return 0.001  # Default minimal slippage

            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001

            # Get real-time market data
            rtd = self.real_time_data.get(symbol)
            if not rtd:
                return atr * 0.1  # Default to 10% of ATR if no real-time data

            # Use order flow information from OrderFlow module
            slippage_estimate = self.order_flow.estimate_slippage(symbol, order_size)

            # If OrderFlow provides a valid estimate, use it
            if slippage_estimate is not None and slippage_estimate > 0:
                return slippage_estimate

            # Fallback to basic calculation
            avg_volume = df['volume'].iloc[-20:].mean() if 'volume' in df.columns else 100
            impact_factor = min(1.0, order_size / (avg_volume * 0.1)) if avg_volume > 0 else 0.1

            return atr * impact_factor

        except Exception as e:
            self.logger.error(f"Error estimating slippage: {str(e)}")
            return 0.001  # Default minimal slippage on error