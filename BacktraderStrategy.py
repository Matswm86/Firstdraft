import backtrader as bt
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import traceback


class BacktraderStrategy(bt.Strategy):
    """
    Enhanced Backtrader strategy that integrates with MarketStructure and OrderFlow modules.
    Uses the modular trading system's analysis capabilities to generate signals and manage positions.
    """
    params = (
        ('signal_generator', None),  # SignalGenerator instance
        ('market_structure', None),  # MarketStructure instance
        ('order_flow', None),  # OrderFlow instance
        ('threshold', 6.0),  # Signal threshold (0-10 scale)
        ('stop_loss_atr_mult', 2.0),  # Default stop loss in ATR multiples
        ('take_profit_atr_mult', 3.0),  # Default take profit in ATR multiples
        ('use_optimal_levels', True),  # Use optimal levels from MarketStructure
        ('position_sizing', 'risk'),  # 'fixed' or 'risk'
        ('risk_per_trade', 0.01),  # Risk per trade (1%)
        ('max_trades_per_day', 5),  # Maximum trades per day
        ('analysis_timeframes', ['15min', '1h', '4h']),  # Timeframes to analyze
        ('primary_timeframe', '1h'),  # Primary timeframe for signal generation
        ('structure_weight', 0.6),  # Weight for market structure (60%)
        ('flow_weight', 0.4),  # Weight for order flow (40%)
    )

    def __init__(self):
        """Initialize strategy with indicators and tracking variables."""
        self.logger = logging.getLogger(__name__)

        # Order tracking
        self.order = None  # Current main order
        self.orders = {}  # All orders including SL/TP {order_ref: type}
        self.pending_orders = {}  # Orders waiting for execution {symbol: [orders]}

        # Trade tracking
        self.trade_count = 0  # Total number of trades
        self.daily_trades = {}  # Trades per day {date: count}
        self.daily_profit = 0  # Current day's profit
        self.daily_loss = 0  # Current day's loss
        self.last_day = None  # Last trading day

        # Performance metrics
        self.win_count = 0  # Number of winning trades
        self.loss_count = 0  # Number of losing trades
        self.total_profit = 0  # Cumulative profit
        self.total_loss = 0  # Cumulative loss

        # Position tracking
        self.active_positions = {}  # Current positions {symbol: details}

        # Signal tracking
        self.signals = {}  # Generated signals {symbol: signal}
        self.last_analyzed_bar = {}  # Track last analyzed bar {symbol: bar_idx}
        self.df_cache = {}  # DataFrames for analysis {symbol: df}

        # Calculate core indicators required for analysis
        self._calculate_indicators()

        self.logger.info("BacktraderStrategy initialized with integrated analysis modules")

    def _calculate_indicators(self):
        """Calculate core indicators needed for analysis."""
        # Calculate ATR for each data feed
        self.atr = {}
        self.sma20 = {}
        self.sma50 = {}
        self.sma200 = {}
        self.rsi = {}

        for i, data in enumerate(self.datas):
            # Get symbol name from data feed
            symbol = self._get_symbol_from_data(data)

            # Calculate ATR with 14-period
            self.atr[symbol] = bt.indicators.ATR(data, period=14)

            # Calculate commonly used moving averages
            self.sma20[symbol] = bt.indicators.SMA(data.close, period=20)
            self.sma50[symbol] = bt.indicators.SMA(data.close, period=50)
            self.sma200[symbol] = bt.indicators.SMA(data.close, period=200)

            # Calculate RSI
            self.rsi[symbol] = bt.indicators.RSI(data.close, period=14)

    def _get_symbol_from_data(self, data):
        """Extract symbol from data feed name."""
        if hasattr(data, '_name'):
            # Format is typically 'SYMBOL_TIMEFRAME'
            parts = data._name.split('_')
            if len(parts) > 0:
                return parts[0]

        # Fallback: use data object index
        for i, d in enumerate(self.datas):
            if d == data:
                return f"data_{i}"

        return "unknown"

    def _get_timeframe_from_data(self, data):
        """Extract timeframe from data feed name."""
        if hasattr(data, '_name'):
            parts = data._name.split('_')
            if len(parts) > 1:
                return parts[1]

        return "unknown"

    def log(self, txt, dt=None, symbol=None):
        """Log message with timestamp and optional symbol."""
        dt = dt or self.datas[0].datetime.datetime(0)
        symbol_str = f"[{symbol}] " if symbol else ""
        self.logger.info(f'{dt.isoformat()} - {symbol_str}{txt}')

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return

        # Get symbol for this order
        data = order.data
        symbol = self._get_symbol_from_data(data)

        # Handle completed orders
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size}",
                         symbol=symbol)
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size}",
                         symbol=symbol)

            # Track order completion based on type
            if order.ref in self.orders:
                order_type = self.orders[order.ref]

                if order_type == 'entry':
                    # New position opened
                    self.trade_count += 1

                    # Update daily trade count
                    current_day = self.datas[0].datetime.date(0)
                    if current_day not in self.daily_trades:
                        self.daily_trades[current_day] = 0
                    self.daily_trades[current_day] += 1

                    # Track active position
                    self.active_positions[symbol] = {
                        'entry_price': order.executed.price,
                        'size': order.executed.size,
                        'direction': 'long' if order.isbuy() else 'short',
                        'entry_time': data.datetime.datetime(0),
                    }

                elif order_type in ['sl', 'tp']:
                    # Position closed via stop loss or take profit
                    self.log(f"Position closed via {order_type.upper()}", symbol=symbol)

                    # Clean up related orders (cancel other exit orders)
                    for ref, otype in list(self.orders.items()):
                        if otype != order_type and otype != 'entry' and ref != order.ref:
                            if ref in self.broker.orders:
                                self.cancel(self.broker.orders[ref])
                            if ref in self.orders:
                                del self.orders[ref]

                    # Clear active position
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]

        # Handle failed orders
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order {order.ref} Canceled/Margin/Rejected: {order.status}", symbol=symbol)

            # Clean up order tracking
            if order.ref in self.orders:
                del self.orders[order.ref]

        # Reset main order tracking if this was the main entry order
        if self.order is not None and order.ref == self.order:
            self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications and update P&L."""
        if trade.isclosed:
            # Trade is closed - calculate P&L
            symbol = self._get_symbol_from_data(trade.data)
            profit = trade.pnlcomm  # Profit/loss including commission

            # Update daily and total P&L
            if profit > 0:
                self.daily_profit += profit
                self.total_profit += profit
                self.win_count += 1
            else:
                self.daily_loss += abs(profit)
                self.total_loss += abs(profit)
                self.loss_count += 1

            # Calculate win rate
            total_completed = self.win_count + self.loss_count
            win_rate = (self.win_count / total_completed * 100) if total_completed > 0 else 0

            # Update signal generator performance tracking
            if self.p.signal_generator is not None:
                last_signal = self.signals.get(symbol)
                if last_signal:
                    try:
                        self.p.signal_generator.update_signal_performance(last_signal, profit)
                    except Exception as e:
                        self.log(f"Error updating signal performance: {str(e)}", symbol=symbol)

            self.log(f"Trade closed: P&L=${profit:.2f}, W/L: {self.win_count}/{self.loss_count}, " +
                     f"Win Rate: {win_rate:.1f}%, Daily P/L: ${self.daily_profit - self.daily_loss:.2f}",
                     symbol=symbol)

    def _prepare_dataframe(self, data, lookback=100):
        """
        Convert backtrader data to pandas DataFrame for analysis modules.

        Args:
            data: Backtrader data feed
            lookback: Number of bars to include

        Returns:
            pd.DataFrame: DataFrame with OHLCV data and indicators
        """
        symbol = self._get_symbol_from_data(data)
        timeframe = self._get_timeframe_from_data(data)

        # Map timeframe to standard format used by analysis modules
        tf_mapping = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                      '1h': '1h', '4h': '4h', '1d': 'daily'}
        std_timeframe = tf_mapping.get(timeframe, timeframe)

        # Only build dataframe if needed (check if we've processed this bar already)
        cache_key = f"{symbol}_{std_timeframe}"
        current_len = len(data)

        if cache_key in self.last_analyzed_bar and self.last_analyzed_bar[cache_key] == current_len:
            return self.df_cache.get(cache_key)

        self.last_analyzed_bar[cache_key] = current_len

        try:
            # Extract OHLCV data with specified lookback
            actual_lookback = min(lookback, len(data))

            # Initialize arrays
            dates = []
            o = np.zeros(actual_lookback)
            h = np.zeros(actual_lookback)
            l = np.zeros(actual_lookback)
            c = np.zeros(actual_lookback)
            v = np.zeros(actual_lookback)

            # Fill arrays with data
            for i in range(actual_lookback):
                bar_idx = -actual_lookback + i
                dates.append(data.datetime.datetime(bar_idx))
                o[i] = data.open[bar_idx]
                h[i] = data.high[bar_idx]
                l[i] = data.low[bar_idx]
                c[i] = data.close[bar_idx]
                v[i] = data.volume[bar_idx]

            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v,
                'symbol': symbol
            })

            # Add ATR
            df['atr'] = np.array([self.atr[symbol].array[i - actual_lookback] for i in range(actual_lookback)])

            # Add average ATR
            df['avg_atr'] = df['atr'].rolling(window=20, min_periods=1).mean()

            # Add adaptive MA as required by analysis modules
            df['adaptive_ma'] = np.array(
                [self.sma20[symbol].array[i - actual_lookback] for i in range(actual_lookback)])

            # Add SMA slope
            df['sma_slope'] = pd.Series(
                np.array([self.sma20[symbol].array[i - actual_lookback] for i in range(actual_lookback)])
            ).diff(periods=5) / 5

            # Add RSI
            df['rsi'] = np.array([self.rsi[symbol].array[i - actual_lookback] for i in range(actual_lookback)])

            # Calculate VWAP (simplified version for backtest)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

                # Add VWAP slope
                df['vwap_slope'] = df['vwap'].diff(periods=5) / 5

                # Add VWAP bands
                df['vwap_std'] = (df['close'] - df['vwap']).rolling(window=20, min_periods=1).std()
                df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
                df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
                df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
                df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']
            else:
                # Fallback if volume data is missing
                df['vwap'] = df['close'].rolling(window=20, min_periods=1).mean()
                df['vwap_slope'] = df['vwap'].diff(periods=5) / 5
                df['vwap_std'] = df['close'].rolling(window=20, min_periods=1).std()
                df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
                df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
                df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
                df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']

            # Calculate Z-score
            mean = df['close'].rolling(window=20, min_periods=1).mean()
            std = df['close'].rolling(window=20, min_periods=1).std()
            df['zscore'] = np.where(std > 0, (df['close'] - mean) / std, 0)

            # Store in cache
            self.df_cache[cache_key] = df
            return df

        except Exception as e:
            self.log(f"Error preparing DataFrame: {str(e)}", symbol=symbol)
            self.logger.error(traceback.format_exc())
            return None

    def _simulate_real_time_data(self, data):
        """
        Create simulated real-time data needed for OrderFlow analysis.

        Args:
            data: Backtrader data feed

        Returns:
            dict: Simulated real-time data
        """
        symbol = self._get_symbol_from_data(data)

        # Create basic real-time data structure
        real_time_data = {
            'last_price': data.close[0],
            'bid': data.close[0] * 0.9999,  # Simulated bid slightly below price
            'ask': data.close[0] * 1.0001,  # Simulated ask slightly above price
            'last_volume': data.volume[0],
            'delta_history': [],
            'bid_ask_imbalance': 0,
            'cumulative_delta': 0,
            'tick': 0,
            'cumulative_tick': 0
        }

        # Generate simulated delta history based on recent price movements
        lookback = 50
        real_time_data['delta_history'] = []

        for i in range(min(lookback, len(data) - 1)):
            price_change = data.close[-i] - data.close[-i - 1]
            volume = data.volume[-i]

            # Delta is positive when price rises, negative when price falls
            if price_change > 0:
                delta = volume
            elif price_change < 0:
                delta = -volume
            else:
                delta = 0

            real_time_data['delta_history'].append(delta)

        # Reverse to get chronological order
        real_time_data['delta_history'].reverse()

        # Calculate cumulative delta
        if real_time_data['delta_history']:
            real_time_data['cumulative_delta'] = sum(real_time_data['delta_history'])

        # Simulate bid-ask imbalance based on recent price direction
        recent_bars = 5
        if len(data) >= recent_bars:
            recent_direction = np.sign(data.close[0] - data.close[-recent_bars])
            real_time_data['bid_ask_imbalance'] = recent_direction * 0.1  # Mild imbalance

        return real_time_data

    def _analyze_symbol(self, data):
        """
        Perform full analysis on a symbol using MarketStructure and OrderFlow modules.

        Args:
            data: Backtrader data feed

        Returns:
            tuple: (signal, structure_analysis, flow_analysis)
        """
        if self.p.market_structure is None or self.p.order_flow is None:
            self.log("Analysis modules not available")
            return None, None, None

        symbol = self._get_symbol_from_data(data)
        timeframe = self._get_timeframe_from_data(data)

        # Map timeframe to standard format used by analysis modules
        tf_mapping = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                      '1h': '1h', '4h': '4h', '1d': 'daily'}
        std_timeframe = tf_mapping.get(timeframe, timeframe)

        # Check if this is a primary analysis timeframe
        if std_timeframe not in self.p.analysis_timeframes:
            return None, None, None

        # Convert data to DataFrame
        df = self._prepare_dataframe(data)
        if df is None or df.empty:
            return None, None, None

        # Get current time
        current_time = data.datetime.datetime(0)

        # Create simulated real-time data
        real_time_data = self._simulate_real_time_data(data)

        try:
            # Perform Market Structure analysis
            structure_analysis = self.p.market_structure.analyze(
                symbol, std_timeframe, df, current_time
            )

            # Perform Order Flow analysis
            flow_analysis = self.p.order_flow.analyze(
                symbol, std_timeframe, df, real_time_data
            )

            # Check if both analyses are valid
            if not structure_analysis.get('valid', False) or not flow_analysis.get('valid', False):
                return None, structure_analysis, flow_analysis

            # Check for signal based on combined analysis
            signal = self._evaluate_combined_analysis(
                symbol, std_timeframe, data, structure_analysis, flow_analysis
            )

            return signal, structure_analysis, flow_analysis

        except Exception as e:
            self.log(f"Error in analysis: {str(e)}", symbol=symbol)
            self.logger.error(traceback.format_exc())
            return None, None, None

    def _evaluate_combined_analysis(self, symbol, timeframe, data, structure_analysis, flow_analysis):
        """
        Determine if a signal should be generated based on combined analysis.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            data: Backtrader data feed
            structure_analysis: MarketStructure analysis results
            flow_analysis: OrderFlow analysis results

        Returns:
            dict: Signal dictionary or None
        """
        try:
            # Get scores from analysis
            structure_score = structure_analysis.get('structure_score', 0)
            flow_score = flow_analysis.get('flow_score', 0)

            # Get trend direction
            direction = structure_analysis.get('direction', 'sideways')

            # Calculate combined score with weights
            combined_score = (structure_score * self.p.structure_weight) + (flow_score * self.p.flow_weight)

            # Only generate signal for clear trend direction and sufficient score
            if direction == 'sideways' or combined_score < self.p.threshold:
                return None

            # Map direction to action
            action = 'buy' if direction == 'uptrend' else 'sell' if direction == 'downtrend' else None

            if action is None:
                return None

            # Get current price for entry
            entry_price = data.close[0]

            # Get optimal stop loss and take profit levels
            if self.p.use_optimal_levels and 'optimal_levels' in structure_analysis:
                stop_loss = structure_analysis['optimal_levels'].get('stop_loss')
                take_profit = structure_analysis['optimal_levels'].get('take_profit')
                risk_reward_ratio = structure_analysis['optimal_levels'].get('risk_reward_ratio', 0)
            else:
                # Default ATR-based levels
                atr = self.atr[symbol][0]

                if action == 'buy':
                    stop_loss = entry_price - (atr * self.p.stop_loss_atr_mult)
                    take_profit = entry_price + (atr * self.p.take_profit_atr_mult)
                else:  # 'sell'
                    stop_loss = entry_price + (atr * self.p.stop_loss_atr_mult)
                    take_profit = entry_price - (atr * self.p.take_profit_atr_mult)

                # Calculate risk-reward ratio
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                risk_reward_ratio = reward / risk if risk > 0 else 0

            # Build signal dictionary
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'action': action,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': data.datetime.datetime(0).isoformat(),
                'score': combined_score,
                'risk_reward_ratio': risk_reward_ratio,
                'structure_score': structure_score,
                'flow_score': flow_score,
                'regime': structure_analysis.get('regime', 'undefined')
            }

            return signal

        except Exception as e:
            self.log(f"Error evaluating analysis: {str(e)}", symbol=symbol)
            return None

    def _calculate_position_size(self, data, price, stop_loss):
        """
        Calculate position size based on risk parameters.

        Args:
            data: Backtrader data feed
            price: Entry price
            stop_loss: Stop loss price

        Returns:
            int: Position size
        """
        symbol = self._get_symbol_from_data(data)

        # Fixed position sizing
        if self.p.position_sizing == 'fixed':
            return 1  # Default fixed size

        # Risk-based position sizing
        elif self.p.position_sizing == 'risk':
            # Calculate risk per trade in account value
            risk_amount = self.broker.getvalue() * self.p.risk_per_trade

            # Calculate risk per unit based on distance to stop loss
            risk_per_unit = abs(price - stop_loss)

            if risk_per_unit <= 0:
                self.log(f"Invalid risk per unit: {risk_per_unit}", symbol=symbol)
                return 1

            # Calculate position size
            size = risk_amount / risk_per_unit

            # Ensure minimum size of 1
            return max(1, int(size))

        # Default to 1 if no valid sizing method
        return 1

    def _check_daily_trade_limit(self):
        """Check if daily trade limit has been reached."""
        if self.p.max_trades_per_day <= 0:
            return False  # No limit

        current_day = self.datas[0].datetime.date(0)
        daily_count = self.daily_trades.get(current_day, 0)

        return daily_count >= self.p.max_trades_per_day

    def _process_signals(self):
        """Process generated signals for all symbols."""
        # Skip if we've reached daily trade limit
        if self._check_daily_trade_limit():
            # self.log("Daily trade limit reached")
            return

        # Skip if an order is pending
        if self.order is not None:
            return

        # Loop through all data feeds
        for data in self.datas:
            symbol = self._get_symbol_from_data(data)

            # Skip if already have an active position for this symbol
            if symbol in self.active_positions:
                continue

            # Analyze this symbol on current timeframe
            signal, structure_analysis, flow_analysis = self._analyze_symbol(data)

            # Process valid signal
            if signal:
                # Store signal for reference
                self.signals[symbol] = signal

                # Extract parameters
                action = signal['action']
                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']
                take_profit = signal['take_profit']

                # Calculate position size
                size = self._calculate_position_size(data, entry_price, stop_loss)

                # Execute trades based on signal
                try:
                    if action == 'buy':
                        # Place buy order
                        self.order = self.buy(data=data, size=size)
                        self.orders[self.order] = 'entry'

                        # Place stop loss and take profit orders
                        sl_order = self.sell(data=data, exectype=bt.Order.Stop,
                                             price=stop_loss, size=size)
                        tp_order = self.sell(data=data, exectype=bt.Order.Limit,
                                             price=take_profit, size=size)

                        self.orders[sl_order] = 'sl'
                        self.orders[tp_order] = 'tp'

                        self.log(f"BUY ORDER PLACED - Price: {entry_price:.5f}, SL: {stop_loss:.5f}, " +
                                 f"TP: {take_profit:.5f}, Size: {size}, Score: {signal['score']:.2f}",
                                 symbol=symbol)

                    elif action == 'sell':
                        # Place sell order
                        self.order = self.sell(data=data, size=size)
                        self.orders[self.order] = 'entry'

                        # Place stop loss and take profit orders
                        sl_order = self.buy(data=data, exectype=bt.Order.Stop,
                                            price=stop_loss, size=size)
                        tp_order = self.buy(data=data, exectype=bt.Order.Limit,
                                            price=take_profit, size=size)

                        self.orders[sl_order] = 'sl'
                        self.orders[tp_order] = 'tp'

                        self.log(f"SELL ORDER PLACED - Price: {entry_price:.5f}, SL: {stop_loss:.5f}, " +
                                 f"TP: {take_profit:.5f}, Size: {size}, Score: {signal['score']:.2f}",
                                 symbol=symbol)

                except Exception as e:
                    self.log(f"Error placing orders: {str(e)}", symbol=symbol)
                    self.order = None

                # Only process one signal per cycle
                break

    def next(self):
        """Execute strategy logic for each bar."""
        # Reset daily metrics at the start of a new day
        current_day = self.datas[0].datetime.date(0)
        if self.last_day is None or current_day > self.last_day:
            self.daily_profit = 0
            self.daily_loss = 0
            self.last_day = current_day

        # Process signals and place trades
        self._process_signals()

    def stop(self):
        """Log final results when backtest completes."""
        # Calculate performance metrics
        total_completed = self.win_count + self.loss_count
        win_rate = (self.win_count / total_completed * 100) if total_completed > 0 else 0

        total_pnl = self.total_profit - self.total_loss
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')

        self.log(f"\nBacktest Completed Summary:")
        self.log(f"  Trade Count: {self.trade_count}")
        self.log(f"  Win Rate: {win_rate:.2f}% ({self.win_count}/{total_completed})")
        self.log(f"  Profit Factor: {profit_factor:.2f}")
        self.log(f"  Total Profit: ${self.total_profit:.2f}")
        self.log(f"  Total Loss: ${self.total_loss:.2f}")
        self.log(f"  Net P&L: ${total_pnl:.2f}")