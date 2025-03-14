import pandas as pd
import logging
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import time
import uuid
import random
import numpy as np


class TradeExecution:
    """
    Trade Execution module for handling order submission and management.
    Handles market and pending orders, position modifications, and order cancellations.
    """

    def __init__(self, config, mt5_api):
        """
        Initialize TradeExecution with configuration settings.

        Args:
            config (dict): Configuration dictionary with trade execution settings
            mt5_api: Instance of MT5API for trade operations
        """
        self.config = config
        self.order_type = config.get('order_type', 'market')
        self.mt5_api = mt5_api
        self.logger = logging.getLogger(__name__)

        # Trading settings from config
        self.max_spread_multiplier = config.get('max_spread_multiplier', 1.5)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.max_slippage_pips = config.get('max_slippage_pips', 2.0)
        self.default_deviation = config.get('default_deviation', 10)
        self.default_magic = config.get('default_magic', 123456)  # Configurable magic number

        # Symbol specifications with dynamic updates from MT5
        self.symbol_specs = {
            'EURUSD': {'digits': 5, 'point': 0.00001, 'pip_multiplier': 10000, 'tick_size': 0.00001, 'min_volume': 0.01,
                       'max_volume': 100.0},
            'GBPJPY': {'digits': 3, 'point': 0.001, 'pip_multiplier': 100, 'tick_size': 0.001, 'min_volume': 0.01,
                       'max_volume': 100.0},
        }
        self.default_specs = {'digits': 5, 'point': 0.00001, 'pip_multiplier': 10000, 'tick_size': 0.00001,
                              'min_volume': 0.01, 'max_volume': 100.0}

        # Order templates with configurable parameters
        self.order_templates = {
            'market_buy': {
                "action": mt5.TRADE_ACTION_DEAL,
                "type": mt5.ORDER_TYPE_BUY,
                "type_time": mt5.ORDER_TIME_GTC,
                "magic": self.default_magic,
                "comment": config.get('order_comment', "Market Buy")
            },
            'market_sell': {
                "action": mt5.TRADE_ACTION_DEAL,
                "type": mt5.ORDER_TYPE_SELL,
                "type_time": mt5.ORDER_TIME_GTC,
                "magic": self.default_magic,
                "comment": config.get('order_comment', "Market Sell")
            },
            'limit_buy': {
                "action": mt5.TRADE_ACTION_PENDING,
                "type": mt5.ORDER_TYPE_BUY_LIMIT,
                "type_time": mt5.ORDER_TIME_GTC,
                "magic": self.default_magic,
                "comment": config.get('order_comment', "Limit Buy")
            },
            'limit_sell': {
                "action": mt5.TRADE_ACTION_PENDING,
                "type": mt5.ORDER_TYPE_SELL_LIMIT,
                "type_time": mt5.ORDER_TIME_GTC,
                "magic": self.default_magic,
                "comment": config.get('order_comment', "Limit Sell")
            }
        }

        # Trading session hours
        self.trading_hours = config.get('trading_hours', {
            'EURUSD': {'start': '00:00', 'end': '24:00'},
            'GBPJPY': {'start': '00:00', 'end': '24:00'}
        })

        # Initialize order history tracking
        self.order_history = {}

        self.logger.info("Trade Execution initialized")

    def execute_trade(self, signal):
        """
        Execute a trade based on the provided signal.

        Args:
            signal (dict): Trading signal with execution parameters

        Returns:
            dict or None: Trade result information or None if execution failed
        """
        self.logger.debug(f"Received signal for execution: {signal}")  # Added for debugging
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.warning("Cannot execute trade: Not in live mode")
            return None

        # Validate signal
        if not self._validate_signal(signal):
            return None

        # Extract signal parameters
        symbol = signal['symbol']
        action = signal['action']
        volume = signal['volume']
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')

        # Check if we're in trading hours for this symbol
        if not self._check_trading_hours(symbol):
            self.logger.warning(f"Outside trading hours for {symbol}, trade not executed")
            return None

        # Get current price
        current_price = self._get_market_price(symbol, action)
        if current_price is None:
            self.logger.error(f"Cannot get current price for {symbol}")
            return None

        # Check spread
        spread = self._check_spread(symbol)
        if spread is None:
            self.logger.warning(f"Cannot determine spread for {symbol}")
        elif spread > self._get_max_allowed_spread(symbol):
            self.logger.warning(f"Spread too high for {symbol}: {spread} pips > allowed maximum")
            return None

        # Prepare order request
        order_template = self.order_templates.get(f'market_{action.lower()}')
        if not order_template:
            self.logger.error(f"Unknown action: {action}")
            return None

        order = order_template.copy()
        order.update({
            "symbol": symbol,
            "volume": self._validate_volume(symbol, volume),
            "price": current_price,
            "deviation": self._get_deviation(symbol)
        })

        # Add stop loss and take profit if provided
        if stop_loss:
            order["sl"] = self._validate_price(symbol, stop_loss)
        if take_profit:
            order["tp"] = self._validate_price(symbol, take_profit)

        # Execute the order with retries
        order_result = None
        for attempt in range(self.max_retries):
            try:
                # Update price on retry (in case market moved)
                if attempt > 0:
                    current_price = self._get_market_price(symbol, action)
                    if current_price is None:
                        self.logger.error(f"Cannot get current price for {symbol} on retry {attempt + 1}")
                        continue
                    order["price"] = current_price

                # Send the order
                order_result = self.mt5_api.order_send(order)
                if order_result is None:
                    self.logger.warning(
                        f"Order send returned None for {symbol}, retrying ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                    continue

                # Check if order was successful
                if isinstance(order_result, dict):
                    if 'ticket' in order_result:
                        break  # Success with our MT5API wrapper
                elif hasattr(order_result, 'retcode') and order_result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Convert MT5 result to standard format
                    order_result = {
                        'ticket': order_result.order,
                        'entry_price': order_result.price,
                        'volume': order_result.volume
                    }
                    break  # Success with direct MT5 result

                # Handle specific error cases
                error_message = f"Order error: {mt5.last_error()}"
                if hasattr(order_result, 'retcode'):
                    if order_result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                        self.logger.warning(f"Requote for {symbol}, retrying ({attempt + 1}/{self.max_retries})")
                    elif order_result.retcode == mt5.TRADE_RETCODE_INVALID_PRICE:
                        self.logger.error(f"Invalid price for {symbol}: {current_price}")
                        return None
                    elif order_result.retcode == mt5.TRADE_RETCODE_INVALID_VOLUME:
                        self.logger.error(f"Invalid volume for {symbol}: {volume}")
                        return None
                    else:
                        self.logger.warning(f"Order failed for {symbol}: {order_result.retcode} - {error_message}")
                time.sleep(self.retry_delay)

            except Exception as e:
                self.logger.error(f"Exception executing trade for {symbol}: {str(e)}")
                time.sleep(self.retry_delay)

        # Check final result
        if order_result is None or 'ticket' not in order_result:
            self.logger.error(f"Trade execution failed for {symbol} after {self.max_retries} attempts")
            return None

        # Create standardized trade result
        trade_result = {
            "symbol": symbol,
            "action": action,
            "position_size": volume,
            "volume": volume,  # Include both for compatibility
            "entry_price": order_result.get('entry_price', current_price),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "profit_loss": 0.0,
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "ticket": order_result.get('ticket', 0),
            "spread_pips": spread,
            "signal_score": signal.get('score', 0)
        }

        # Record in order history
        self._record_order(trade_result)

        self.logger.info(
            f"Trade executed successfully: {action.upper()} {volume} {symbol} at {trade_result['entry_price']}")
        return trade_result

    def _validate_signal(self, signal):
        """
        Validate trading signal has all required fields.

        Args:
            signal (dict): Trading signal to validate

        Returns:
            bool: True if signal is valid, False otherwise
        """
        if not isinstance(signal, dict):
            self.logger.error("Signal must be a dictionary")
            return False

        required_fields = ['symbol', 'action', 'volume', 'entry_price']
        for field in required_fields:
            if field not in signal:
                self.logger.error(f"Missing required field in signal: {field}")
                return False

        # Validate action
        valid_actions = ['buy', 'sell']
        if signal['action'].lower() not in valid_actions:
            self.logger.error(f"Invalid action: {signal['action']}. Must be one of {valid_actions}")
            return False

        # Validate volume
        try:
            volume = float(signal['volume'])
            if volume <= 0:
                self.logger.error(f"Invalid volume: {volume}. Must be positive")
                return False
        except (ValueError, TypeError):
            self.logger.error(f"Invalid volume format: {signal['volume']}")
            return False

        return True

    def _get_symbol_specs(self, symbol):
        """
        Get specifications for a symbol, dynamically fetched from MT5.

        Args:
            symbol (str): Symbol to get specifications for

        Returns:
            dict: Symbol specifications
        """
        mt5_specs = self.mt5_api._get_symbol_info(symbol)
        if mt5_specs:
            return {
                'digits': mt5_specs.get('digits', self.symbol_specs.get(symbol, self.default_specs)['digits']),
                'point': mt5_specs.get('point', self.symbol_specs.get(symbol, self.default_specs)['point']),
                'pip_multiplier': 10000 if mt5_specs.get('digits', 5) == 5 else 100,
                'tick_size': mt5_specs.get('trade_tick_size',
                                           self.symbol_specs.get(symbol, self.default_specs)['tick_size']),
                'min_volume': mt5_specs.get('trade_volume_min',
                                            self.symbol_specs.get(symbol, self.default_specs)['min_volume']),
                'max_volume': mt5_specs.get('trade_volume_max',
                                            self.symbol_specs.get(symbol, self.default_specs)['max_volume']),
                'volume_step': mt5_specs.get('trade_volume_step', 0.01)
            }
        return self.symbol_specs.get(symbol, self.default_specs)

    def _get_market_price(self, symbol, action):
        """
        Get current market price for a symbol and action.

        Args:
            symbol (str): Symbol to get price for
            action (str): Action (buy or sell)

        Returns:
            float or None: Current price or None if unavailable
        """
        try:
            # Get tick data from MT5
            tick = self.mt5_api.get_tick_data(symbol)
            if tick is None:
                self.logger.error(f"Cannot get tick data for {symbol}")
                return None

            # Use appropriate price based on action
            if action.lower() == 'buy':
                return tick['ask']
            elif action.lower() == 'sell':
                return tick['bid']
            else:
                self.logger.error(f"Invalid action for price determination: {action}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting market price for {symbol}: {str(e)}")
            return None

    def _check_spread(self, symbol):
        """
        Check current spread for a symbol.

        Args:
            symbol (str): Symbol to check spread for

        Returns:
            float or None: Spread in pips or None if unavailable
        """
        try:
            # Get tick data from MT5
            tick = self.mt5_api.get_tick_data(symbol)
            if tick is None:
                return None

            # Calculate spread in pips
            specs = self._get_symbol_specs(symbol)
            pip_multiplier = specs['pip_multiplier']
            spread_points = tick['ask'] - tick['bid']
            spread_pips = spread_points * pip_multiplier

            return spread_pips
        except Exception as e:
            self.logger.error(f"Error checking spread for {symbol}: {str(e)}")
            return None

    def _get_max_allowed_spread(self, symbol):
        """
        Get maximum allowed spread for a symbol, dynamically adjusted by volatility.

        Args:
            symbol (str): Symbol to get max spread for

        Returns:
            float: Maximum allowed spread in pips
        """
        # Base typical spread
        typical_spread = {
            'EURUSD': 1.0,
            'GBPJPY': 2.0
        }.get(symbol, 3.0)

        # Adjust based on volatility (using ATR if available)
        tick = self.mt5_api.get_tick_data(symbol)
        atr = tick.get('atr', 0.001) if tick and 'atr' in tick else 0.001  # Fallback ATR
        volatility_factor = 1 + (atr * 100)  # Simple scaling based on ATR

        return typical_spread * self.max_spread_multiplier * volatility_factor

    def _validate_volume(self, symbol, volume):
        """
        Validate and normalize volume for a symbol using MT5-specific step sizes.

        Args:
            symbol (str): Symbol to validate volume for
            volume (float): Volume to validate

        Returns:
            float: Normalized volume
        """
        specs = self._get_symbol_specs(symbol)
        min_volume = specs['min_volume']
        max_volume = specs['max_volume']
        step_size = specs['volume_step']

        # Ensure volume is within allowed range
        volume = max(min_volume, min(max_volume, float(volume)))

        # Round to valid step size
        volume = round(volume / step_size) * step_size

        return volume

    def _validate_price(self, symbol, price):
        """
        Validate and normalize price for a symbol.

        Args:
            symbol (str): Symbol to validate price for
            price (float): Price to validate

        Returns:
            float: Normalized price
        """
        specs = self._get_symbol_specs(symbol)
        digits = specs['digits']
        tick_size = specs['tick_size']

        # Round to valid tick size
        price = round(price / tick_size) * tick_size

        # Format to correct number of digits
        price = round(price, digits)

        return price

    def _get_deviation(self, symbol):
        """
        Get appropriate price deviation for a symbol.

        Args:
            symbol (str): Symbol to get deviation for

        Returns:
            int: Price deviation in points
        """
        # Get from config or use default
        return self.config.get('deviation', {}).get(symbol, self.default_deviation)

    def _check_trading_hours(self, symbol):
        """
        Check if current time is within trading hours for a symbol.

        Args:
            symbol (str): Symbol to check trading hours for

        Returns:
            bool: True if within trading hours, False otherwise
        """
        # Default to 24/7 trading if no hours specified
        if symbol not in self.trading_hours:
            return True

        hours = self.trading_hours[symbol]
        start_time_str = hours.get('start', '00:00')
        end_time_str = hours.get('end', '24:00')

        # Handle special case for 24/7 trading
        if start_time_str == '00:00' and end_time_str == '24:00':
            return True

        try:
            # Parse trading hours
            start_hour, start_minute = map(int, start_time_str.split(':'))
            end_hour, end_minute = map(int, end_time_str.split(':'))

            # Get current time in UTC
            now = datetime.now(pytz.UTC)
            current_hour = now.hour
            current_minute = now.minute

            # Convert to minutes for easier comparison
            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute
            current_minutes = current_hour * 60 + current_minute

            # Check if current time is within trading hours
            if start_minutes <= end_minutes:
                # Simple case: start time is before end time
                return start_minutes <= current_minutes <= end_minutes
            else:
                # Complex case: trading hours span midnight
                return current_minutes >= start_minutes or current_minutes <= end_minutes
        except Exception as e:
            self.logger.error(f"Error checking trading hours for {symbol}: {str(e)}")
            return True  # Default to allowing trading on error

    def _record_order(self, trade_result):
        """
        Record an executed order in history.

        Args:
            trade_result (dict): Trade result information
        """
        ticket = trade_result.get('ticket')
        if not ticket:
            return

        self.order_history[str(ticket)] = trade_result

    def get_account_status(self):
        """
        Get current account status from MT5.

        Returns:
            dict: Account status information
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.debug("Account status not available in backtest mode")
            return None

        try:
            # Get account info
            account_info = self.mt5_api.get_account_info()
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', balance)
            margin = account_info.get('margin', 0)

            # Get open positions
            all_positions = self.mt5_api.positions_get()

            # Group positions by symbol
            positions_by_symbol = {}
            for pos in all_positions:
                symbol = pos.get('symbol')
                if symbol not in positions_by_symbol:
                    positions_by_symbol[symbol] = []
                pos_dict = {
                    'ticket': pos.get('ticket'),
                    'type': 'buy' if pos.get('type') == mt5.ORDER_TYPE_BUY else 'sell',
                    'volume': pos.get('volume'),
                    'price_open': pos.get('price_open'),
                    'profit': pos.get('profit'),
                    'sl': pos.get('sl'),
                    'tp': pos.get('tp')
                }
                positions_by_symbol[symbol].append(pos_dict)

            # Create summary by symbol
            symbols_summary = {}
            for symbol, positions in positions_by_symbol.items():
                buy_volume = sum(pos['volume'] for pos in positions if pos['type'] == 'buy')
                sell_volume = sum(pos['volume'] for pos in positions if pos['type'] == 'sell')
                net_volume = buy_volume - sell_volume
                total_profit = sum(pos['profit'] for pos in positions)

                symbols_summary[symbol] = {
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'net_volume': net_volume,
                    'count': len(positions),
                    'profit': total_profit
                }

            return {
                'balance': balance,
                'equity': equity,
                'margin': margin,
                'free_margin': equity - margin,
                'margin_level': (equity / margin * 100) if margin > 0 else 0,
                'positions': positions_by_symbol,
                'symbols_summary': symbols_summary,
                'total_positions': len(all_positions),
                'timestamp': datetime.now(pytz.UTC).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting account status: {str(e)}")
            return {
                'balance': 0,
                'equity': 0,
                'margin': 0,
                'free_margin': 0,
                'margin_level': 0,
                'positions': {},
                'symbols_summary': {},
                'total_positions': 0,
                'error': str(e),
                'timestamp': datetime.now(pytz.UTC).isoformat()
            }

    def close_position(self, position_id):
        """
        Close an open position.

        Args:
            position_id: Position ticket ID

        Returns:
            dict or None: Close result information or None if failed
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.warning(f"Cannot close position {position_id}: Not in live mode")
            return None

        try:
            # Validate position ID
            position_id = int(position_id)

            # Get position information
            positions = self.mt5_api.positions_get()
            position = None
            for pos in positions:
                if pos.get('ticket') == position_id:
                    position = pos
                    break

            if not position:
                self.logger.error(f"Position {position_id} not found")
                return None

            # Determine closing parameters
            symbol = position['symbol']
            volume = position['volume']
            position_type = 'buy' if position['type'] == mt5.ORDER_TYPE_BUY else 'sell'

            # Determine action (opposite of position type)
            action = 'sell' if position_type == 'buy' else 'buy'

            # Get current price
            price = self._get_market_price(symbol, action)
            if price is None:
                self.logger.error(f"Cannot get current price for {symbol}")
                return None

            # Create close request
            close_result = self.mt5_api.close_position(position_id)

            if close_result is None or not close_result.get('closed', False):
                self.logger.error(f"Failed to close position {position_id}")
                return None

            self.logger.info(f"Position {position_id} closed: {symbol} {volume} lots at {price}")

            # Create standardized result
            result = {
                'ticket': position_id,
                'symbol': symbol,
                'volume': volume,
                'close_price': price,
                'profit': close_result.get('profit', 0),
                'timestamp': datetime.now(pytz.UTC).isoformat()
            }

            return result
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {str(e)}")
            return None

    def modify_position(self, position_id, stop_loss=None, take_profit=None):
        """
        Modify stop loss and take profit for an open position.

        Args:
            position_id: Position ticket ID
            stop_loss (float, optional): New stop loss price
            take_profit (float, optional): New take profit price

        Returns:
            bool: True if modification successful, False otherwise
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.warning(f"Cannot modify position {position_id}: Not in live mode")
            return False

        try:
            # Validate position ID
            position_id = int(position_id)

            # Check if modification is needed
            if stop_loss is None and take_profit is None:
                self.logger.warning("No modification parameters provided")
                return False

            # Get position to validate symbol
            positions = self.mt5_api.positions_get()
            position = next((pos for pos in positions if pos.get('ticket') == position_id), None)
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return False

            symbol = position['symbol']

            # Normalize prices
            if stop_loss is not None:
                stop_loss = self._validate_price(symbol, stop_loss)
            if take_profit is not None:
                take_profit = self._validate_price(symbol, take_profit)

            # Send modification request
            result = self.mt5_api.modify_position(position_id, sl=stop_loss, tp=take_profit)

            if result:
                self.logger.info(f"Position {position_id} modified: SL={stop_loss}, TP={take_profit}")
                return True
            else:
                self.logger.error(f"Failed to modify position {position_id}")
                return False
        except Exception as e:
            self.logger.error(f"Error modifying position {position_id}: {str(e)}")
            return False

    def execute_backtest_trades(self, signals, historical_data, risk_manager):
        """
        Simulate trade execution for backtesting.

        Args:
            signals (dict): Dictionary of signals by symbol
            historical_data (dict): Dictionary of historical data by symbol and timeframe
            risk_manager: Risk management instance

        Returns:
            list: List of simulated trade results
        """
        if self.config['central_trading_bot']['mode'] == 'live':
            self.logger.warning("Cannot execute backtest trades in live mode")
            return []

        trade_results = []

        for symbol, signal in signals.items():
            if not signal:
                continue

            # Get appropriate historical data
            timeframe = signal.get('timeframe', '1h')
            if (symbol, timeframe) not in historical_data:
                self.logger.warning(f"No historical data for {symbol} on {timeframe}")
                continue

            data = historical_data[(symbol, timeframe)]
            if data is None or data.empty:
                continue

            # Use signal timestamp or latest data timestamp
            timestamp = signal.get('timestamp')
            if timestamp:
                trade_time = pd.to_datetime(timestamp)
            else:
                trade_time = data.index[-1]

            # Get entry price from signal or data
            entry_price = signal.get('entry_price')
            if entry_price is None:
                entry_price = data['close'].iloc[-1]

            # Get stop loss and take profit
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')

            # If no SL/TP in signal, calculate based on ATR
            if stop_loss is None or take_profit is None:
                atr = signal.get('atr')
                if atr is None and 'atr' in data.columns:
                    atr = data['atr'].iloc[-1]

                if atr:
                    # Calculate SL/TP based on ATR
                    specs = self._get_symbol_specs(symbol)
                    point = specs['point']
                    atr_multiple = 2.0

                    if stop_loss is None:
                        if signal['action'] == 'buy':
                            stop_loss = entry_price - (atr * atr_multiple)
                        else:
                            stop_loss = entry_price + (atr * atr_multiple)

                    if take_profit is None:
                        if signal['action'] == 'buy':
                            take_profit = entry_price + (atr * atr_multiple * 1.5)
                        else:
                            take_profit = entry_price - (atr * atr_multiple * 1.5)

            # Get volume
            volume = signal.get('volume', signal.get('position_size', 0.1))

            # Create trade result
            trade_result = {
                "symbol": symbol,
                "action": signal['action'],
                "position_size": volume,
                "volume": volume,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timestamp": trade_time.isoformat(),
                "ticket": f"sim-{uuid.uuid4().hex[:8]}",
                "score": signal.get('score', 0)
            }

            # Simulate trade outcome with limited data
            outcome, exit_price, exit_time, pips = self._simulate_trade_outcome(
                trade_result, data, timeframe)

            # Calculate profit/loss
            specs = self._get_symbol_specs(symbol)
            pip_value = specs['pip_multiplier']
            pip_value_per_lot = 10.0  # Approximate standard value
            profit_loss = pips * (pip_value_per_lot / pip_value) * volume

            # Apply commission
            commission = volume * risk_manager.commission_per_lot
            net_profit_loss = profit_loss - commission

            # Update trade result with outcome
            trade_result.update({
                "outcome": outcome,
                "exit_price": exit_price,
                "exit_time": exit_time.isoformat() if exit_time else None,
                "pips": pips,
                "profit_loss": net_profit_loss,
                "commission": commission
            })

            trade_results.append(trade_result)
            self.logger.info(f"Simulated backtest trade: {symbol} {signal['action']} - Outcome: {outcome}")

        return trade_results

    def _simulate_trade_outcome(self, trade, data, timeframe):
        """
        Simulate outcome of a backtest trade using historical data, limited to 20 bars.

        Args:
            trade (dict): Trade parameters
            data (pd.DataFrame): Historical price data
            timeframe (str): Timeframe string

        Returns:
            tuple: (outcome, exit_price, exit_time, pips)
        """
        try:
            # Get trade parameters
            symbol = trade['symbol']
            action = trade['action']
            entry_price = trade['entry_price']
            stop_loss = trade.get('stop_loss')
            take_profit = trade.get('take_profit')
            entry_time = pd.to_datetime(trade['timestamp'])

            # Limit future data to 20 bars for realism
            future_data = data[data.index > entry_time].head(20)

            # If no future data, use random outcome
            if future_data.empty:
                return self._random_outcome(entry_price, stop_loss, take_profit, action)

            # Initialize tracking variables
            hit_sl = False
            hit_tp = False
            exit_price = None
            exit_time = None

            # Track trade through future bars
            for idx, row in future_data.iterrows():
                high = row['high']
                low = row['low']
                close = row['close']

                # Check if SL or TP hit
                if action == 'buy':
                    if stop_loss and low <= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                        exit_time = idx
                        break
                    elif take_profit and high >= take_profit:
                        hit_tp = True
                        exit_price = take_profit
                        exit_time = idx
                        break
                else:  # 'sell'
                    if stop_loss and high >= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                        exit_time = idx
                        break
                    elif take_profit and low <= take_profit:
                        hit_tp = True
                        exit_price = take_profit
                        exit_time = idx
                        break

            # If trade didn't hit SL or TP, close at end of data
            if not hit_sl and not hit_tp:
                exit_price = future_data['close'].iloc[-1]
                exit_time = future_data.index[-1]

            # Calculate outcome
            if hit_sl:
                outcome = "stop_loss"
            elif hit_tp:
                outcome = "take_profit"
            else:
                outcome = "market_close"

            # Calculate pips gained/lost
            specs = self._get_symbol_specs(symbol)
            pip_multiplier = specs['pip_multiplier']

            if action == 'buy':
                pips = (exit_price - entry_price) * pip_multiplier
            else:  # 'sell'
                pips = (entry_price - exit_price) * pip_multiplier

            return outcome, exit_price, exit_time, pips
        except Exception as e:
            self.logger.error(f"Error simulating trade outcome: {str(e)}")
            return self._random_outcome(entry_price, stop_loss, take_profit, action)

    def _random_outcome(self, entry_price, stop_loss, take_profit, action):
        """
        Generate a random trade outcome when historical data is unavailable.

        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            action (str): Trade action ('buy' or 'sell')

        Returns:
            tuple: (outcome, exit_price, exit_time, pips)
        """
        # Default outcomes and probabilities
        outcomes = ['stop_loss', 'take_profit', 'market_close']
        probabilities = [0.35, 0.45, 0.2]  # 35% SL, 45% TP, 20% market close

        # Select outcome based on probabilities
        outcome = np.random.choice(outcomes, p=probabilities)

        # Determine exit price based on outcome
        if outcome == 'stop_loss':
            exit_price = stop_loss if stop_loss else (entry_price * 0.98 if action == 'buy' else entry_price * 1.02)
        elif outcome == 'take_profit':
            exit_price = take_profit if take_profit else (entry_price * 1.03 if action == 'buy' else entry_price * 0.97)
        else:  # market_close
            # Random price between entry and halfway to SL/TP
            if action == 'buy':
                min_price = stop_loss if stop_loss else entry_price * 0.99
                max_price = take_profit if take_profit else entry_price * 1.02
                exit_price = entry_price + (random.uniform(-0.5, 1.0) * (max_price - entry_price))
            else:
                min_price = take_profit if take_profit else entry_price * 0.98
                max_price = stop_loss if stop_loss else entry_price * 1.01
                exit_price = entry_price + (random.uniform(-1.0, 0.5) * (entry_price - min_price))

        # Calculate pips
        if action == 'buy':
            pips = (exit_price - entry_price) * 10000
        else:  # 'sell'
            pips = (entry_price - exit_price) * 10000

        # Random exit time (1-20 bars later)
        exit_time = datetime.now(pytz.UTC) + timedelta(hours=random.randint(1, 20))

        return outcome, exit_price, exit_time, pips

    def get_order_history(self, symbol=None, start_time=None, end_time=None):
        """
        Get history of executed orders with optional filtering.

        Args:
            symbol (str, optional): Filter by symbol
            start_time (datetime, optional): Start time for history
            end_time (datetime, optional): End time for history

        Returns:
            list: Filtered order history
        """
        # Set default time range if not specified
        if end_time is None:
            end_time = datetime.now(pytz.UTC)
        if start_time is None:
            start_time = end_time - timedelta(days=7)

        # Convert times to string format if needed
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()

        # Filter order history
        filtered_orders = []
        for ticket, order in self.order_history.items():
            # Check timestamp
            order_time = order.get('timestamp')
            if order_time:
                if order_time < start_time or order_time > end_time:
                    continue

            # Check symbol
            if symbol and order.get('symbol') != symbol:
                continue

            filtered_orders.append(order)

        return filtered_orders

    def get_pending_orders(self):
        """
        Get list of pending orders from MT5.

        Returns:
            list: Pending orders
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            return []

        try:
            # MT5API doesn't have a direct get_pending_orders; use positions_get and filter
            all_positions = self.mt5_api.positions_get()
            pending_orders = [pos for pos in all_positions if pos.get('type') in [
                mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT,
                mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP]]

            # Standardize order format
            standardized_orders = []
            for order in pending_orders:
                std_order = {
                    'ticket': order.get('ticket'),
                    'symbol': order.get('symbol'),
                    'type': self._order_type_to_string(order.get('type')),
                    'volume': order.get('volume_current', order.get('volume')),
                    'open_price': order.get('price_open'),
                    'sl': order.get('sl'),
                    'tp': order.get('tp'),
                    'comment': order.get('comment'),
                    'time_setup': datetime.fromtimestamp(order.get('time', 0),
                                                         tz=pytz.UTC).isoformat() if order.get('time') else None
                }
                standardized_orders.append(std_order)

            return standardized_orders
        except Exception as e:
            self.logger.error(f"Error getting pending orders: {str(e)}")
            return []

    def _order_type_to_string(self, order_type):
        """Convert MT5 order type to string."""
        order_types = {
            mt5.ORDER_TYPE_BUY_LIMIT: 'buy_limit',
            mt5.ORDER_TYPE_SELL_LIMIT: 'sell_limit',
            mt5.ORDER_TYPE_BUY_STOP: 'buy_stop',
            mt5.ORDER_TYPE_SELL_STOP: 'sell_stop'
        }
        return order_types.get(order_type, 'unknown')

    def cancel_order(self, order_id):
        """
        Cancel a pending order.

        Args:
            order_id: Order ticket ID

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.warning(f"Cannot cancel order {order_id}: Not in live mode")
            return False

        try:
            # Validate order ID
            order_id = int(order_id)

            # Send cancellation request
            result = self.mt5_api.order_close(order_id)  # Assuming MT5API has this method; adjust if needed
            if result:
                self.logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                self.logger.error(f"Failed to cancel order {order_id}")
                return False
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False

    def place_limit_order(self, symbol, action, price, volume, sl=None, tp=None, expiry=None):
        """
        Place a limit order.

        Args:
            symbol (str): Symbol to trade
            action (str): 'buy' or 'sell'
            price (float): Limit price
            volume (float): Trade volume in lots
            sl (float, optional): Stop loss price
            tp (float, optional): Take profit price
            expiry (datetime, optional): Order expiry time

        Returns:
            dict or None: Order result or None if failed
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.warning(f"Cannot place limit order: Not in live mode")
            return None

        try:
            # Validate parameters
            if action.lower() not in ['buy', 'sell']:
                self.logger.error(f"Invalid action: {action}")
                return None

            # Get order template
            template_key = f"limit_{action.lower()}"
            if template_key not in self.order_templates:
                self.logger.error(f"Unknown order template: {template_key}")
                return None

            order = self.order_templates[template_key].copy()

            # Set order parameters
            order.update({
                "symbol": symbol,
                "volume": self._validate_volume(symbol, volume),
                "price": self._validate_price(symbol, price),
                "deviation": self._get_deviation(symbol)
            })

            # Add SL/TP if provided
            if sl is not None:
                order["sl"] = self._validate_price(symbol, sl)
            if tp is not None:
                order["tp"] = self._validate_price(symbol, tp)

            # Add expiry if provided
            if expiry:
                if isinstance(expiry, datetime):
                    expiry_timestamp = int(expiry.timestamp())
                    order["type_time"] = mt5.ORDER_TIME_SPECIFIED
                    order["expiration"] = expiry_timestamp
                else:
                    self.logger.warning(f"Invalid expiry format, using GTC instead")

            # Send order
            result = self.mt5_api.order_send(order)

            if result and hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
                trade_result = {
                    'ticket': result.order,
                    'symbol': symbol,
                    'action': action,
                    'volume': volume,
                    'price': price,
                    'sl': sl,
                    'tp': tp,
                    'timestamp': datetime.now(pytz.UTC).isoformat()
                }
                self._record_order(trade_result)
                self.logger.info(f"Limit order placed for {symbol}: {action} {volume} lots at {price}")
                return trade_result
            else:
                self.logger.error(f"Failed to place limit order for {symbol}")
                return None
        except Exception as e:
            self.logger.error(f"Error placing limit order: {str(e)}")
            return None

    def get_closed_positions(self, symbol=None, start_time=None, end_time=None):
        """
        Get history of closed positions from MT5.

        Args:
            symbol (str, optional): Filter by symbol
            start_time (datetime, optional): Start time for history
            end_time (datetime, optional): End time for history

        Returns:
            list: Closed positions history
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            return []

        try:
            # Set default time range if not specified
            if end_time is None:
                end_time = datetime.now(pytz.UTC)
            if start_time is None:
                start_time = end_time - timedelta(days=7)

            # Get history from MT5
            history = self.mt5_api.get_historical_deals(from_date=start_time, to_date=end_time, symbol=symbol)

            # Standardize format
            standardized_history = []
            for deal in history:
                if deal.get('entry') == mt5.DEAL_ENTRY_OUT:  # Only closed positions
                    std_deal = {
                        'ticket': deal.get('ticket'),
                        'symbol': deal.get('symbol'),
                        'action': 'buy' if deal.get('type') == mt5.DEAL_TYPE_BUY else 'sell',
                        'volume': deal.get('volume'),
                        'entry_price': deal.get('price'),
                        'profit': deal.get('profit'),
                        'close_time': datetime.fromtimestamp(deal.get('time', 0),
                                                             tz=pytz.UTC).isoformat() if deal.get('time') else None
                    }
                    standardized_history.append(std_deal)

            return standardized_history
        except Exception as e:
            self.logger.error(f"Error getting closed positions: {str(e)}")
            return []