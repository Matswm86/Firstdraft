import logging
from datetime import datetime, date, time, timedelta
import pytz
import json
import os


class RiskManagement:
    """
    Risk Management module for controlling trading risk parameters.
    Handles position sizing, risk limits, drawdown protection, and profit targets.
    """

    def __init__(self, config):
        """
        Initialize RiskManagement with configuration settings.

        Args:
            config (dict): Configuration dictionary with risk management settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self._validate_config()

        # Extract risk parameters with defaults
        self.max_drawdown = config['risk_management'].get('max_drawdown', 0.04)
        self.max_daily_loss = config['risk_management'].get('max_daily_loss', 0.02)
        self.max_profit_per_day = config['risk_management'].get('max_profit_per_day', 0.04)
        self.risk_per_trade = config['risk_management'].get('risk_per_trade', 0.01)
        self.max_trades_per_day = config['risk_management'].get('max_trades_per_day', 5)
        self.initial_balance = config['risk_management'].get('initial_balance', 100000.0)
        self.commission_per_lot = config['risk_management'].get('commission_per_lot', 7.0)

        # Set up slippage in points per symbol with dynamic capability
        self.slippage_points = {}
        slippage_pips = config['risk_management'].get('slippage_pips', {})
        for symbol, pips in slippage_pips.items():
            if symbol.startswith(('EUR', 'GBP', 'AUD', 'NZD')):
                self.slippage_points[symbol] = pips * 10  # 1 pip = 10 points for 5-digit brokers
            elif symbol.endswith(('JPY')):
                self.slippage_points[symbol] = pips  # 1 pip = 1 point for JPY pairs
            else:
                self.slippage_points[symbol] = pips * 10  # Default assumption

        # Set up default slippage
        self.default_slippage_points = 20  # 2 pips default

        # Symbol-specific point values and contract specifications
        self.symbol_specs = {
            'EURUSD': {'point': 0.00001, 'pip_value_per_lot': 10.0, 'min_lot': 0.01, 'lot_step': 0.01,
                       'contract_size': 100000},
            'GBPJPY': {'point': 0.001, 'pip_value_per_lot': 8.0, 'min_lot': 0.01, 'lot_step': 0.01,
                       'contract_size': 100000},
        }
        self.default_symbol_spec = {'point': 0.00001, 'pip_value_per_lot': 10.0, 'min_lot': 0.01, 'lot_step': 0.01,
                                    'contract_size': 100000}

        # Initialize tracking variables
        self.current_balance = self.initial_balance
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.trades_today = 0
        self.positions = {}  # Track all open positions by ticket
        self.last_reset_date = None
        self.trade_history = []  # Track all closed trades
        self._last_save = None  # For debounced state saving

        # Correlation settings
        self.max_correlation_exposure = config['risk_management'].get('max_correlation_exposure', 0.03)
        self.correlation_matrix = config['risk_management'].get('correlation_matrix', {})

        # Risk level modifiers
        self.current_risk_level = 1.0  # 1.0 = normal, <1.0 = reduced risk
        self.risk_level_expiry = None

        # Try to load state from file
        self._load_state()

        # Initialize daily metrics for today
        self.reset_daily_metrics(datetime.now(pytz.UTC).date())

        self.logger.info(
            "Risk Management initialized: max_drawdown={:.1%}, risk_per_trade={:.1%}, max_trades_per_day={}".format(
                self.max_drawdown, self.risk_per_trade, self.max_trades_per_day))

    def _validate_config(self):
        """Validate configuration parameters"""
        if 'risk_management' not in self.config:
            raise ValueError("Missing 'risk_management' section in configuration")

        risk_config = self.config['risk_management']
        required_params = ['max_drawdown', 'max_daily_loss', 'risk_per_trade', 'max_trades_per_day']
        for param in required_params:
            if param not in risk_config:
                self.logger.warning(f"Missing recommended risk parameter: {param}, using default")

    def reset_daily_metrics(self, current_date):
        """
        Reset daily trading metrics when a new day starts.

        Args:
            current_date (date): Current trading date
        """
        if isinstance(current_date, datetime):
            current_date = current_date.date()

        if self.last_reset_date is None or current_date > self.last_reset_date:
            if self.last_reset_date is not None:
                self._save_daily_metrics(self.last_reset_date)

            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.trades_today = 0
            self.last_reset_date = current_date

            self.logger.info(f"Daily metrics reset for {current_date}")
            self._debounced_save_state()

    def _save_daily_metrics(self, trade_date):
        """
        Save daily metrics to history.

        Args:
            trade_date (date): Trading date to save
        """
        daily_record = {
            'date': trade_date.isoformat(),
            'profit': self.daily_profit,
            'loss': self.daily_loss,
            'net': self.daily_profit - self.daily_loss,
            'trades': self.trades_today,
            'end_balance': self.current_balance
        }

        history_file = self.config['risk_management'].get('history_file', 'logs/risk_history.json')
        if history_file:
            try:
                os.makedirs(os.path.dirname(history_file), exist_ok=True)
                history = []
                if os.path.exists(history_file):
                    with open(history_file, 'r') as f:
                        try:
                            history = json.load(f)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse history file {history_file}, starting new history")
                history.append(daily_record)
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
                self.logger.info(f"Saved daily metrics for {trade_date} to {history_file}")
            except Exception as e:
                self.logger.error(f"Error saving daily metrics: {str(e)}")

    def _save_state(self):
        """Save current state to file for recovery with debouncing"""
        state_file = self.config['risk_management'].get('state_file', 'state/risk_state.json')
        if not state_file:
            return

        try:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            state = {
                'current_balance': self.current_balance,
                'daily_loss': self.daily_loss,
                'daily_profit': self.daily_profit,
                'trades_today': self.trades_today,
                'last_reset_date': self.last_reset_date.isoformat() if self.last_reset_date else None,
                'positions': self.positions,
                'current_risk_level': self.current_risk_level,
                'risk_level_expiry': self.risk_level_expiry.isoformat() if self.risk_level_expiry else None
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            self._last_save = datetime.now(pytz.UTC)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Saved risk management state to {state_file}")
        except Exception as e:
            self.logger.error(f"Error saving risk management state: {str(e)}")

    def _debounced_save_state(self):
        """Save state only if 5 seconds have passed since last save"""
        now = datetime.now(pytz.UTC)
        if not self._last_save or (now - self._last_save).total_seconds() > 5:
            self._save_state()

    def _load_state(self):
        """Load previously saved state if available"""
        state_file = self.config['risk_management'].get('state_file', 'state/risk_state.json')
        if not state_file or not os.path.exists(state_file):
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            self.current_balance = state.get('current_balance', self.initial_balance)
            self.daily_loss = state.get('daily_loss', 0.0)
            self.daily_profit = state.get('daily_profit', 0.0)
            self.trades_today = state.get('trades_today', 0)

            last_reset_date = state.get('last_reset_date')
            if last_reset_date:
                try:
                    self.last_reset_date = datetime.fromisoformat(last_reset_date).date()
                except (ValueError, TypeError):
                    self.last_reset_date = None

            positions = state.get('positions', {})
            if isinstance(positions, dict):
                self.positions = {str(k): v for k, v in positions.items()}  # Ensure ticket keys are strings

            self.current_risk_level = state.get('current_risk_level', 1.0)
            risk_level_expiry = state.get('risk_level_expiry')
            if risk_level_expiry:
                try:
                    self.risk_level_expiry = datetime.fromisoformat(risk_level_expiry)
                    if self.risk_level_expiry < datetime.now(pytz.UTC):
                        self.current_risk_level = 1.0
                        self.risk_level_expiry = None
                except (ValueError, TypeError):
                    self.risk_level_expiry = None

            self.logger.info(f"Loaded risk management state from {state_file}")
        except Exception as e:
            self.logger.error(f"Error loading risk management state: {str(e)}")

    def get_symbol_specs(self, symbol):
        """
        Get trading specifications for a symbol.

        Args:
            symbol (str): Symbol to get specs for

        Returns:
            dict: Symbol specifications including point value, pip value, etc.
        """
        return self.symbol_specs.get(symbol, self.default_symbol_spec)

    def get_slippage_points(self, symbol, atr=None):
        """
        Get dynamic slippage points for a symbol, adjusted by ATR if provided.

        Args:
            symbol (str): Symbol name
            atr (float, optional): Average True Range for volatility adjustment

        Returns:
            int: Slippage in points
        """
        base_slippage = self.slippage_points.get(symbol, self.default_slippage_points)
        if atr:
            point = self.get_symbol_specs(symbol)['point']
            dynamic_slippage = int(atr / point * 0.5)  # 50% of ATR in points
            return max(base_slippage, dynamic_slippage)
        return base_slippage

    def check_risk_limits(self, account_status):
        """
        Check if current risk exposure is within configured limits.

        Args:
            account_status (dict): Current account status from MT5API

        Returns:
            tuple: (allowed, reason) - whether trading is allowed and reason if not
        """
        try:
            if not isinstance(account_status, dict):
                return False, "Invalid account status format"

            balance = account_status.get('balance', 0.0)
            equity = account_status.get('equity', balance)
            positions = account_status.get('positions', {})

            if equity == 0:
                equity = self.current_balance

            drawdown = max(0, (self.initial_balance - equity) / self.initial_balance)
            if drawdown >= self.max_drawdown:
                message = f"Max drawdown breached: {drawdown:.2%} >= {self.max_drawdown:.2%}"
                self.logger.warning(message)
                return False, message

            if self.daily_loss >= self.max_daily_loss * self.initial_balance:
                message = f"Max daily loss reached: ${self.daily_loss:.2f} >= ${self.max_daily_loss * self.initial_balance:.2f}"
                self.logger.warning(message)
                return False, message

            if self.trades_today >= self.max_trades_per_day:
                message = f"Max trades per day reached: {self.trades_today} >= {self.max_trades_per_day}"
                self.logger.warning(message)
                return False, message

            if self.daily_profit >= self.max_profit_per_day * self.initial_balance:
                message = f"Max daily profit reached: ${self.daily_profit:.2f} >= ${self.max_profit_per_day * self.initial_balance:.2f}"
                self.logger.info(message)

            return True, "Risk limits within acceptable range"
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return False, f"Risk check error: {str(e)}"

    def evaluate_signal(self, signal, current_date, account_status):
        """
        Evaluate a trading signal against risk management rules.

        Args:
            signal (dict): Trading signal to evaluate
            current_date (date or datetime): Current trading date
            account_status (dict): Current account status from MT5API

        Returns:
            dict or None: Adjusted signal or None if rejected
        """
        try:
            if not isinstance(signal, dict) or not signal:
                self.logger.warning("Invalid signal format")
                return None

            self.reset_daily_metrics(current_date.date() if isinstance(current_date, datetime) else current_date)

            risk_allowed, reason = self.check_risk_limits(account_status)
            if not risk_allowed:
                self.logger.info(f"Signal rejected due to risk limits: {reason}")
                return None

            symbol = signal.get('symbol')
            entry_price = signal.get('entry_price')
            action = signal.get('action')
            atr = signal.get('atr', 0.001)

            if not all([symbol, entry_price, action]):
                self.logger.error(
                    f"Missing required signal parameters: symbol={symbol}, entry_price={entry_price}, action={action}")
                return None

            if action not in ['buy', 'sell']:
                self.logger.error(f"Invalid action in signal: {action}")
                return None

            symbol_spec = self.get_symbol_specs(symbol)
            point = symbol_spec['point']
            pip_value_per_lot = symbol_spec['pip_value_per_lot']
            min_lot = symbol_spec['min_lot']
            lot_step = symbol_spec['lot_step']

            atr_multiple = self.config['risk_management'].get('atr_multiple_for_sl', 2.0)
            atr_multiple_tp = self.config['risk_management'].get('atr_multiple_for_tp', 1.5)

            stop_loss_points = int(atr / point * atr_multiple)
            min_sl_points = {'EURUSD': 50, 'GBPJPY': 50}.get(symbol, 50)
            stop_loss_points = max(stop_loss_points, min_sl_points)

            if action == 'buy':
                stop_loss = entry_price - (stop_loss_points * point)
                take_profit = entry_price + (stop_loss_points * point * atr_multiple_tp / atr_multiple)
            else:  # sell
                stop_loss = entry_price + (stop_loss_points * point)
                take_profit = entry_price - (stop_loss_points * point * atr_multiple_tp / atr_multiple)

            adjusted_risk = self.risk_per_trade * self.check_risk_level()  # Check and update risk level
            correlated_exposure = self._calculate_correlated_exposure(symbol)
            if correlated_exposure > 0:
                correlation_factor = max(0, 1 - (correlated_exposure / self.max_correlation_exposure))
                adjusted_risk *= correlation_factor
                self.logger.info(f"Reduced position size due to correlation: factor={correlation_factor:.2f}")

            risk_amount = self.current_balance * adjusted_risk
            risk_per_pip = risk_amount / (stop_loss_points / 10)
            volume = risk_per_pip / pip_value_per_lot
            volume = max(min_lot, round(volume / lot_step) * lot_step)
            max_volume = self.config['risk_management'].get('max_position_size', 10.0)
            volume = min(volume, max_volume)

            adjusted_signal = signal.copy()
            adjusted_signal['volume'] = volume
            adjusted_signal['stop_loss'] = stop_loss
            adjusted_signal['take_profit'] = take_profit
            adjusted_signal['risk_amount'] = risk_amount
            adjusted_signal['risk_per_pip'] = risk_per_pip

            self.logger.info(
                f"Signal evaluated for {symbol}: Volume={volume:.2f}, SL={stop_loss:.5f}, "
                f"TP={take_profit:.5f}, Risk=${risk_amount:.2f}"
            )

            return adjusted_signal
        except Exception as e:
            self.logger.error(f"Error evaluating signal: {str(e)}")
            return None

    def _calculate_correlated_exposure(self, new_symbol):
        """
        Calculate exposure to correlated assets using covariance-like weighting.

        Args:
            new_symbol (str): Symbol for potential new position

        Returns:
            float: Current exposure to correlated assets (0-1 scale)
        """
        try:
            if not self.positions or not self.correlation_matrix:
                same_symbol_count = sum(1 for pos in self.positions.values() if pos['symbol'] == new_symbol)
                return min(0.5, same_symbol_count * 0.25)

            total_exposure = 0.0
            for ticket, position in self.positions.items():
                if 'volume' not in position:
                    continue
                symbol = position['symbol']
                correlation_key = f"{new_symbol}_{symbol}"
                reverse_key = f"{symbol}_{new_symbol}"
                correlation = self.correlation_matrix.get(correlation_key,
                                                          self.correlation_matrix.get(reverse_key, 0.0))
                if abs(correlation) > 0.2:
                    position_exposure = position['volume'] / 10.0
                    total_exposure += position_exposure * abs(correlation)

            return min(1.0, total_exposure)
        except Exception as e:
            self.logger.error(f"Error calculating correlated exposure: {str(e)}")
            return 0.25

    def update_pnl(self, trade_result, exit_price):
        """
        Update profit/loss tracking when a position is closed.

        Args:
            trade_result (dict): Trade information
            exit_price (float): Exit price

        Returns:
            float: Net profit/loss amount
        """
        try:
            if not isinstance(trade_result, dict) or not trade_result:
                self.logger.error("Invalid trade result format")
                return 0.0

            symbol = trade_result.get('symbol')
            entry_price = trade_result.get('entry_price')
            volume = trade_result.get('volume', trade_result.get('position_size', 0))
            action = trade_result.get('action')

            if not all([symbol, entry_price, volume, action, exit_price]):
                self.logger.error(
                    f"Missing required trade data: symbol={symbol}, entry={entry_price}, volume={volume}, action={action}, exit={exit_price}")
                return 0.0

            symbol_spec = self.get_symbol_specs(symbol)
            point = symbol_spec['point']
            pip_value_per_lot = symbol_spec['pip_value_per_lot']

            if action == 'buy':
                profit_loss_points = (exit_price - entry_price) / point
            else:  # sell
                profit_loss_points = (entry_price - exit_price) / point

            profit_loss_pips = profit_loss_points / 10
            profit_loss = profit_loss_pips * pip_value_per_lot * volume

            commission_paid = volume * self.commission_per_lot
            slippage_points = self.get_slippage_points(symbol, trade_result.get('atr'))
            slippage_cost = (slippage_points / 10) * pip_value_per_lot * volume

            net_profit_loss = profit_loss - commission_paid - slippage_cost

            if net_profit_loss > 0:
                self.daily_profit += net_profit_loss
            else:
                self.daily_loss += abs(net_profit_loss)

            self.current_balance += net_profit_loss

            ticket = str(trade_result.get('ticket', ''))
            if ticket in self.positions:
                del self.positions[ticket]

            trade_record = {
                'symbol': symbol,
                'action': action,
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'volume': float(volume),
                'profit_loss': float(net_profit_loss),
                'commission': float(commission_paid),
                'slippage': float(slippage_cost),
                'points': float(profit_loss_points),
                'open_time': trade_result.get('timestamp'),
                'close_time': datetime.now(pytz.UTC).isoformat(),
                'ticket': ticket
            }

            self.trade_history.append(trade_record)

            self.logger.info(
                f"Trade closed for {symbol}: {action.upper()}, {volume} lots, "
                f"Entry={entry_price:.5f}, Exit={exit_price:.5f}, "
                f"P&L=${net_profit_loss:.2f}, Points={profit_loss_points:.1f}"
            )

            self._debounced_save_state()

            return net_profit_loss
        except Exception as e:
            self.logger.error(f"Error updating P&L: {str(e)}")
            return 0.0

    def track_position(self, position_info):
        """
        Track a new position for risk management using ticket.

        Args:
            position_info (dict): Position information

        Returns:
            bool: True if position successfully tracked
        """
        try:
            if not isinstance(position_info, dict) or not position_info:
                self.logger.error("Invalid position information format")
                return False

            ticket = str(position_info.get('ticket', ''))
            if not ticket:
                self.logger.error("Missing ticket in position information")
                return False

            # Ensure position info has required fields
            required_fields = ['symbol', 'action', 'entry_price', 'volume']
            for field in required_fields:
                if field not in position_info:
                    self.logger.error(f"Missing required field in position info: {field}")
                    return False

            self.positions[ticket] = {
                'symbol': position_info['symbol'],
                'action': position_info['action'],
                'entry_price': float(position_info['entry_price']),
                'volume': float(position_info['volume']),
                'stop_loss': position_info.get('stop_loss'),
                'take_profit': position_info.get('take_profit'),
                'timestamp': position_info.get('timestamp', datetime.now(pytz.UTC).isoformat())
            }
            self.trades_today += 1
            self._debounced_save_state()
            self.logger.info(f"Now tracking position {ticket} for {position_info['symbol']}")
            return True
        except Exception as e:
            self.logger.error(f"Error tracking position: {str(e)}")
            return False

    def set_risk_level(self, level, duration_minutes=60):
        """
        Set a temporary risk level modifier.

        Args:
            level (float): Risk level multiplier (0-1 range, where 1 is normal risk)
            duration_minutes (int): How long this level should remain active

        Returns:
            bool: True if risk level was set
        """
        try:
            level = max(0.1, min(1.0, float(level)))
            duration_minutes = int(duration_minutes)
            if duration_minutes <= 0:
                self.logger.error("Duration must be positive")
                return False

            self.current_risk_level = level
            self.risk_level_expiry = datetime.now(pytz.UTC) + timedelta(minutes=duration_minutes)
            self._debounced_save_state()
            self.logger.info(f"Risk level set to {level:.2f} for {duration_minutes} minutes")
            return True
        except Exception as e:
            self.logger.error(f"Error setting risk level: {str(e)}")
            return False

    def check_risk_level(self):
        """
        Check current risk level and reset if expired.

        Returns:
            float: Current risk level
        """
        try:
            now = datetime.now(pytz.UTC)
            if self.risk_level_expiry and now > self.risk_level_expiry:
                self.current_risk_level = 1.0
                self.risk_level_expiry = None
                self.logger.info("Risk level reset to normal (1.0)")
                self._debounced_save_state()
            return self.current_risk_level
        except Exception as e:
            self.logger.error(f"Error checking risk level: {str(e)}")
            return 1.0

    def get_status(self):
        """
        Get current risk management status.

        Returns:
            dict: Risk management status information
        """
        try:
            return {
                'current_balance': float(self.current_balance),
                'initial_balance': float(self.initial_balance),
                'daily_profit': float(self.daily_profit),
                'daily_loss': float(self.daily_loss),
                'trades_today': int(self.trades_today),
                'max_trades_per_day': int(self.max_trades_per_day),
                'risk_per_trade': float(self.risk_per_trade),
                'current_risk_level': float(self.current_risk_level),
                'risk_level_expiry': self.risk_level_expiry.isoformat() if self.risk_level_expiry else None,
                'active_positions': len(self.positions),
                'drawdown': float(
                    (self.initial_balance - self.current_balance) / self.initial_balance if self.initial_balance > 0 else 0.0)
            }
        except Exception as e:
            self.logger.error(f"Error generating status: {str(e)}")
            return {}