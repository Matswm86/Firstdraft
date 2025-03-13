import logging
from datetime import datetime
import pytz

class RiskManagement:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.max_drawdown = config['risk_management'].get('max_drawdown', 0.04)
        self.max_daily_loss = config['risk_management'].get('max_daily_loss', 0.02)
        self.max_profit_per_day = config['risk_management'].get('max_profit_per_day', 0.04)
        self.risk_per_trade = config['risk_management'].get('risk_per_trade', 0.01)
        self.max_trades_per_day = config['risk_management'].get('max_trades_per_day', 5)
        self.initial_balance = config['risk_management'].get('initial_balance', 100000)
        self.commission_per_lot = config['risk_management'].get('commission_per_lot', 7.0)
        self.slippage_pips = config['risk_management'].get('slippage_pips', {'EURUSD': 2.0, 'GBPJPY': 3.0})

        self.current_balance = self.initial_balance
        self.daily_loss = 0
        self.daily_profit = 0
        self.trades_today = 0
        self.last_reset_date = None

    def reset_daily_metrics(self, current_date):
        if self.last_reset_date is None or current_date > self.last_reset_date:
            self.daily_loss = 0
            self.daily_profit = 0
            self.trades_today = 0
            self.last_reset_date = current_date
            self.logger.info(f"Daily metrics reset for {current_date}")

    def check_risk_limits(self, account_status):
        balance = account_status['balance']
        equity = balance + sum(pos.profit for pos_list in account_status['positions'].values()
                               for pos in (pos_list or []))
        drawdown = max(0, (self.initial_balance - equity) / self.initial_balance)
        if drawdown >= self.max_drawdown:
            self.logger.warning(f"Risk limits exceeded: Max drawdown breached ({drawdown:.2%} >= {self.max_drawdown:.2%})")
            return False

        if self.daily_loss >= self.max_daily_loss * self.initial_balance:
            self.logger.warning(f"Max daily loss reached ({self.daily_loss:.2f} >= {self.max_daily_loss * self.initial_balance:.2f})")
            return False

        if self.trades_today >= self.max_trades_per_day:
            self.logger.warning("Max trades per day reached")
            return False

        if self.daily_profit >= self.max_profit_per_day * self.initial_balance:
            self.logger.info("Max daily profit reached; trading can continue or pause based on strategy")
        return True

    def evaluate_signal(self, signal, current_date, account_status):
        self.reset_daily_metrics(current_date)

        if not self.check_risk_limits(account_status):
            return None

        symbol = signal.get('symbol')
        entry_price = signal.get('entry_price')
        action = signal.get('action')
        atr = signal.get('atr', 0.001)  # Default to 10 pips if ATR not calculated

        if entry_price is None or action not in ['buy', 'sell'] or symbol is None:
            self.logger.error(f"Invalid signal for {symbol}: Missing entry_price, action, or symbol")
            return None

        point = 0.0001 if symbol == 'EURUSD' else 0.01
        stop_loss_pips = atr / point * 2  # 2x ATR for stop loss
        stop_loss = entry_price - (stop_loss_pips * point) if action == 'buy' else entry_price + (stop_loss_pips * point)

        max_risk_amount = self.current_balance * self.risk_per_trade  # e.g., $1000
        pip_value_per_lot = 10 if symbol == 'EURUSD' else 1000 / entry_price
        risk_per_unit = abs(entry_price - stop_loss)
        stop_loss_pips = risk_per_unit / point
        volume = max_risk_amount / (stop_loss_pips * pip_value_per_lot)
        volume = max(0.1, round(volume, 2))  # Minimum 0.1 lots

        if volume <= 0:
            self.logger.warning(f"Calculated volume is zero or negative for {symbol}")
            return None

        signal['volume'] = volume
        signal['stop_loss'] = stop_loss
        self.logger.info(f"Signal approved for {symbol}: Volume = {volume}, Stop Loss = {stop_loss}")
        return signal

    def update_pnl(self, trade_result, exit_price):
        symbol = trade_result['symbol']
        entry_price = trade_result['entry_price']
        volume = trade_result['volume']
        action = trade_result['action']

        point = 0.0001 if symbol == 'EURUSD' else 0.01
        profit_loss_pips = (exit_price - entry_price) / point if action == 'buy' else (entry_price - exit_price) / point
        pip_value_per_lot = 10 if symbol == 'EURUSD' else 1000 / exit_price
        profit_loss = profit_loss_pips * pip_value_per_lot * volume

        commission_paid = trade_result.get('commission_paid', volume * self.commission_per_lot)
        net_profit_loss = profit_loss - commission_paid

        if net_profit_loss > 0:
            self.daily_profit += net_profit_loss
        else:
            self.daily_loss += abs(net_profit_loss)

        self.current_balance += net_profit_loss
        self.logger.info(f"P&L updated for {symbol}: Net P&L = {net_profit_loss:.2f}, Balance = {self.current_balance:.2f}")
        return net_profit_loss