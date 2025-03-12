import logging
from datetime import datetime


class RiskManagement:
    def __init__(self, config):
        """
        Initialize RiskManagement with configuration settings for The 5%ers MT5 trading.

        Args:
            config (dict): Configuration dictionary with risk management settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Risk parameters from config
        self.max_drawdown = config['risk_management'].get('max_drawdown', 0.04)  # 4% max drawdown per The 5%ers
        self.max_daily_loss = config['risk_management'].get('max_daily_loss', 0.02)  # 2% daily loss limit
        self.max_profit_per_day = config['risk_management'].get('max_profit_per_day', 0.04)  # 4% daily profit target
        self.risk_per_trade = config['risk_management'].get('risk_per_trade', 0.01)  # 1% risk per trade
        self.max_trades_per_day = config['risk_management'].get('max_trades_per_day', 5)  # Max 5 trades/day
        self.initial_balance = config['risk_management'].get('initial_balance', 100000)  # Starting capital
        self.commission_per_lot = config['risk_management'].get('commission_per_lot', 7.0)  # $7 per lot
        self.slippage_pips = config['risk_management'].get('slippage_pips', {'EURUSD': 2.0, 'GBPJPY': 3.0})  # Pips

        # Tracking variables
        self.current_balance = self.initial_balance
        self.daily_loss = 0
        self.daily_profit = 0
        self.trades_today = 0
        self.last_reset_date = None

    def reset_daily_metrics(self, current_date):
        """
        Reset daily metrics at the start of a new trading day.

        Args:
            current_date (date): The current trading day's date.
        """
        if self.last_reset_date is None or current_date > self.last_reset_date:
            self.daily_loss = 0
            self.daily_profit = 0
            self.trades_today = 0
            self.last_reset_date = current_date
            self.logger.info(f"Daily metrics reset for {current_date}")

    def check_risk_limits(self, account_status):
        """
        Check if current metrics comply with risk limits.

        Args:
            account_status (dict): Current account balance and positions.

        Returns:
            bool: True if within limits, False otherwise.
        """
        # Check max drawdown
        balance = account_status['balance']
        equity = balance + sum(pos.profit for pos_list in account_status['positions'].values()
                               for pos in (pos_list or []))
        drawdown = max(0, (self.initial_balance - equity) / self.initial_balance)
        if drawdown >= self.max_drawdown:
            self.logger.warning(
                f"Risk limits exceeded: Max drawdown breached ({drawdown:.2%} >= {self.max_drawdown:.2%})")
            return False

        # Check daily loss limit
        if self.daily_loss >= self.max_daily_loss * self.initial_balance:
            self.logger.warning(
                f"Max daily loss reached ({self.daily_loss:.2f} >= {self.max_daily_loss * self.initial_balance:.2f})")
            return False

        # Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            self.logger.warning("Max trades per day reached")
            return False

        # Check daily profit target (optional action)
        if self.daily_profit >= self.max_profit_per_day * self.initial_balance:
            self.logger.info("Max daily profit reached; trading can continue or pause based on strategy")
            # Here, we allow continuation, but this can be adjusted

        return True

    def evaluate_signal(self, signal, current_date, account_status):
        """
        Evaluate a trading signal against risk criteria and calculate order size.

        Args:
            signal (dict): Contains 'entry_price', 'stop_loss', 'action', and 'symbol'.
            current_date (date): Current trading day's date.
            account_status (dict): Current account balance and positions.

        Returns:
            dict or None: Updated signal with volume, or None if invalid.
        """
        self.reset_daily_metrics(current_date)

        # Check trade count and risk limits
        if not self.check_risk_limits(account_status):
            return None

        # Extract signal details
        symbol = signal.get('symbol')
        entry_price = signal.get('entry_price')
        stop_loss = signal.get('stop_loss')
        action = signal.get('action')

        if stop_loss is None or entry_price is None or action not in ['buy', 'sell']:
            self.logger.error(f"Invalid signal for {symbol}: Missing entry_price, stop_loss, or action")
            return None

        # Calculate risk per unit (in price terms)
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            self.logger.error(f"Invalid stop-loss for {symbol}: Same as entry price")
            return None

        # Calculate dynamic order size (in lots)
        max_risk_amount = self.current_balance * self.risk_per_trade
        point = 0.0001 if symbol == 'EURUSD' else 0.01  # Pip size per The 5%ers asset specs
        stop_loss_pips = risk_per_unit / point
        pip_value_per_lot = 10 if symbol == 'EURUSD' else 1000 / entry_price  # Approx for GBPJPY
        volume = max_risk_amount / (stop_loss_pips * pip_value_per_lot)
        volume = round(volume, 2)  # MT5 lot size precision

        if volume <= 0:
            self.logger.warning(f"Calculated volume is zero or negative for {symbol}")
            return None

        # Update signal with volume
        signal['volume'] = volume
        self.trades_today += 1
        self.logger.info(f"Signal approved for {symbol}: Volume = {volume}")
        return signal

    def update_pnl(self, trade_result, exit_price):
        """
        Update P&L based on the result of a closed trade.

        Args:
            trade_result (dict): Contains 'profit_loss' (initially 0), 'commission_paid', 'entry_price', 'volume', etc.
            exit_price (float): Price at which the trade was closed.

        Returns:
            float: Net profit/loss after commission.
        """
        symbol = trade_result['symbol']
        entry_price = trade_result['entry_price']
        volume = trade_result['volume']  # In lots
        action = trade_result['action']

        # Calculate profit/loss in pips
        point = 0.0001 if symbol == 'EURUSD' else 0.01  # Pip size per The 5%ers asset specs
        profit_loss_pips = (exit_price - entry_price) / point if action == 'buy' else (entry_price - exit_price) / point

        # Convert to USD (pip value per lot: 10 for EURUSD, dynamic for GBPJPY)
        pip_value_per_lot = 10 if symbol == 'EURUSD' else 1000 / exit_price  # Approx for GBPJPY
        profit_loss = profit_loss_pips * pip_value_per_lot * volume

        # Subtract commission
        commission_paid = trade_result.get('commission_paid', volume * self.commission_per_lot)
        net_profit_loss = profit_loss - commission_paid

        # Update daily metrics
        if net_profit_loss > 0:
            self.daily_profit += net_profit_loss
        else:
            self.daily_loss += abs(net_profit_loss)

        # Update balance
        self.current_balance += net_profit_loss
        self.logger.info(
            f"P&L updated for {symbol}: Net P&L = {net_profit_loss:.2f}, Balance = {self.current_balance:.2f}")
        return net_profit_loss


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "risk_management": {
            "max_drawdown": 0.04,
            "max_daily_loss": 0.02,
            "max_profit_per_day": 0.04,
            "risk_per_trade": 0.01,
            "max_trades_per_day": 5,
            "initial_balance": 100000,
            "commission_per_lot": 7.0,
            "slippage_pips": {"EURUSD": 2.0, "GBPJPY": 3.0}
        }
    }
    rm = RiskManagement(config)
    signal = {
        "symbol": "EURUSD",
        "entry_price": 1.0900,
        "stop_loss": 1.0890,
        "action": "buy"
    }
    account_status = {"balance": 100000, "positions": {"EURUSD": []}}
    adjusted_signal = rm.evaluate_signal(signal, datetime.utcnow().date(), account_status)
    print(adjusted_signal)
    trade_result = {"symbol": "EURUSD", "entry_price": 1.0900, "volume": 0.1, "commission_paid": 0.7, "action": "buy"}
    profit_loss = rm.update_pnl(trade_result, 1.0920)
    print(f"Profit/Loss: {profit_loss}")