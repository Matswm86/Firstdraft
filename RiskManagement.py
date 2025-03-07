import logging
from datetime import datetime, timedelta

class RiskManagement:
    def __init__(self, config, trade_execution=None):
        """
        Initialize RiskManagement with configuration and optional TradeExecution instance.

        Args:
            config (dict): Configuration dictionary with risk parameters.
            trade_execution (TradeExecution, optional): Instance for live account updates.
        """
        # Risk parameters from configuration
        self.max_drawdown = config.get('max_drawdown', 0.065)
        self.max_daily_loss = config.get('max_daily_loss', 0.02)
        self.max_profit_per_day = config.get('max_profit_per_day', 0.04)
        self.risk_per_trade = config.get('risk_per_trade', 0.01)
        self.max_trades_per_day = config.get('max_trades_per_day', 5)
        self.order_size_limit = config.get('order_size_limit', 100)
        self.contract_multiplier = config.get('contract_multiplier', 20)
        self.account_balance = config.get('initial_balance', 100000)
        self.trade_execution = trade_execution  # For live updates, None in backtest

        # Daily tracking variables
        self.daily_loss = 0
        self.daily_profit = 0
        self.trade_count = 0
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Logging setup
        self.logger = logging.getLogger(self.__class__.__name__)

    def reset_daily_counters(self):
        """
        Reset daily tracking variables at the start of a new trading day.
        """
        now = datetime.utcnow()
        if now >= self.daily_reset_time + timedelta(days=1):
            self.daily_loss = 0
            self.daily_profit = 0
            self.trade_count = 0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self.logger.info("Daily counters reset.")

    def update_account_balance(self):
        """
        Fetch real-time account balance from TradeExecution in live mode.

        Returns:
            bool: True if balance updated successfully, False otherwise.
        """
        if self.trade_execution:
            try:
                account_info = self.trade_execution.get_account_status()
                if account_info:
                    self.account_balance = account_info.get('balance', self.account_balance)
                    self.logger.debug(f"Account balance updated to {self.account_balance}")
                    return True
            except Exception as e:
                self.logger.error(f"Failed to update account balance: {e}")
        else:
            self.logger.debug("No TradeExecution instance; using static balance for backtest")
        return False

    def calculate_position_size(self, signal):
        """
        Calculate position size and stop-loss based on risk parameters and signal data.

        Args:
            signal (dict): Trading signal with 'entry_price', 'atr', etc.

        Returns:
            tuple: (position_size, stop_loss_price)
        """
        entry_price = signal.get('entry_price')
        atr = signal.get('atr', 1.0)  # Default to 1.0 if missing
        stop_loss_distance = atr * 1.5  # Customizable ATR multiplier

        # Calculate stop-loss price
        stop_loss_price = (entry_price - stop_loss_distance if signal['action'] == 'buy'
                         else entry_price + stop_loss_distance)
        risk_per_contract = abs(entry_price - stop_loss_price) * self.contract_multiplier

        if risk_per_contract <= 0:
            self.logger.warning("Risk per contract is zero or negative")
            return 0, stop_loss_price

        risk_per_trade_amount = self.account_balance * self.risk_per_trade
        position_size = risk_per_trade_amount / risk_per_contract

        if position_size > self.order_size_limit:
            position_size = self.order_size_limit
            self.logger.info(f"Position size capped at {self.order_size_limit}")

        return round(position_size), stop_loss_price

    def check_risk_limits(self):
        """
        Check if risk limits allow a new trade.

        Returns:
            bool: True if within limits, False if exceeded.
        """
        self.reset_daily_counters()  # Ensure counters are reset if day has passed

        if self.daily_loss >= self.max_daily_loss * self.account_balance:
            self.logger.warning("Maximum daily loss exceeded")
            return False

        if self.daily_profit >= self.max_profit_per_day * self.account_balance:
            self.logger.warning("Maximum daily profit exceeded")
            return False

        if self.trade_count >= self.max_trades_per_day:
            self.logger.warning("Maximum daily trade count reached")
            return False

        # Check drawdown (simplified as cumulative loss from initial balance)
        if (self.account_balance - self.daily_loss) < (1 - self.max_drawdown) * config.get('initial_balance', 100000):
            self.logger.warning("Maximum drawdown exceeded")
            return False

        return True

    def evaluate_signal(self, signal):
        """
        Evaluate a trading signal, adjust with risk parameters, and return the adjusted signal.

        Args:
            signal (dict): Trading signal with 'action', 'entry_price', 'atr', etc.

        Returns:
            dict or None: Adjusted signal if valid, None if rejected by risk limits.
        """
        if 'entry_price' not in signal or 'atr' not in signal:
            self.logger.warning("Signal lacks required data (entry_price or atr)")
            return None

        # Update balance and check limits
        self.update_account_balance()
        if not self.check_risk_limits():
            self.logger.info("Risk limits breached; signal rejected")
            return None

        position_size, stop_loss = self.calculate_position_size(signal)
        if position_size <= 0:
            self.logger.warning("Position size is zero or negative; signal rejected")
            return None

        entry_price = signal['entry_price']
        take_profit_distance = abs(entry_price - stop_loss) * 2  # 1:2 risk-reward
        take_profit = (entry_price + take_profit_distance if signal['action'] == 'buy'
                      else entry_price - take_profit_distance)

        # Break-even logic (simplified for backtest; live mode uses real-time price)
        current_price = signal.get('current_price', entry_price)
        break_even_price = (entry_price + take_profit_distance * 0.625 if signal['action'] == 'buy'
                           else entry_price - take_profit_distance * 0.625)
        break_even = ((signal['action'] == 'buy' and current_price >= break_even_price) or
                      (signal['action'] == 'sell' and current_price <= break_even_price))

        if break_even:
            self.logger.info("Break-even met; stop-loss adjusted to entry price")
            stop_loss = entry_price

        adjusted_signal = {
            'action': signal['action'],
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'break_even': break_even,
            'timestamp': datetime.utcnow()
        }

        self.trade_count += 1
        # Preemptive daily loss increment (adjusted later by actual P&L)
        self.daily_loss += self.risk_per_trade * self.account_balance
        self.logger.debug(f"Adjusted signal: {adjusted_signal}")
        return adjusted_signal

    def update_trade_result(self, pnl):
        """
        Update daily profit/loss counters based on trade outcome.

        Args:
            pnl (float): Profit/loss from the trade.
        """
        self.reset_daily_counters()  # Ensure counters are current

        # Adjust daily_loss preemption from evaluate_signal
        self.daily_loss -= self.risk_per_trade * self.account_balance  # Remove preemptive loss
        if pnl >= 0:
            self.daily_profit += pnl
            self.daily_loss = max(0, self.daily_loss - pnl)  # Reduce loss if over-counted
        else:
            self.daily_loss += abs(pnl)

        self.account_balance += pnl  # Update balance for backtest simulation
        self.logger.info(f"Trade result updated: P&L={pnl}, Balance={self.account_balance}")

        if self.daily_profit >= self.max_profit_per_day * self.account_balance:
            self.logger.info("Max daily profit reached; consider halting trading")
        if self.daily_loss >= self.max_daily_loss * self.account_balance:
            self.logger.info("Max daily loss reached; consider halting trading")

# Example usage (for testing, commented out)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "max_drawdown": 0.065,
        "max_daily_loss": 0.02,
        "max_profit_per_day": 0.04,
        "risk_per_trade": 0.01,
        "max_trades_per_day": 5,
        "order_size_limit": 100,
        "contract_multiplier": 20,
        "initial_balance": 100000
    }
    rm = RiskManagement(config)
    signal = {"action": "buy", "entry_price": 12000, "atr": 25, "current_price": 12030}
    evaluated_signal = rm.evaluate_signal(signal)
    print("Evaluated Signal:", evaluated_signal)
    if evaluated_signal:
        rm.update_trade_result(100)  # Example profit