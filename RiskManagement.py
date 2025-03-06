import logging
from datetime import datetime, timedelta

class RiskManagement:
    def __init__(self, config, trade_execution=None):
        # Risk parameters from configuration
        self.max_drawdown = config.get('max_drawdown', 0.065)
        self.max_daily_loss = config.get('max_daily_loss', 0.02)
        self.max_profit_per_day = config.get('max_profit_per_day', 0.04)
        self.risk_per_trade = config.get('risk_per_trade', 0.01)
        self.max_trades_per_day = config.get('max_trades_per_day', 5)
        self.order_size_limit = config.get('order_size_limit', 100)
        self.contract_multiplier = config.get('contract_multiplier', 20)
        self.account_balance = config.get('initial_balance', 100000)
        self.trade_execution = trade_execution  # For live updates

        # Daily tracking variables
        self.daily_loss = 0
        self.daily_profit = 0
        self.trade_count = 0
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Logging setup
        self.logger = logging.getLogger(self.__class__.__name__)

    def reset_daily_counters(self):
        self.daily_loss = 0
        self.daily_profit = 0
        self.trade_count = 0
        self.logger.info("Daily counters reset.")

    def update_account_balance(self):
        """Fetch real-time account balance from TradeExecution."""
        if self.trade_execution:
            account_info = self.trade_execution.get_account_status()
            if account_info:
                self.account_balance = account_info.get('balance', self.account_balance)
                self.logger.info(f"Account balance updated to {self.account_balance}")
        else:
            self.logger.warning("No TradeExecution instance; balance not updated.")

    def calculate_position_size(self, signal):
        entry_price = signal.get('entry_price')
        atr = signal.get('atr', 1.0)
        stop_loss_distance = atr * 1.5  # customizable ATR multiplier

        # Calculate stop-loss price
        stop_loss_price = entry_price - stop_loss_distance if signal['action'] == 'buy' else entry_price + stop_loss_distance
        risk_per_contract = abs(entry_price - stop_loss_price) * self.contract_multiplier

        if risk_per_contract <= 0:
            self.logger.warning("Risk per contract is zero or negative.")
            return 0, stop_loss_price

        risk_per_trade_amount = self.account_balance * self.risk_per_trade
        position_size = risk_per_trade_amount / risk_per_contract

        if position_size > self.order_size_limit:
            position_size = self.order_size_limit
            self.logger.info("Order size capped by limit.")

        return position_size, stop_loss_price

    def check_risk_limits(self):
        if self.daily_loss >= self.max_daily_loss * self.account_balance:
            self.logger.warning("Maximum daily loss exceeded.")
            return False

        if self.daily_profit >= self.max_profit_per_day * self.account_balance:
            self.logger.warning("Maximum daily profit exceeded.")
            return False

        if self.trade_count >= self.max_trades_per_day:
            self.logger.warning("Maximum daily trade count reached.")
            return False

        return True

    def evaluate_signal(self, signal):
        if 'entry_price' not in signal or 'atr' not in signal:
            self.logger.warning("Signal lacks required data (entry_price or atr).")
            return None

        # Update balance and check limits
        self.update_account_balance()
        if not self.check_risk_limits():
            self.logger.info("Risk limits breached. Signal rejected.")
            return None

        position_size, stop_loss = self.calculate_position_size(signal)
        if position_size <= 0:
            self.logger.warning("Position size is zero or negative. Signal rejected.")
            return None

        entry_price = signal['entry_price']
        take_profit_distance = abs(entry_price - stop_loss) * 2  # 1:2 risk-reward
        take_profit = entry_price + take_profit_distance if signal['action'] == 'buy' else entry_price - take_profit_distance

        # Break-even logic
        current_price = signal.get('current_price', entry_price)
        break_even_price = entry_price + (take_profit_distance * 0.625) if signal['action'] == 'buy' else entry_price - (take_profit_distance * 0.625)
        break_even = (signal['action'] == 'buy' and current_price >= break_even_price) or \
                     (signal['action'] == 'sell' and current_price <= break_even_price)

        if break_even:
            self.logger.info("Break-even met; stop-loss adjusted to entry price.")
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
        self.daily_loss += self.risk_per_trade * self.account_balance
        return adjusted_signal

    def update_trade_result(self, pnl):
        if pnl >= 0:
            self.daily_profit += pnl
        else:
            self.daily_loss += abs(pnl)

        if self.daily_profit >= self.max_profit_per_day * self.account_balance:
            self.logger.info("Max daily profit reached, halting trading.")

# Example usage
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