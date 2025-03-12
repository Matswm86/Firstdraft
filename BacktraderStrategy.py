import backtrader as bt
import logging
from datetime import datetime


class BacktraderStrategy(bt.Strategy):
    """
    Backtrader strategy for backtesting NASDAQ futures trading signals.
    Note: This remains unchanged for backtesting and does not apply to live MT5 trading with The 5%ers.
    """
    params = (
        ('size', 1),  # Default position size for futures contracts
        ('stop_loss', 50),  # Stop loss in points
        ('take_profit', 100),  # Take profit in points
    )

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.order = None  # Track active order
        self.trade_count = 0
        self.daily_profit = 0
        self.daily_loss = 0
        self.last_day = None

        # Indicators
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=10)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=20)

    def log(self, txt, dt=None):
        """Log message with timestamp."""
        dt = dt or self.datas[0].datetime.datetime(0)
        self.logger.info(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}")
            self.trade_count += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")

        self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications and update P&L."""
        if trade.isclosed:
            profit = trade.pnlcomm  # Profit/loss including commission
            if profit > 0:
                self.daily_profit += profit
            else:
                self.daily_loss += abs(profit)
            self.log(f"Trade Closed, P&L: {profit:.2f}, Daily Profit: {self.daily_profit:.2f}, "
                     f"Daily Loss: {self.daily_loss:.2f}")

    def next(self):
        """Execute strategy logic for each bar (15m timeframe for backtesting)."""
        # Reset daily metrics at the start of a new day
        current_day = self.datas[0].datetime.date(0)
        if self.last_day is None or current_day > self.last_day:
            self.daily_profit = 0
            self.daily_loss = 0
            self.last_day = current_day

        # Skip if an order is pending
        if self.order:
            return

        # Check position and generate signal (simple SMA crossover)
        if not self.position:  # No open position
            if self.sma_short[0] > self.sma_long[0] and self.sma_short[-1] <= self.sma_long[-1]:
                # Buy signal
                self.order = self.buy(size=self.params.size)
                self.log(f"BUY ORDER PLACED, Price: {self.data.close[0]:.2f}")
                # Set stop loss and take profit
                self.sell(exectype=bt.Order.Stop, price=self.data.close[0] - self.params.stop_loss,
                          size=self.params.size)
                self.sell(exectype=bt.Order.Limit, price=self.data.close[0] + self.params.take_profit,
                          size=self.params.size)

            elif self.sma_short[0] < self.sma_long[0] and self.sma_short[-1] >= self.sma_long[-1]:
                # Sell signal
                self.order = self.sell(size=self.params.size)
                self.log(f"SELL ORDER PLACED, Price: {self.data.close[0]:.2f}")
                # Set stop loss and take profit
                self.buy(exectype=bt.Order.Stop, price=self.data.close[0] + self.params.stop_loss,
                         size=self.params.size)
                self.buy(exectype=bt.Order.Limit, price=self.data.close[0] - self.params.take_profit,
                         size=self.params.size)

    def stop(self):
        """Log final results when backtest completes."""
        self.log(f"Backtest Completed - Trade Count: {self.trade_count}, "
                 f"Final Daily Profit: {self.daily_profit:.2f}, Final Daily Loss: {self.daily_loss:.2f}")


# Example usage (for testing)
if __name__ == "__main__":
    import backtrader as bt
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    # Create a cerebro instance
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BacktraderStrategy)

    # Load sample data (replace with your actual CSV path)
    data_path = "C:\\Users\\matsw\\PycharmProjects\\Firstdraft\\data\\backtrader_15m.csv"
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    data = bt.feeds.PandasData(dataname=df)

    # Add data to cerebro
    cerebro.adddata(data)

    # Set initial cash and commission
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)  # Example commission

    # Run the backtest
    cerebro.run()
    cerebro.plot()