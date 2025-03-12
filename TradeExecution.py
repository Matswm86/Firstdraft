import logging
from datetime import datetime
from MT5API import MT5API


class TradeExecution:
    def __init__(self, config):
        """
        Initialize TradeExecution with configuration for The 5%ers MT5 trading.

        Args:
            config (dict): Configuration dictionary with 'order_type' and other settings.
        """
        self.order_type = config.get('order_type', 'market')  # Default to 'market'
        self.config = config
        self.mt5_api = MT5API(
            config['mt5_settings']['server'],
            config['mt5_settings']['login'],
            config['mt5_settings']['password']
        )
        self.logger = logging.getLogger(__name__)

    def execute_trade(self, signal):
        """
        Send trade order to MT5 for live trading with The 5%ers.

        Args:
            signal (dict): Trading signal with 'action', 'volume', 'stop_loss', 'take_profit', etc.

        Returns:
            dict or None: Trade result if successful, None if failed.
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.error("Cannot execute trade: Not in live mode (MT5 not used in backtest)")
            return None

        symbol = signal['symbol']
        action = signal['action']
        volume = signal['volume']  # In lots, calculated by RiskManagement
        entry_price = signal['entry_price']
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')

        # Prepare MT5 order request
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": stop_loss if stop_loss else 0.0,
            "tp": take_profit if take_profit else 0.0,
            "deviation": 10,  # Max deviation in points
            "magic": 123456,  # Unique identifier for the order
            "comment": "Automated Trade",
            "type_time": mt5.ORDER_TIME_GTC,  # Good Till Cancelled
            "type_filling": mt5.ORDER_FILLING_IOC  # Immediate or Cancel
        }

        try:
            response = self.mt5_api.order_send(order)
            if response:
                trade_result = {
                    "symbol": symbol,
                    "action": action,
                    "position_size": volume,  # Kept as 'position_size' for consistency with backtest
                    "entry_price": response['entry_price'],
                    "profit_loss": 0.0,  # Initial P&L, updated later when closed
                    "timestamp": datetime.utcnow().isoformat(),
                    "order_id": response['order_id']
                }
                self.logger.info(f"Trade executed successfully: {trade_result}")
                return trade_result
            self.logger.error(f"Trade failed: Invalid response {response}")
            return None
        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
            return None

    def get_account_status(self):
        """
        Fetch account status from MT5 for live trading.

        Returns:
            dict or None: Account status or None if failed or in backtest mode.
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.debug("Account status not available in backtest mode")
            return None

        try:
            balance = self.mt5_api.account_balance()
            positions = self.mt5_api.positions_get()
            if balance is not None:
                self.logger.debug(f"Account status retrieved: Balance = {balance}, Positions = {len(positions)}")
                return {'balance': balance, 'positions': {pos.symbol: [pos] for pos in positions}}
            self.logger.error("Failed to retrieve account status: Empty response")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get account status: {str(e)}")
            return None

    def close_position(self, position_id):
        """
        Close a specific position via MT5 in live trading mode.

        Args:
            position_id (str): Identifier of the position to close.

        Returns:
            dict or None: Response if successful, None if failed or in backtest mode.
        """
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.error(f"Cannot close position {position_id}: Not in live mode (MT5 not used in backtest)")
            return None

        try:
            # Fetch position details
            positions = self.mt5_api.positions_get()
            position = next((pos for pos in positions if str(pos.ticket) == str(position_id)), None)
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return None

            # Prepare close order (opposite action)
            action = 'sell' if position.type == mt5.POSITION_TYPE_BUY else 'buy'
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if action == 'sell' else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,  # Link to the position to close
                "price": self.mt5_api.get_tick_data(position.symbol)['bid' if action == 'sell' else 'ask'],
                "deviation": 10,
                "magic": 123456,
                "comment": "Close Position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }

            response = self.mt5_api.order_send(request)
            if response:
                self.logger.info(f"Position {position_id} closed successfully: {response}")
                return response
            self.logger.error(f"Failed to close position {position_id}: Empty response")
            return None
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {str(e)}")
            return None

    def execute_backtest_trades(self):
        """
        Simulate trade execution for backtesting (unchanged from original).

        Returns:
            list: List of simulated trade results.
        """
        if self.config['central_trading_bot']['mode'] == 'live':
            self.logger.error("execute_backtest_trades called in live mode; use execute_trade instead")
            return []

        # Assuming SignalGenerator provides last_signal via CentralTradingBot
        if not hasattr(self.signal_generator, 'last_signals') or not any(self.signal_generator.last_signals.values()):
            self.logger.debug("No signal available for backtest trade execution")
            return []

        trade_results = []
        for symbol, signal in self.signal_generator.last_signals.items():
            if signal:
                trade_result = {
                    "symbol": symbol,
                    "action": signal['action'],
                    "position_size": signal.get('position_size', 1),  # Default for backtest
                    "entry_price": signal['entry_price'],
                    "stop_loss": signal['stop_loss'],
                    "take_profit": signal['take_profit'],
                    "profit_loss": self._simulate_trade_outcome(signal),
                    "timestamp": signal['timestamp'],
                    "order_id": f"sim-{datetime.utcnow().timestamp()}"  # Simulated ID
                }
                self.logger.info(f"Simulated backtest trade: {trade_result}")
                trade_results.append(trade_result)
        return trade_results

    def _simulate_trade_outcome(self, signal):
        """
        Simulate trade outcome for backtesting (unchanged from original).

        Args:
            signal (dict): Trading signal with entry_price, stop_loss, take_profit.

        Returns:
            float: Simulated profit/loss.
        """
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        position_size = signal.get('position_size', 1)

        import random
        outcome = take_profit if random.choice([True, False]) else stop_loss
        profit_loss = (outcome - entry_price) * position_size if signal['action'] == 'buy' else (entry_price - outcome) * position_size
        return profit_loss


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "central_trading_bot": {"mode": "live"},
        "mt5_settings": {
            "server": "The5ers-Server",
            "login": "your_login",
            "password": "your_password"
        },
        "order_type": "market"
    }
    te = TradeExecution(config)
    signal = {
        "symbol": "EURUSD",
        "action": "buy",
        "volume": 0.1,
        "entry_price": 1.0900,
        "stop_loss": 1.0890,
        "take_profit": 1.0920
    }
    result = te.execute_trade(signal)
    print(result)
    te.mt5_api.shutdown()