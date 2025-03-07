import logging
from datetime import datetime

class TradeExecution:
    def __init__(self, config, ninja_trader_api):
        """
        Initialize TradeExecution with configuration and NinjaTraderAPI instance.

        Args:
            config (dict): Configuration dictionary with 'order_type' and other settings.
            ninja_trader_api (NinjaTraderAPI, optional): Instance for live API interactions.
        """
        self.order_type = config.get('order_type', 'market')  # Default to 'market'
        self.ninja_trader_api = ninja_trader_api  # None in backtest mode
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_trade(self, signal):
        """
        Send trade order to NinjaTrader via CrossTrade REST API in live mode.

        Args:
            signal (dict): Trading signal with 'action', 'position_size', 'stop_loss', 'take_profit', etc.

        Returns:
            dict or None: Trade result if successful, None if failed.
        """
        if self.ninja_trader_api is None:
            self.logger.error("Cannot execute trade: NinjaTraderAPI not initialized (backtest mode)")
            return None

        order = {
            "symbol": "NQ",
            "action": signal['action'],
            "quantity": signal['position_size'],
            "order_type": self.order_type,
            "stop_loss": signal['stop_loss'],
            "take_profit": signal['take_profit']
        }
        try:
            response = self.ninja_trader_api.place_order(order)
            if response and 'order_id' in response:  # Adjust based on CrossTrade REST response
                trade_result = {
                    "symbol": order["symbol"],
                    "action": order["action"],
                    "position_size": order["quantity"],
                    "entry_price": signal["entry_price"],
                    "profit_loss": response.get("profit_loss", 0.0),  # May not be immediate
                    "timestamp": datetime.utcnow().isoformat(),
                    "order_id": response["order_id"]
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
        Fetch account status from NinjaTrader via CrossTrade REST API in live mode.

        Returns:
            dict or None: Account status or None if failed or in backtest mode.
        """
        if self.ninja_trader_api is None:
            self.logger.debug("Account status not available in backtest mode")
            return None

        try:
            account_status = self.ninja_trader_api.get_account_status()
            if account_status:
                self.logger.debug(f"Account status retrieved: {account_status}")
                return account_status
            self.logger.error("Failed to retrieve account status: Empty response")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get account status: {str(e)}")
            return None

    def close_position(self, position_id):
        """
        Close a specific position via CrossTrade REST API in live mode.

        Args:
            position_id (str): Identifier of the position to close.

        Returns:
            dict or None: Response if successful, None if failed or in backtest mode.
        """
        if self.ninja_trader_api is None:
            self.logger.error(f"Cannot close position {position_id}: NinjaTraderAPI not initialized (backtest mode)")
            return None

        try:
            response = self.ninja_trader_api.close_position(position_id)
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
        Simulate trade execution for backtesting based on signals from SignalGenerator.

        Returns:
            list: List of simulated trade results.
        """
        if self.ninja_trader_api is not None:
            self.logger.error("execute_backtest_trades called in live mode; use execute_trade instead")
            return []

        # Assuming SignalGenerator provides last_signal via reference (e.g., CentralTradingBot)
        if not hasattr(self, 'signal_generator') or not self.signal_generator.last_signal:
            self.logger.debug("No signal available for backtest trade execution")
            return []

        signal = self.signal_generator.last_signal
        trade_result = {
            "symbol": "NQ",
            "action": signal['action'],
            "position_size": signal['position_size'],
            "entry_price": signal['entry_price'],
            "stop_loss": signal['stop_loss'],
            "take_profit": signal['take_profit'],
            "profit_loss": self._simulate_trade_outcome(signal),
            "timestamp": signal['timestamp'],
            "order_id": f"sim-{datetime.utcnow().timestamp()}"  # Simulated ID
        }
        self.logger.info(f"Simulated backtest trade: {trade_result}")
        return [trade_result]

    def _simulate_trade_outcome(self, signal):
        """
        Simulate trade outcome for backtesting.

        Args:
            signal (dict): Trading signal with entry_price, stop_loss, take_profit.

        Returns:
            float: Simulated profit/loss.
        """
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        position_size = signal['position_size']

        import random
        outcome = take_profit if random.choice([True, False]) else stop_loss
        profit_loss = (outcome - entry_price) * position_size if signal['action'] == 'buy' else (entry_price - outcome) * position_size
        return profit_loss

# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {"order_type": "market"}
    from NinjaTraderAPI import NinjaTraderAPI  # Assumes updated hybrid version
    ninja_api = NinjaTraderAPI({"ws_url": "ws://127.0.0.1:8088", "rest_url": "http://127.0.0.1:8080/api", "api_key": "test_key"})
    te = TradeExecution(config, ninja_api)
    signal = {"action": "buy", "position_size": 2, "stop_loss": 14950, "take_profit": 15100, "entry_price": 15000}
    result = te.execute_trade(signal)
    print(result)