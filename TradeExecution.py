import logging


class TradeExecution:
    def __init__(self, config, ninja_trader_api):
        """
        Initialize TradeExecution with configuration and NinjaTraderAPI instance.

        Args:
            config (dict): Configuration dictionary containing 'order_type' and other settings.
            ninja_trader_api (NinjaTraderAPI): Instance of NinjaTraderAPI for API interactions.
        """
        self.order_type = config.get('order_type', 'market')  # Default to 'market' if not specified
        self.ninja_trader_api = ninja_trader_api
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_trade(self, signal):
        """
        Send trade order to NinjaTrader via NinjaTraderAPI.

        Args:
            signal (dict): Trading signal with 'action', 'position_size', 'stop_loss', 'take_profit'.

        Returns:
            dict or None: Response from NinjaTraderAPI if successful, None if failed.
        """
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
            self.logger.info("Trade executed successfully")
            return response  # Contains order details, e.g., order ID
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return None

    def get_account_status(self):
        """
        Fetch account status from NinjaTrader via NinjaTraderAPI.

        Returns:
            dict or None: Account status with 'balance', 'daily_profit', 'daily_loss', 'open_positions',
                          or None if the request fails.
        """
        try:
            account_status = self.ninja_trader_api.get_account_status()
            self.logger.debug(f"Account status retrieved: {account_status}")
            return account_status
        except Exception as e:
            self.logger.error(f"Failed to get account status: {e}")
            return None

    def close_position(self, position_id):
        """
        Close a specific position via NinjaTraderAPI.

        Args:
            position_id (str): Identifier of the position to close.

        Returns:
            dict or None: Response from NinjaTraderAPI if successful, None if failed.
        """
        try:
            response = self.ninja_trader_api.close_position(position_id)
            self.logger.info(f"Position {position_id} closed successfully")
            return response  # May contain P&L information
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {e}")
            return None