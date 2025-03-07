import websocket
import requests
import threading
import queue
import json
import logging


class NinjaTraderAPI:
    def __init__(self, config):
        """
        Initialize the NinjaTraderAPI with configuration settings.

        Args:
            config (dict): Configuration dictionary with NinjaTrader API settings.
        """
        self.ws_url = config.get('ws_url', 'wss://ninjatrader-ws-url')
        self.rest_url = config.get('rest_url', 'https://ninjatrader-rest-url')
        self.api_key = config.get('api_key', 'ENV:NINJATRADER_API_KEY')
        self.api_secret = config.get('api_secret', 'ENV:NINJATRADER_API_SECRET')
        self.logger = logging.getLogger(self.__class__.__name__)

        self.ws = None
        self.ws_thread = None
        self.tick_queue = queue.Queue()
        self.order_book = {}
        self.market_internals = {}

    def start_websocket(self, subscriptions):
        """
        Start WebSocket connection to NinjaTrader.

        Args:
            subscriptions (list): List of subscription requests (e.g., symbol and data type).
        """

        def on_message(ws, message):
            """Handle incoming WebSocket messages."""
            try:
                data = json.loads(message)
                if data.get('type') == 'tick':
                    tick = {
                        'timestamp': data['timestamp'],
                        'price': float(data['price']),
                        'volume': data.get('volume', 0),
                        'bid': data.get('bid'),
                        'ask': data.get('ask')
                    }
                    self.tick_queue.put(tick)
                elif data.get('type') == 'orderbook':
                    self.order_book = data
                elif data.get('type') == 'market_internal':
                    self.market_internals[data['name']] = data['value']
            except Exception as e:
                self.logger.error(f"WebSocket message processing failed: {str(e)}")

        def on_error(ws, error):
            """Handle WebSocket errors."""
            self.logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            """Handle WebSocket closure."""
            self.logger.info("WebSocket closed")

        def on_open(ws):
            """Handle WebSocket opening and send subscriptions."""
            self.logger.info("WebSocket opened")
            for sub in subscriptions:
                ws.send(json.dumps(sub))

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
            header={"Authorization": f"Bearer {self.api_key}"}
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def get_live_data(self):
        """
        Fetch the latest tick from the queue.

        Returns:
            dict or None: Latest tick data or None if queue is empty.
        """
        try:
            return self.tick_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def place_order(self, order):
        """
        Place an order via REST API.

        Args:
            order (dict): Order details (e.g., symbol, action, quantity).

        Returns:
            dict or None: API response or None if failed.
        """
        url = f"{self.rest_url}/orders"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(url, headers=headers, json=order)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            return None

    def get_account_status(self):
        """
        Fetch account status via REST API.

        Returns:
            dict or None: Account details or None if failed.
        """
        url = f"{self.rest_url}/account"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Account status query failed: {e}")
            return None

    def close_position(self, position_id):
        """
        Close a position via REST API.

        Args:
            position_id (str): ID of the position to close.

        Returns:
            dict or None: API response or None if failed.
        """
        url = f"{self.rest_url}/positions/{position_id}/close"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Position close failed: {e}")
            return None

# Example usage (for testing, commented out)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     config = {
#         "ws_url": "wss://ninjatrader-ws-url",
#         "rest_url": "https://ninjatrader-rest-url",
#         "api_key": "your_api_key",
#         "api_secret": "your_api_secret"
#     }
#     api = NinjaTraderAPI(config)
#     api.start_websocket([{"action": "subscribe", "symbol": "NQ", "type": "tick"}])