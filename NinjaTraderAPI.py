import websocket
import threading
import queue
import json
import requests
import logging
from datetime import datetime

class NinjaTraderAPI:
    def __init__(self, config):
        """
        Initialize NinjaTraderAPI with WebSocket (ninja-socket) and REST (CrossTrade) support.

        Args:
            config (dict): Configuration with ws_url, rest_url, and api_key.
        """
        self.config = config
        self.ws_url = config.get('ws_url', 'ws://127.0.0.1:8088')  # ninja-socket default
        self.rest_url = config.get('rest_url', 'https://app.crosstrade.io/v1/send/Ih0yVAcJ/F4xqDBYyh-R-w8_6FaYUCA')  # CrossTrade default (adjust port)
        self.api_key = config.get('A_HhRVALPcyadtJ61U0YM-20LG_aqh7IQPQQhpZWw-Q')  # From CrossTrade setup
        self.logger = logging.getLogger(__name__)
        self.tick_queue = queue.Queue()      # For WebSocket market data
        self.response_queue = queue.Queue()  # For WebSocket command responses
        self.ws = None
        self.ws_thread = None
        self.connected = False
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def start_websocket(self, subscriptions):
        """
        Start WebSocket connection to ninja-socket for live data.

        Args:
            subscriptions (list): List of subscription requests (e.g., [{"command": "subscribe", "symbol": "NQ"}]).
        """
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'data' in data:
                    tick = {
                        'timestamp': data.get('timestamp', datetime.utcnow().isoformat()),
                        'price': float(data.get('price', data['data'])),
                        'volume': data.get('volume', 0),
                        'bid': data.get('bid'),
                        'ask': data.get('ask')
                    }
                    self.tick_queue.put(tick)
                elif 'response' in data or 'order_id' in data:
                    self.response_queue.put(data)
                self.logger.debug(f"WebSocket message: {data}")
            except Exception as e:
                self.logger.error(f"WebSocket message processing failed: {str(e)}")

        def on_open(ws):
            self.connected = True
            self.logger.info("Connected to ninja-socket WebSocket")
            for sub in subscriptions:
                ws.send(json.dumps(sub))
                self.logger.info(f"Sent subscription: {sub}")

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=lambda ws, error: self.logger.error(f"WebSocket error: {error}"),
            on_close=lambda ws, code, msg: self.logger.info(f"WebSocket closed: {code} - {msg}"),
            on_open=on_open
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def get_live_data(self):
        """
        Fetch latest tick data from ninja-socket WebSocket queue.

        Returns:
            dict or None: Latest tick data or None if queue is empty.
        """
        try:
            return self.tick_queue.get_nowait() if not self.tick_queue.empty() else None
        except Exception as e:
            self.logger.error(f"Failed to fetch live data: {str(e)}")
            return None

    def place_order(self, order):
        """
        Place an order via CrossTrade REST API.

        Args:
            order (dict): Order details (e.g., symbol, action, quantity).

        Returns:
            dict or None: REST response or None if failed.
        """
        url = f"{self.rest_url}/orders"  # Adjust based on CrossTrade API docs
        try:
            response = requests.post(url, headers=self.headers, json=order)
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"Order placed via REST: {result}")
            return result
        except Exception as e:
            self.logger.error(f"REST order placement failed: {str(e)}")
            return None

    def get_account_status(self):
        """
        Get account status via CrossTrade REST API.

        Returns:
            dict or None: Account details or None if failed.
        """
        url = f"{self.rest_url}/account"  # Adjust based on CrossTrade API docs
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            self.logger.debug(f"Account status: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get account status via REST: {str(e)}")
            return None

    def close_position(self, position_id):
        """
        Close a position via CrossTrade REST API.

        Args:
            position_id (str): ID of the position to close.

        Returns:
            dict or None: REST response or None if failed.
        """
        url = f"{self.rest_url}/positions/{position_id}/close"  # Adjust based on CrossTrade API docs
        try:
            response = requests.post(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"Position {position_id} closed via REST: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id} via REST: {str(e)}")
            return None