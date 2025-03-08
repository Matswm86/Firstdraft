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
        Initialize NinjaTraderAPI with WebSocket (ninja-socket), REST (CrossTrade/NinjaTrader), and webhook support.

        Args:
            config (dict): Configuration with ws_url, rest_url, webhook_url, and api_key.
        """
        self.config = config
        self.ws_url = config.get('ws_url', 'ws://127.0.0.1:8088')  # ninja-socket WebSocket
        self.rest_url = config.get('rest_url', 'https://app.crosstrade.io/v1/api')  # CrossTrade REST
        self.webhook_url = config.get('webhook_url')  # CrossTrade webhook
        self.api_key = config.get('api_key')  # CrossTrade or NinjaTrader key
        self.logger = logging.getLogger(__name__)
        self.tick_queue = queue.Queue()      # For WebSocket live data
        self.response_queue = queue.Queue()  # For WebSocket/REST responses
        self.ws = None
        self.ws_thread = None
        self.connected = False
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def start_websocket(self, subscriptions):
        """
        Start WebSocket connection to ninja-socket for live data.

        Args:
            subscriptions (list): List of subscription requests (e.g., [{"command": "subscribe", "symbol": "NQ 03-25"}]).
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

    def get_quote(self, instrument, account="Sim101"):
        """
        Fetch a quote via CrossTrade REST API.

        Args:
            instrument (str): Symbol (e.g., "NQ 03-25").
            account (str): NinjaTrader account name (default: "Sim101").

        Returns:
            dict or None: Quote data or None if failed.
        """
        url = f"{self.rest_url}/accounts/{account}/quote?instrument={instrument.replace(' ', '%20')}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            self.logger.debug(f"Quote retrieved: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get quote via REST: {str(e)}")
            return None

    def place_order(self, order, use_webhook=True):
        """
        Place an order via CrossTrade webhook (default) or REST/WebSocket if specified.

        Args:
            order (dict): Order details (e.g., symbol, action, quantity).
            use_webhook (bool): If True, use webhook; if False, attempt REST/WebSocket.

        Returns:
            dict or None: Response if successful, None if failed.
        """
        if use_webhook and self.webhook_url:
            # Webhook method (proven working)
            headers = {"Content-Type": "text/plain", "Origin": "crosstrade.io"}
            payload = (
                f"key={self.api_key};"
                f"command=PLACE;"
                f"account=Sim101;"  # Adjust if your account differs
                f"instrument={order['symbol']};"
                f"action={order['action'].upper()};"
                f"qty={order['quantity']};"
                f"tif=DAY;"
                f"order_type={order['order_type'].upper()};"
            )
            if order.get('stop_loss'):
                payload += f"stop_loss={order['stop_loss']};"
            if order.get('take_profit'):
                payload += f"take_profit={order['take_profit']};"
            try:
                response = requests.post(self.webhook_url, headers=headers, data=payload)
                response.raise_for_status()
                result = {"status": "success", "response": response.text}
                self.logger.info(f"Order placed via webhook: {result}")
                return result
            except Exception as e:
                self.logger.error(f"Webhook order placement failed: {str(e)}")
                return None
        else:
            # REST method (tentative, needs correct endpoint)
            url = f"{self.rest_url}/accounts/Sim101/orders"  # Placeholder—adjust if confirmed
            try:
                response = requests.post(url, headers=self.headers, json=order)
                response.raise_for_status()
                result = response.json()
                self.logger.info(f"Order placed via REST: {result}")
                return result
            except Exception as e:
                self.logger.error(f"REST order placement failed: {str(e)}")
                # Fallback to WebSocket if connected
                if self.connected and self.ws:
                    order_msg = {
                        "command": "place_order",
                        "symbol": order["symbol"],
                        "action": order["action"],
                        "quantity": order["quantity"],
                        "order_type": order["order_type"],
                        "stop_loss": order.get("stop_loss"),
                        "take_profit": order.get("take_profit")
                    }
                    try:
                        self.ws.send(json.dumps(order_msg))
                        self.logger.info("Order sent via WebSocket")
                        response = self.response_queue.get(timeout=5)
                        self.logger.info(f"WebSocket order response: {response}")
                        return response
                    except Exception as e2:
                        self.logger.error(f"WebSocket order failed: {str(e2)}")
                return None

    def get_account_status(self):
        """
        Get account status via CrossTrade REST API (placeholder endpoint).

        Returns:
            dict or None: Account details or None if failed.
        """
        url = f"{self.rest_url}/accounts/Sim101/account"  # Placeholder—needs confirmation
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
        Close a position via CrossTrade REST API (placeholder endpoint).

        Args:
            position_id (str): ID of the position to close.

        Returns:
            dict or None: Response if successful, None if failed.
        """
        url = f"{self.rest_url}/accounts/Sim101/positions/{position_id}/close"  # Placeholder
        try:
            response = requests.post(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"Position {position_id} closed via REST: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id} via REST: {str(e)}")
            return None