import MetaTrader5 as mt5
from datetime import datetime
import logging


class MT5API:
    def __init__(self, server, login, password):
        """
        Initialize the MT5API to connect to The 5%ers' MT5 server.

        Args:
            server (str): MT5 server name (e.g., 'The5ers-Server').
            login (str): MT5 account login ID.
            password (str): MT5 account password.
        """
        self.logger = logging.getLogger(__name__)
        self.server = server
        self.login = int(login)  # MT5 expects login as integer
        self.password = password

        # Initialize MT5 connection
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            raise Exception("MT5 initialization failed")

        # Login to MT5 account
        if not mt5.login(login=self.login, password=self.password, server=self.server):
            error = mt5.last_error()
            self.logger.error(f"MT5 login failed: {error}")
            mt5.shutdown()
            raise Exception(f"MT5 login failed: {error}")

        self.logger.info(f"Connected to MT5 server: {server} with login {login}")

    def get_tick_data(self, symbol):
        """
        Fetch real-time tick data for a given symbol.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD', 'GBPJPY').

        Returns:
            dict: Tick data with timestamp, bid, ask, last price, and volume, or None if failed.
        """
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get tick data for {symbol}: {error}")
                return None
            return {
                'timestamp': datetime.fromtimestamp(tick.time).isoformat(),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume
            }
        except Exception as e:
            self.logger.error(f"Exception getting tick data for {symbol}: {str(e)}")
            return None

    def get_ohlc_data(self, symbol, timeframe, count):
        """
        Fetch OHLC data for a given symbol and timeframe.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD', 'GBPJPY').
            timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M15).
            count (int): Number of bars to retrieve.

        Returns:
            list: List of OHLC data dictionaries, or None if failed.
        """
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get OHLC data for {symbol}: {error}")
                return None
            return [
                {
                    'timestamp': datetime.fromtimestamp(rate['time']).isoformat(),
                    'open': rate['open'],
                    'high': rate['high'],
                    'low': rate['low'],
                    'close': rate['close'],
                    'volume': rate['tick_volume'],
                    'spread': rate['spread']
                } for rate in rates
            ]
        except Exception as e:
            self.logger.error(f"Exception getting OHLC data for {symbol}: {str(e)}")
            return None

    def get_dom(self, symbol):
        """
        Fetch Depth of Market (DOM) data for a given symbol (Level 1).

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD', 'GBPJPY').

        Returns:
            list: List of DOM entries (bid/ask prices and volumes), or None if failed.
        """
        try:
            if not mt5.market_book_add(symbol):
                error = mt5.last_error()
                self.logger.error(f"Failed to subscribe to DOM for {symbol}: {error}")
                return None
            time.sleep(0.1)  # Brief delay to ensure subscription
            dom = mt5.market_book_get(symbol)
            mt5.market_book_release(symbol)  # Release subscription after fetch
            if dom is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get DOM for {symbol}: {error}")
                return None
            return [
                {
                    'type': 'bid' if entry.type == mt5.BOOK_TYPE_BUY else 'ask',
                    'price': entry.price,
                    'volume': entry.volume
                } for entry in dom
            ]
        except Exception as e:
            self.logger.error(f"Exception getting DOM for {symbol}: {str(e)}")
            return None

    def order_send(self, request):
        """
        Send a trade order to MT5.

        Args:
            request (dict): Order details (e.g., action, symbol, volume, type, price, sl, tp).

        Returns:
            dict: Trade result with order ID, entry price, and volume, or None if failed.
        """
        try:
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error = mt5.last_error()
                self.logger.error(
                    f"Order send failed for {request['symbol']}: {result.comment if result else 'No result'}, Error: {error}")
                return None
            self.logger.info(f"Order executed successfully for {request['symbol']}: Order ID {result.order}")
            return {
                'order_id': result.order,
                'entry_price': result.price,
                'volume': result.volume
            }
        except Exception as e:
            self.logger.error(f"Exception sending order for {request['symbol']}: {str(e)}")
            return None

    def positions_get(self, symbol=None):
        """
        Get current open positions, optionally filtered by symbol.

        Args:
            symbol (str, optional): Trading symbol to filter positions (e.g., 'EURUSD').

        Returns:
            list: List of position objects, or empty list if failed.
        """
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get positions for {symbol if symbol else 'all symbols'}: {error}")
                return []
            return positions
        except Exception as e:
            self.logger.error(f"Exception getting positions for {symbol if symbol else 'all symbols'}: {str(e)}")
            return []

    def account_balance(self):
        """
        Get the current account balance.

        Returns:
            float: Account balance in USD, or 0 if failed.
        """
        try:
            balance = mt5.account_balance()
            if balance is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get account balance: {error}")
                return 0
            return balance
        except Exception as e:
            self.logger.error(f"Exception getting account balance: {str(e)}")
            return 0

    def shutdown(self):
        """
        Shut down the MT5 connection.
        """
        try:
            mt5.shutdown()
            self.logger.info("MT5 connection shut down successfully")
        except Exception as e:
            self.logger.error(f"Exception during MT5 shutdown: {str(e)}")

# Example Usage (for testing, commented out)
# if __name__ == "__main__":
#     config = {
#         'server': 'The5ers-Server',
#         'login': 'your_login',
#         'password': 'your_password'
#     }
#     logging.basicConfig(level=logging.INFO)
#     api = MT5API(config['server'], config['login'], config['password'])
#     tick = api.get_tick_data('EURUSD')
#     print(f"Tick data: {tick}")
#     api.shutdown()