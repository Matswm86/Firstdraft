import MetaTrader5 as mt5
import logging
from datetime import datetime
import pytz
import time


class MT5API:
    """
    MetaTrader 5 API wrapper for trading operations and market data retrieval.
    """

    def __init__(self, server, login, password):
        """
        Initialize MT5API with connection parameters.

        Args:
            server (str): MT5 server address
            login (str): Account login
            password (str): Account password
        """
        self.server = server
        self.login = login
        self.password = password
        self.logger = logging.getLogger(__name__)
        self.connected = False

        # Initialize MT5 connection
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize connection to MetaTrader 5 terminal."""
        try:
            # Initialize MT5 library
            if not mt5.initialize():
                self.logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
                return

            # Login to the trading account
            authorized = mt5.login(
                login=int(self.login),
                password=self.password,
                server=self.server
            )

            if not authorized:
                self.logger.error(f"Failed to login to MT5 account {self.login}: {mt5.last_error()}")
                return

            self.connected = True
            account_info = mt5.account_info()
            self.logger.info(
                f"Connected to MT5: Server={self.server}, Account={self.login}, "
                f"Balance={account_info.balance if account_info else 'Unknown'}"
            )
        except Exception as e:
            self.logger.error(f"Error initializing MT5 connection: {str(e)}")
            self.connected = False

    def _check_connection(self):
        """Check if connection is active, attempt reconnect if not."""
        if not self.connected or not mt5.terminal_info():
            self.logger.warning("MT5 connection lost, attempting to reconnect...")
            self.connected = False
            self._initialize_connection()
        return self.connected

    def get_tick_data(self, symbol):
        """
        Get latest tick data for a symbol.

        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD')

        Returns:
            dict or None: Tick data or None if failed
        """
        if not self._check_connection():
            self.logger.error(f"Cannot get tick data for {symbol}: Not connected")
            return None

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"Failed to get tick data for {symbol}: {mt5.last_error()}")
                return None

            tick_data = {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(tick.time, tz=pytz.UTC).isoformat(),
                'bid': float(tick.bid),
                'ask': float(tick.ask),
                'last': float(tick.last),
                'volume': float(tick.volume)
            }
            return tick_data
        except Exception as e:
            self.logger.error(f"Error getting tick data for {symbol}: {str(e)}")
            return None

    def get_ohlc_data(self, symbol, timeframe, count):
        """
        Get OHLC data for a symbol.

        Args:
            symbol (str): Trading symbol
            timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M1)
            count (int): Number of bars to retrieve

        Returns:
            list or None: List of OHLC dictionaries or None if failed
        """
        if not self._check_connection():
            self.logger.error(f"Cannot get OHLC data for {symbol}: Not connected")
            return None

        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to get OHLC data for {symbol}: {mt5.last_error()}")
                return None

            ohlc_data = [
                {
                    'timestamp': datetime.fromtimestamp(rate['time'], tz=pytz.UTC).isoformat(),
                    'open': float(rate['open']),
                    'high': float(rate['high']),
                    'low': float(rate['low']),
                    'close': float(rate['close']),
                    'volume': float(rate['tick_volume']),
                    'spread': float(rate['spread'])
                }
                for rate in rates
            ]
            return ohlc_data
        except Exception as e:
            self.logger.error(f"Error getting OHLC data for {symbol}: {str(e)}")
            return None

    def get_dom(self, symbol):
        """
        Get Depth of Market (DOM) data for a symbol.

        Args:
            symbol (str): Trading symbol

        Returns:
            list or None: List of DOM entries or None if failed
        """
        if not self._check_connection():
            self.logger.error(f"Cannot get DOM data for {symbol}: Not connected")
            return None

        try:
            # Ensure market book is subscribed
            if not mt5.market_book_add(symbol):
                self.logger.error(f"Failed to subscribe to market book for {symbol}: {mt5.last_error()}")
                return None

            # Give some time for DOM data to update
            time.sleep(0.1)

            dom = mt5.market_book_get(symbol)
            if dom is None:
                self.logger.error(f"Failed to get DOM data for {symbol}: {mt5.last_error()}")
                mt5.market_book_release(symbol)
                return None

            dom_data = [
                {
                    'type': 'bid' if item.type == mt5.BOOK_TYPE_SELL else 'ask',
                    'price': float(item.price),
                    'volume': float(item.volume)
                }
                for item in dom
            ]

            # Release market book subscription
            mt5.market_book_release(symbol)

            return dom_data
        except Exception as e:
            self.logger.error(f"Error getting DOM data for {symbol}: {str(e)}")
            mt5.market_book_release(symbol)
            return None

    def order_send(self, request):
        """
        Send a trading order to MT5.

        Args:
            request (dict): Order request parameters

        Returns:
            dict or None: Order result or None if failed
        """
        if not self._check_connection():
            self.logger.error("Cannot send order: Not connected")
            return None

        try:
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order send failed: {mt5.last_error() if result is None else result.comment}")
                return None

            order_result = {
                'ticket': result.order,
                'entry_price': result.price,
                'volume': result.volume
            }
            return order_result
        except Exception as e:
            self.logger.error(f"Error sending order: {str(e)}")
            return None

    def positions_get(self, symbol=None):
        """
        Get all open positions, optionally filtered by symbol.

        Args:
            symbol (str, optional): Symbol to filter positions

        Returns:
            list: List of position dictionaries
        """
        if not self._check_connection():
            self.logger.error("Cannot get positions: Not connected")
            return []

        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                self.logger.error(f"Failed to get positions: {mt5.last_error()}")
                return []

            position_list = [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,
                    'volume': float(pos.volume),
                    'price_open': float(pos.price_open),
                    'price_current': float(pos.price_current),
                    'profit': float(pos.profit),
                    'sl': float(pos.sl) if pos.sl != 0 else None,
                    'tp': float(pos.tp) if pos.tp != 0 else None,
                    'time': pos.time,
                    'comment': pos.comment
                }
                for pos in positions
            ]
            return position_list
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []

    def close_position(self, position_id):
        """
        Close an open position by ticket ID.

        Args:
            position_id (int): Position ticket ID

        Returns:
            dict or None: Result of closing operation or None if failed
        """
        if not self._check_connection():
            self.logger.error(f"Cannot close position {position_id}: Not connected")
            return None

        try:
            positions = self.positions_get()
            position = next((pos for pos in positions if pos['ticket'] == position_id), None)
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return None

            symbol = position['symbol']
            volume = position['volume']
            position_type = position['type']

            # Determine closing action
            action = mt5.ORDER_TYPE_SELL if position_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            # Get current market price
            tick = self.get_tick_data(symbol)
            if not tick:
                self.logger.error(f"Cannot get market price to close position {position_id}")
                return None

            price = tick['bid'] if action == mt5.ORDER_TYPE_SELL else tick['ask']

            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'position': position_id,
                'symbol': symbol,
                'volume': volume,
                'type': action,
                'price': price,
                'deviation': 10,
                'magic': position.get('magic', 123456),
                'comment': 'Close Position',
                'type_time': mt5.ORDER_TIME_GTC
            }

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Failed to close position {position_id}: {mt5.last_error() if result is None else result.comment}")
                return None

            close_result = {
                'ticket': position_id,
                'closed': True,
                'profit': result.profit,
                'price': result.price
            }
            return close_result
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {str(e)}")
            return None

    def modify_position(self, position_id, sl=None, tp=None):
        """
        Modify stop loss and take profit for an open position.

        Args:
            position_id (int): Position ticket ID
            sl (float, optional): New stop loss price
            tp (float, optional): New take profit price

        Returns:
            bool: True if modification successful, False otherwise
        """
        if not self._check_connection():
            self.logger.error(f"Cannot modify position {position_id}: Not connected")
            return False

        try:
            if sl is None and tp is None:
                self.logger.warning(f"No changes specified for position {position_id}")
                return False

            positions = self.positions_get()
            position = next((pos for pos in positions if pos['ticket'] == position_id), None)
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return False

            request = {
                'action': mt5.TRADE_ACTION_SLTP,
                'position': position_id,
                'symbol': position['symbol'],
                'sl': sl if sl is not None else position['sl'],
                'tp': tp if tp is not None else position['tp']
            }

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Failed to modify position {position_id}: {mt5.last_error() if result is None else result.comment}")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error modifying position {position_id}: {str(e)}")
            return False

    def get_account_info(self):
        """
        Get current account information.

        Returns:
            dict: Account information
        """
        if not self._check_connection():
            self.logger.error("Cannot get account info: Not connected")
            return {}

        try:
            account = mt5.account_info()
            if account is None:
                self.logger.error(f"Failed to get account info: {mt5.last_error()}")
                return {}

            return {
                'balance': float(account.balance),
                'equity': float(account.equity),
                'margin': float(account.margin),
                'free_margin': float(account.margin_free),
                'margin_level': float(account.margin_level),
                'login': account.login
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return {}

    def _get_symbol_info(self, symbol):
        """
        Get symbol specifications.

        Args:
            symbol (str): Trading symbol

        Returns:
            dict: Symbol information
        """
        if not self._check_connection():
            self.logger.error(f"Cannot get symbol info for {symbol}: Not connected")
            return {}

        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
                return {}

            return {
                'digits': symbol_info.digits,
                'point': float(symbol_info.point),
                'trade_tick_size': float(symbol_info.trade_tick_size),
                'trade_volume_min': float(symbol_info.volume_min),
                'trade_volume_max': float(symbol_info.volume_max),
                'trade_volume_step': float(symbol_info.volume_step)
            }
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return {}

    def get_historical_deals(self, from_date, to_date, symbol=None):
        """
        Get historical deals (closed trades).

        Args:
            from_date (datetime): Start date
            to_date (datetime): End date
            symbol (str, optional): Symbol to filter deals

        Returns:
            list: List of deal dictionaries
        """
        if not self._check_connection():
            self.logger.error("Cannot get historical deals: Not connected")
            return []

        try:
            from_ts = int(from_date.timestamp())
            to_ts = int(to_date.timestamp())
            deals = mt5.history_deals_get(from_ts, to_ts)

            if deals is None:
                self.logger.error(f"Failed to get historical deals: {mt5.last_error()}")
                return []

            deal_list = [
                {
                    'ticket': deal.ticket,
                    'order': deal.order,
                    'time': deal.time,
                    'type': deal.type,
                    'entry': deal.entry,
                    'symbol': deal.symbol,
                    'volume': float(deal.volume),
                    'price': float(deal.price),
                    'profit': float(deal.profit),
                    'commission': float(deal.commission),
                    'swap': float(deal.swap),
                    'comment': deal.comment
                }
                for deal in deals
                if (symbol is None or deal.symbol == symbol) and deal.entry == mt5.DEAL_ENTRY_OUT
            ]
            return deal_list
        except Exception as e:
            self.logger.error(f"Error getting historical deals: {str(e)}")
            return []

    def shutdown(self):
        """Shutdown MT5 connection."""
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                self.logger.info("MT5 connection shut down")
        except Exception as e:
            self.logger.error(f"Error shutting down MT5 connection: {str(e)}")