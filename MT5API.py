import MetaTrader5 as mt5
from datetime import datetime
import logging
import time
import pytz
import os

class MT5API:
    def __init__(self, server, login, password):
        self.logger = logging.getLogger(__name__)
        self.server = server
        self.login = int(login)
        self.password = password

        self.logger.info(f"Loading MT5API from: {os.path.abspath(__file__)}")
        self.logger.info(f"MetaTrader5 version: {mt5.__version__}")

        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            raise Exception("MT5 initialization failed")

        if not mt5.login(login=self.login, password=self.password, server=self.server):
            error = mt5.last_error()
            self.logger.error(f"MT5 login failed: {error}")
            mt5.shutdown()
            raise Exception(f"MT5 login failed: {error}")

        self.logger.info(f"Connected to MT5 server: {server} with login {login}")
        self.server_timezone_offset = self._get_server_timezone_offset()
        self.logger.info(f"Detected server timezone offset: {self.server_timezone_offset} seconds")

    def _get_server_timezone_offset(self):
        try:
            tick = mt5.symbol_info_tick("EURUSD")
            if tick is None:
                self.logger.error("Failed to fetch tick for server time estimation")
                return 7200
            server_time = datetime.fromtimestamp(tick.time, tz=pytz.UTC)
            utc_time = datetime.now(pytz.UTC)
            offset = int((server_time - utc_time).total_seconds())
            return offset
        except Exception as e:
            self.logger.error(f"Error calculating server timezone offset: {str(e)}")
            return 7200

    def get_tick_data(self, symbol):
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get tick data for {symbol}: {error}")
                return None
            server_time = datetime.fromtimestamp(tick.time, tz=pytz.UTC)
            utc_time = server_time
            return {
                'timestamp': utc_time.isoformat(),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'symbol': symbol
            }
        except Exception as e:
            self.logger.error(f"Exception getting tick data for {symbol}: {str(e)}")
            return None

    def get_ohlc_data(self, symbol, timeframe, count):
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get OHLC data for {symbol}: {error}")
                return None
            return [
                {
                    'timestamp': datetime.fromtimestamp(rate['time'], pytz.UTC).isoformat(),
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
        try:
            if not mt5.market_book_add(symbol):
                error = mt5.last_error()
                self.logger.error(f"Failed to subscribe to DOM for {symbol}: {error}")
                return None
            time.sleep(0.1)
            dom = mt5.market_book_get(symbol)
            mt5.market_book_release(symbol)
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
        try:
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            if positions is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get positions for {symbol if symbol else 'all symbols'}: {error}")
                return []
            return positions
        except Exception as e:
            self.logger.error(f"Exception getting positions for {symbol if symbol else 'all symbols'}: {str(e)}")
            return []

    def account_balance(self):
        try:
            info = mt5.account_info()
            if info is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get account information: {error}")
                return 0
            # Handle both float and AccountInfo object cases
            if isinstance(info, float):
                return info
            return info.balance
        except Exception as e:
            self.logger.error(f"Exception getting account balance: {str(e)}")
            return 0

    def shutdown(self):
        try:
            mt5.shutdown()
            self.logger.info("MT5 connection shut down successfully")
        except Exception as e:
            self.logger.error(f"Exception during MT5 shutdown: {str(e)}")