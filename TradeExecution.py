import logging
import MetaTrader5 as mt5
from datetime import datetime
import pytz
import time

class TradeExecution:
    def __init__(self, config, mt5_api):
        self.order_type = config.get('order_type', 'market')
        self.config = config
        self.mt5_api = mt5_api
        self.logger = logging.getLogger(__name__)

    def execute_trade(self, signal):
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.error("Cannot execute trade: Not in live mode (MT5 not used in backtest)")
            return None

        symbol = signal['symbol']
        action = signal['action']
        volume = signal['volume']
        entry_price = signal['entry_price']
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')

        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": stop_loss if stop_loss else 0.0,
            "tp": take_profit if take_profit else 0.0,
            "deviation": 10,
            "magic": 123456,
            "comment": "FTA Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.mt5_api.order_send(order)
                if result is None:
                    error = mt5.last_error()
                    self.logger.error(f"Trade failed for {symbol}: No result, Error: {error}")
                    return None
                # Handle dictionary result
                retcode = result.get('retcode', None) if isinstance(result, dict) else result.retcode
                if retcode == mt5.TRADE_RETCODE_REQUOTE:
                    self.logger.warning(f"Requote for {symbol}, retrying ({attempt + 1}/{max_retries})")
                    time.sleep(0.5)
                    continue
                if retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(f"Trade failed for {symbol}: {result.get('comment', 'Unknown')}, Error: {mt5.last_error()}")
                    return None
                trade_result = {
                    "symbol": symbol,
                    "action": action,
                    "position_size": volume,
                    "entry_price": result.get('price', entry_price),
                    "profit_loss": 0.0,
                    "timestamp": datetime.now(pytz.UTC).isoformat(),
                    "order_id": result.get('order', 0)
                }
                self.logger.info(f"Trade executed successfully: {trade_result}")
                return trade_result
            except Exception as e:
                self.logger.error(f"Trade execution failed for {symbol}: {str(e)}")
                return None
        self.logger.error(f"Trade failed for {symbol} after {max_retries} retries")
        return None

    def get_account_status(self):
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
            return {'balance': 0, 'positions': {}}
        except Exception as e:
            self.logger.error(f"Failed to get account status: {str(e)}")
            return {'balance': 0, 'positions': {}}

    def close_position(self, position_id):
        if self.config['central_trading_bot']['mode'] != 'live':
            self.logger.error(f"Cannot close position {position_id}: Not in live mode (MT5 not used in backtest)")
            return None

        try:
            positions = self.mt5_api.positions_get()
            position = next((pos for pos in positions if str(pos.ticket) == str(position_id)), None)
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return None

            action = 'sell' if position.type == mt5.POSITION_TYPE_BUY else 'buy'
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if action == 'sell' else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": self.mt5_api.get_tick_data(position.symbol)['bid' if action == 'sell' else 'ask'],
                "deviation": 10,
                "magic": 123456,
                "comment": "Close Position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }

            response = self.mt5_api.order_send(request)
            if response and response.get('retcode', mt5.TRADE_RETCODE_DONE) == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {position_id} closed successfully: {response}")
                return response
            self.logger.error(f"Failed to close position {position_id}: {response.get('comment', 'No response') if response else 'No response'}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {str(e)}")
            return None

    def execute_backtest_trades(self):
        if self.config['central_trading_bot']['mode'] == 'live':
            self.logger.error("execute_backtest_trades called in live mode; use execute_trade instead")
            return []

        if not hasattr(self, 'signal_generator') or not any(self.signal_generator.last_signals.values()):
            self.logger.debug("No signal available for backtest trade execution")
            return []

        trade_results = []
        for symbol, signal in self.signal_generator.last_signals.items():
            if signal:
                trade_result = {
                    "symbol": symbol,
                    "action": signal['action'],
                    "position_size": signal.get('position_size', 1),
                    "entry_price": signal['entry_price'],
                    "stop_loss": signal.get('stop_loss'),
                    "take_profit": signal.get('take_profit'),
                    "profit_loss": self._simulate_trade_outcome(signal),
                    "timestamp": signal['timestamp'],
                    "order_id": f"sim-{datetime.now(pytz.UTC).timestamp()}"
                }
                self.logger.info(f"Simulated backtest trade: {trade_result}")
                trade_results.append(trade_result)
        return trade_results

    def _simulate_trade_outcome(self, signal):
        entry_price = signal['entry_price']
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        position_size = signal.get('position_size', 1)

        import random
        outcome = take_profit if random.choice([True, False]) else stop_loss
        profit_loss = (outcome - entry_price) * position_size if signal['action'] == 'buy' else (entry_price - outcome) * position_size
        return profit_loss