import logging
import time
from datetime import datetime
import pytz
from MT5API import MT5API
from DataIngestion import DataIngestion
from SignalGenerator import SignalGenerator
from RiskManagement import RiskManagement
from TradeExecution import TradeExecution
from Notification import Notification
from APIServer import APIServer
from TradeLogger import TradeLogger

class CentralTradingBot:
    def __init__(self, config):
        self.config = config
        self.mode = config['central_trading_bot']['mode']
        self.state = 'idle'
        self.logger = logging.getLogger(__name__)

        if self.mode == 'live':
            self.mt5_api = MT5API(
                config['mt5_settings']['server'],
                config['mt5_settings']['login'],
                config['mt5_settings']['password']
            )
        else:
            self.mt5_api = None

        self.data_ingestion = DataIngestion(config, self.mt5_api)
        self.signal_generator = SignalGenerator(config, self.mt5_api)
        self.risk_management = RiskManagement(config)
        self.trade_execution = TradeExecution(config, self.mt5_api)
        self.notification = Notification(config)
        self.api_server = APIServer(config.get('api_server', {}), self)
        self.trade_logger = TradeLogger(config.get('trade_logger', {}))

        self.current_positions = {symbol: None for symbol in config['symbols']}
        self.last_trade_times = {symbol: None for symbol in config['symbols']}  # Per-symbol cooldown

    def start_live_trading(self):
        if self.mode != 'live':
            raise ValueError("Bot not configured for live mode")
        self.state = 'running'
        self.logger.info("Starting live trading with The 5%ers on MT5 for EURUSD and GBPJPY...")
        self.api_server.start()

        try:
            while self.state == 'running':
                current_date = datetime.now(pytz.UTC).date()
                current_time = datetime.now(pytz.UTC)

                for symbol in self.config['symbols']:
                    # Check 15-minute cooldown
                    if self.last_trade_times[symbol] and (current_time - self.last_trade_times[symbol]).total_seconds() < 900:  # 15 minutes
                        self.logger.debug(f"Cooldown active for {symbol}. Last trade at {self.last_trade_times[symbol]}")
                        continue

                    tick = self.data_ingestion.fetch_live_data(symbol)
                    if tick:
                        signal = self.signal_generator.process_tick(tick)
                        self.logger.debug(f"Processed tick for {symbol}: Signal = {signal}")

                        if self.current_positions[symbol]:
                            positions = self.mt5_api.positions_get(symbol=symbol)
                            if not positions:
                                exit_price = tick['price']
                                profit_loss = self.risk_management.update_pnl(self.current_positions[symbol], exit_price)
                                self.trade_logger.log_trade(self.current_positions[symbol])
                                self.notification.send_notification(
                                    "Trade Closed",
                                    {"symbol": symbol, "profit_loss": profit_loss}
                                )
                                self.current_positions[symbol] = None

                        if signal and signal['timeframe'] in ['15min', '5min', '1min'] and not self.current_positions[symbol]:
                            self.logger.debug(f"Signal eligible for execution: {signal}")
                            account_status = self.trade_execution.get_account_status()
                            self.logger.debug(f"Account status for {symbol}: {account_status}")
                            adjusted_signal = self.risk_management.evaluate_signal(signal, current_date, account_status)
                            self.logger.debug(f"Adjusted signal for {symbol}: {adjusted_signal}")
                            if adjusted_signal:
                                trade_result = self.trade_execution.execute_trade(adjusted_signal)
                                self.logger.debug(f"Trade result for {symbol}: {trade_result}")
                                if trade_result:
                                    self.current_positions[symbol] = trade_result
                                    self.trade_logger.log_trade(trade_result)
                                    self.last_trade_times[symbol] = datetime.now(pytz.UTC)
                                    self.notification.send_notification("Trade Executed", trade_result)
                                    self.logger.info(
                                        f"Trade executed for {symbol} on {signal['timeframe']}: {trade_result['action']}")
                    time.sleep(0.1)
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Live trading failed: {str(e)}")
            self.notification.send_notification("Live Trading Failed", {"error": str(e)})
        finally:
            self.state = 'idle'
            self.api_server.stop()
            if self.mt5_api:
                self.mt5_api.shutdown()

    def start_backtest(self):
        if self.mode != 'backtest':
            raise ValueError("Bot not configured for backtest mode")
        self.state = 'running'
        self.logger.info("Starting backtest with existing NASDAQ futures data...")
        try:
            historical_data = self.data_ingestion.load_historical_data()
            if not historical_data:
                raise ValueError("No historical data loaded for backtesting")
            for (symbol, tf), data in historical_data.items():
                if data is not None and tf == '15m':
                    self.logger.info(f"Processing backtest data for {symbol} {tf}")
                    for index, row in data.iterrows():
                        tick = {
                            'timestamp': index.isoformat(),
                            'price': row['Close'],
                            'symbol': symbol
                        }
                        signal = self.signal_generator.process_tick(tick)
                        if signal and signal['timeframe'] == '15m':
                            self.trade_logger.log_trade(signal)
            self.notification.send_notification("Backtest Completed", {"mode": self.mode, "status": "success"})
            self.logger.info("Backtest completed successfully")
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            self.notification.send_notification("Backtest Failed", {"error": str(e)})
        finally:
            self.state = 'idle'

    def stop(self):
        self.logger.info("Stopping the bot...")
        self.state = 'stopped'
        self.api_server.stop()
        if self.mt5_api:
            self.mt5_api.shutdown()
        self.notification.send_notification("Bot Stopped", {"mode": self.mode})