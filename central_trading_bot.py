import logging
import time
from datetime import datetime
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
        """
        Initialize the CentralTradingBot with configuration for MT5 and The 5%ers.

        Args:
            config (dict): Full configuration dictionary from config.json.
        """
        self.config = config
        self.mode = config['central_trading_bot']['mode']
        self.state = 'idle'
        self.logger = logging.getLogger(__name__)

        # Initialize MT5API for live trading
        if self.mode == 'live':
            self.mt5_api = MT5API(
                config['mt5_settings']['server'],
                config['mt5_settings']['login'],
                config['mt5_settings']['password']
            )
        else:
            self.mt5_api = None  # No MT5 connection needed for backtesting

        # Initialize modules with their respective config sections
        self.data_ingestion = DataIngestion(config)
        self.signal_generator = SignalGenerator(config)
        self.risk_management = RiskManagement(config)
        self.trade_execution = TradeExecution(config)
        self.notification = Notification(config)
        self.api_server = APIServer(config.get('api_server', {}), self)  # Pass self for status/control
        self.trade_logger = TradeLogger(config.get('trade_logger', {}))

        # Track positions per symbol
        self.current_positions = {symbol: None for symbol in config['symbols']}

    def start_live_trading(self):
        """
        Run the bot in live trading mode with MT5 and The 5%ers, executing trades on 15m, 5m, and 1m signals.
        """
        if self.mode != 'live':
            raise ValueError("Bot not configured for live mode")
        self.state = 'running'
        self.logger.info("Starting live trading with The 5%ers on MT5 for EURUSD and GBPJPY...")
        self.api_server.start()

        try:
            while self.state == 'running':
                current_date = datetime.utcnow().date()
                for symbol in self.config['symbols']:
                    # Fetch real-time data
                    tick = self.data_ingestion.fetch_live_data(symbol)
                    if tick:
                        # Generate signal
                        signal = self.signal_generator.process_tick(tick)

                        # Check position status
                        if self.current_positions[symbol]:
                            positions = self.mt5_api.positions_get(symbol=symbol)
                            if not positions:  # Position closed
                                exit_price = tick['price']
                                profit_loss = self.risk_management.update_pnl(self.current_positions[symbol],
                                                                              exit_price)
                                self.trade_logger.log_trade(self.current_positions[symbol])
                                self.notification.send_notification(
                                    "Trade Closed",
                                    {"symbol": symbol, "profit_loss": profit_loss}
                                )
                                self.current_positions[symbol] = None

                        # Process new signal
                        if signal and signal['timeframe'] in ['15m', '5m', '1m'] and not self.current_positions[symbol]:
                            # Apply stronger confluence for 1m signals
                            if signal['timeframe'] == '1m':
                                threshold = self.config['signal_generation']['thresholds'].get('1m', 18)
                                if signal.get('score', 0) < threshold:
                                    self.logger.debug(
                                        f"1m signal for {symbol} rejected: insufficient confluence (score: {signal.get('score', 0)} < {threshold})")
                                    continue

                            account_status = self.trade_execution.get_account_status()
                            adjusted_signal = self.risk_management.evaluate_signal(signal, current_date, account_status)
                            if adjusted_signal:
                                trade_result = self.trade_execution.execute_trade(adjusted_signal)
                                if trade_result:
                                    self.current_positions[symbol] = trade_result
                                    self.trade_logger.log_trade(trade_result)
                                    self.notification.send_notification("Trade Executed", trade_result)
                                    self.logger.info(
                                        f"Trade executed for {symbol} on {signal['timeframe']}: {trade_result['action']}")
                    time.sleep(0.1)  # Small delay per symbol to avoid overloading
                time.sleep(1)  # Main loop polling interval
        except Exception as e:
            self.logger.error(f"Live trading failed: {str(e)}")
            self.notification.send_notification("Live Trading Failed", {"error": str(e)})
        finally:
            self.state = 'idle'
            self.api_server.stop()
            if self.mt5_api:
                self.mt5_api.shutdown()

    def start_backtest(self):
        """
        Run the bot in backtest mode with existing NASDAQ futures data, filtering for 15m signals.
        """
        if self.mode != 'backtest':
            raise ValueError("Bot not configured for backtest mode")
        self.state = 'running'
        self.logger.info("Starting backtest with existing NASDAQ futures data...")

        try:
            # Load historical data for backtesting (unchanged from original)
            historical_data = self.data_ingestion.load_historical_data()
            if not historical_data:
                raise ValueError("No historical data loaded for backtesting")

            # Process historical data through signal generation and logging
            for (symbol, tf), data in historical_data.items():
                if data is not None and tf == '15m':  # Filter for 15m signals only
                    self.logger.info(f"Processing backtest data for {symbol} {tf}")
                    for index, row in data.iterrows():
                        tick = {
                            'timestamp': index.isoformat(),
                            'price': row['Close'],
                            'symbol': symbol
                        }
                        signal = self.signal_generator.process_tick(tick)
                        if signal and signal['timeframe'] == '15m':  # Only 15m signals
                            self.trade_logger.log_trade(signal)

            self.notification.send_notification(
                "Backtest Completed",
                {"mode": self.mode, "status": "success"}
            )
            self.logger.info("Backtest completed successfully")
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            self.notification.send_notification("Backtest Failed", {"error": str(e)})
        finally:
            self.state = 'idle'

    def stop(self):
        """
        Stop the bot and clean up resources.
        """
        self.logger.info("Stopping the bot...")
        self.state = 'stopped'
        self.api_server.stop()
        if self.mt5_api:
            self.mt5_api.shutdown()
        self.notification.send_notification("Bot Stopped", {"mode": self.mode})

# Example Usage (for reference, commented out)
# if __name__ == "__main__":
#     config = load_config('config.json')
#     bot = CentralTradingBot(config)
#     if bot.mode == "backtest":
#         bot.start_backtest()
#     else:
#         bot.start_live_trading()