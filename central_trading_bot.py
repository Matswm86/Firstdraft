import logging
import time
from NinjaTraderAPI import NinjaTraderAPI
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
        Initialize the CentralTradingBot with configuration.

        Args:
            config (dict): Full configuration dictionary from config.json.
        """
        self.config = config
        self.mode = config['central_trading_bot']['mode']
        self.state = 'idle'

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize NinjaTraderAPI for live trading only
        if self.mode == 'live':
            self.ninja_trader_api = NinjaTraderAPI(config['ninja_trader_api'])
        else:
            self.ninja_trader_api = None  # Not needed for backtesting

        # Initialize modules with their respective config sections
        self.data_ingestion = DataIngestion(config['data_ingestion'],
                                            self.ninja_trader_api if self.mode == 'live' else None)
        self.signal_generator = SignalGenerator(config['signal_generation'],
                                                trade_execution=None)  # Placeholder for future integration
        self.risk_management = RiskManagement(config['risk_management'],
                                              trade_execution=None)  # Placeholder for future integration
        self.trade_execution = TradeExecution(config['ninja_trader_api'],
                                              self.ninja_trader_api if self.mode == 'live' else None)
        self.notification = Notification(config['notification'])
        self.api_server = APIServer(config['api_server'], self)  # Pass self for status/control
        self.trade_logger = TradeLogger(config['trade_logger'])

    def start_backtest(self):
        """
        Run the bot in backtest mode.
        """
        if self.mode != 'backtest':
            raise ValueError("Bot not configured for backtest mode")
        self.state = 'running'
        self.logger.info("Starting backtest...")

        try:
            historical_data = self.data_ingestion.load_historical_data()
            if not historical_data:
                raise ValueError("No historical data loaded for backtesting")

            self.signal_generator.load_historical_data(historical_data)
            for tf in self.signal_generator.trading_timeframes:
                self.signal_generator.generate_signal_for_tf(tf)
                # Note: execute_backtest_trades not implemented; simulate trades here if needed
                self.trade_logger.log_trades([])  # Placeholder until full backtest logic

            self.notification.send_notification("Backtest Completed",
                                                {"mode": self.mode, "status": "success"})
            self.logger.info("Backtest completed successfully")
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            self.notification.send_notification("Backtest Failed", {"error": str(e)})
        finally:
            self.state = 'idle'

    def start_live_trading(self):
        """
        Run the bot in live trading mode.
        """
        if self.mode != 'live':
            raise ValueError("Bot not configured for live mode")
        self.state = 'running'
        self.logger.info("Starting live trading...")
        self.api_server.start()

        try:
            # Start WebSocket subscriptions
            self.data_ingestion.start_websocket(self.config['live_trading']['subscriptions'])

            while self.state == 'running':
                tick = self.data_ingestion.fetch_live_data()
                if tick:
                    self.signal_generator.process_tick(tick)
                    # Assuming SignalGenerator sets a last_signal attribute (adjust if needed)
                    if hasattr(self.signal_generator, 'last_signal') and self.signal_generator.last_signal:
                        adjusted_signal = self.risk_management.evaluate_signal(self.signal_generator.last_signal)
                        if adjusted_signal:
                            trade_result = self.trade_execution.execute_trade(adjusted_signal)
                            if trade_result:
                                self.trade_logger.log_trade(trade_result)
                                # Assuming RiskManagement has on_trade_closed; adjust if named differently
                                if hasattr(self.risk_management, 'update_trade_result'):
                                    profit_loss = trade_result.get('profit_loss', 0.0)
                                    self.risk_management.update_trade_result(profit_loss)
                                self.notification.send_notification("Trade Executed", trade_result)
                time.sleep(1)  # Adjustable polling interval
        except Exception as e:
            self.logger.error(f"Live trading failed: {str(e)}")
            self.notification.send_notification("Live Trading Failed", {"error": str(e)})
        finally:
            self.state = 'idle'
            self.api_server.stop()

    def stop(self):
        """
        Stop the bot.
        """
        self.logger.info("Stopping the bot...")
        self.state = 'stopped'
        self.api_server.stop()
        self.notification.send_notification("Bot Stopped", {"mode": self.mode})

# Example Usage (for reference, commented out)
# if __name__ == "__main__":
#     config = {...}  # Omitted for brevity; use config.json
#     bot = CentralTradingBot(config)
#     if bot.mode == "backtest":
#         bot.start_backtest()
#     else:
#         bot.start_live_trading()