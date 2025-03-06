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
        """Initialize the CentralTradingBot with configuration."""
        self.config = config
        self.mode = config['central_trading_bot']['mode']
        self.state = 'idle'

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize NinjaTraderAPI for live trading only
        if self.mode == 'live':
            self.ninja_trader_api = NinjaTraderAPI(config['ninja_trader_api'])
        else:
            self.ninja_trader_api = None  # Not needed for backtesting

        # Initialize modules
        self.data_ingestion = DataIngestion(config['data_ingestion'], self.ninja_trader_api if self.mode == 'live' else None)
        self.signal_generator = SignalGenerator(config['signal_generation'])
        self.risk_management = RiskManagement(config['risk_management'])
        self.trade_execution = TradeExecution(config['trade_execution'], self.ninja_trader_api if self.mode == 'live' else None)
        self.notification = Notification(config['notification'])
        self.api_server = APIServer(config['api_server'], self)
        self.trade_logger = TradeLogger(config['trade_logger'])

    def start_backtest(self):
        """Run the bot in backtest mode."""
        if self.mode != 'backtest':
            raise ValueError("Bot not configured for backtest mode")
        self.state = 'running'
        self.logger.info("Starting backtest...")

        historical_data = self.data_ingestion.load_historical_data()
        if not historical_data:
            raise ValueError("No historical data loaded for backtesting")

        self.signal_generator.load_historical_data(historical_data)
        for tf in self.signal_generator.trading_timeframes:
            self.signal_generator.generate_signal_for_tf(tf)  # Updated to accept timeframe
            self.trade_execution.execute_backtest_trades()
            self.trade_logger.log_trades()

        self.logger.info("Backtest completed successfully")
        self.state = 'idle'

    def start_live_trading(self):
        """Run the bot in live trading mode."""
        if self.mode != 'live':
            raise ValueError("Bot not configured for live mode")
        self.state = 'running'
        self.logger.info("Starting live trading...")
        self.api_server.start()

        try:
            while self.state == 'running':
                tick = self.data_ingestion.fetch_live_data()
                if tick:
                    self.signal_generator.process_tick(tick)
                    if self.signal_generator.last_signal:  # Assuming last_signal is set by process_tick
                        adjusted_signal = self.risk_management.evaluate_signal(self.signal_generator.last_signal)
                        if adjusted_signal:
                            trade_result = self.trade_execution.execute_trade(adjusted_signal)
                            if trade_result:
                                self.risk_management.on_trade_closed(trade_result)
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Live trading failed: {str(e)}")
            self.state = 'idle'
            self.api_server.stop()

    def stop(self):
        """Stop the bot."""
        self.logger.info("Stopping the bot...")
        self.state = 'stopped'
        self.api_server.stop()

# Example Usage
if __name__ == "__main__":
    config = {
        "central_trading_bot": {"mode": "backtest"},
        "data_ingestion": {
            "historical_data_path": r"C:\Users\matsw\PycharmProjects\Firstdraft\data",
            "timeframe_files": {
                "1d": "backtrader_1d.csv",
                "4h": "backtrader_4h.csv",
                "1h": "backtrader_1h.csv",
                "30m": "backtrader_30m.csv",
                "15m": "backtrader_15m.csv",
                "5m": "backtrader_5m.csv",
                "1m": "backtrader_1m.csv"
            },
            "delimiter": ",",
            "column_names": ["datetime", "Open", "High", "Low", "Close", "Volume"]
        },
        "signal_generation": {
            "timeframes": ["1min", "5min", "15min", "30min", "1h", "4h", "daily"],
            "trading_timeframes": ["15min", "5min", "1min"],
            "thresholds": {"15min": 15, "5min": 15, "1min": 18},
            "max_bars": {"1min": 20160, "5min": 4032, "15min": 1344, "30min": 672, "1h": 336, "4h": 84, "daily": 14}
        },
        "risk_management": {},
        "trade_execution": {},
        "notification": {},
        "api_server": {},
        "trade_logger": {},
        "ninja_trader_api": {}  # Add NinjaTraderAPI config if needed
    }
    bot = CentralTradingBot(config)
    if bot.mode == "backtest":
        bot.start_backtest()
    else:
        bot.start_live_trading()