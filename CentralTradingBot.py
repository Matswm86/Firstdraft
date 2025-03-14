import logging
import time
from datetime import datetime
import pandas as pd
import pytz
import threading
import signal
import queue
import os
import json
import traceback
from typing import Dict, Optional, Callable, Any, Union, List


class CentralTradingBot:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.mode: str = config['central_trading_bot'].get('mode', 'backtest')
        self.state: str = 'idle'
        self.start_time: Optional[datetime] = None
        self.shutdown_requested: bool = False
        self.mt5_api: Optional[Any] = None

        self.modules_enabled: Dict[str, bool] = config.get('modules_enabled', {
            'signal_generator': True,
            'risk_management': True,
            'trade_execution': True,
            'market_news': True,
            'machine_learning': True,
            'api_server': True,
            'dashboard': True,
            'notification': True
        })

        disabled_modules: List[str] = [module for module, enabled in self.modules_enabled.items() if not enabled]
        if disabled_modules:
            self.logger.info(f"Running with the following modules disabled: {', '.join(disabled_modules)}")

        self.component_status: Dict[str, bool] = {
            'mt5_api': False,
            'data_ingestion': False,
            'signal_generator': False,
            'risk_management': False,
            'trade_execution': False,
            'notification': False,
            'api_server': False,
            'trade_logger': False,
            'market_news': False,
            'machine_learning': False,
            'dashboard': False
        }

        self.logger.info(f"Initializing Central Trading Bot in {self.mode} mode")

        self.initialize_components()

        symbols: List[str] = config.get('symbols', [])
        self.current_positions: Dict[str, Optional[Dict[str, Any]]] = {symbol: None for symbol in symbols}
        self.last_trade_time: Dict[str, Optional[datetime]] = {symbol: None for symbol in symbols}
        self.cool_down_period: int = config.get('central_trading_bot', {}).get('cool_down_minutes', 15) * 60  # Seconds

        self.event_queue: queue.Queue = queue.Queue()
        self.stop_event: threading.Event = threading.Event()

        self.performance_metrics: Dict[str, Union[int, float]] = {
            'trades_executed': 0,
            'signals_generated': 0,
            'signals_filtered': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'errors': 0,
            'uptime': 0.0
        }

        self.status_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        if self.mode == 'live':
            self._setup_signal_handlers()

        self.logger.info("Central Trading Bot initialized successfully")

    def initialize_components(self) -> None:
        try:
            if self.mode == 'live':
                self._init_mt5_api()
            else:
                self.mt5_api = None

            self._init_data_ingestion()

            if self.modules_enabled.get('SignalGenerator', True):
                self._init_signal_generator()
            else:
                self.logger.info("Signal Generator module disabled by configuration")
                self.signal_generator = self._create_stub('signal_generator')

            if self.modules_enabled.get('RiskManagement', True):
                self._init_risk_management()
            else:
                self.logger.info("Risk Management module disabled by configuration")
                self.risk_management = self._create_stub('risk_management')

            if self.modules_enabled.get('TradeExecution', True):
                self._init_trade_execution()
            else:
                self.logger.info("Trade Execution module disabled by configuration")
                self.trade_execution = self._create_stub('trade_execution')

            if self.modules_enabled.get('Notification', True):
                self._init_notification()
            else:
                self.logger.info("Notification module disabled by configuration")
                self.notification = self._create_stub('notification')

            self._init_trade_logger()

            if self.modules_enabled.get('APIServer', True):
                self._init_api_server()
            else:
                self.logger.info("API Server module disabled by configuration")
                self.api_server = self._create_stub('api_server')

            if self.modules_enabled.get('MarketNewsAPI', True):
                self._init_market_news()
            else:
                self.logger.info("Market News module disabled by configuration")
                self.market_news = self._create_stub('market_news')
                self.news_queue = queue.Queue()

            if self.modules_enabled.get('MachineLearning', True) and 'machine_learning' in self.config:
                self._init_machine_learning()
            else:
                self.logger.info("Machine Learning module disabled by configuration")
                self.machine_learning = self._create_stub('machine_learning')

            if self.modules_enabled.get('Dashboard', True) and 'dashboard' in self.config:
                self._init_dashboard()
            else:
                self.logger.info("Dashboard module disabled by configuration")
                self.dashboard = self._create_stub('dashboard')

        except Exception as e:
            self.logger.error(f"Error during component initialization: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _create_stub(self, module_name: str) -> Any:
        self.logger.info(f"Creating stub for {module_name} module")

        class ModuleStub:
            def __init__(self, logger: logging.Logger, module_name: str) -> None:
                self.logger = logger
                self.module_name = module_name

            def start(self) -> bool:
                self.logger.debug(f"Stub call to {self.module_name}.start() - No action taken")
                return True

            def stop(self) -> bool:
                self.logger.debug(f"Stub call to {self.module_name}.stop() - No action taken")
                return True

            def reset_daily_metrics(self, date: datetime.date) -> None:
                self.logger.debug(f"Stub call to {self.module_name}.reset_daily_metrics() - No action taken")

            def process_tick(self, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                self.logger.debug(f"Stub call to {self.module_name}.process_tick() - No action taken")
                return None

            def enhance_signals(self, signals: List[Dict[str, Any]], data: pd.DataFrame) -> List[Dict[str, Any]]:
                self.logger.debug(f"Stub call to {self.module_name}.enhance_signals() - No action taken")
                return signals

            def get_account_status(self) -> Dict[str, Any]:
                self.logger.debug(f"Stub call to {self.module_name}.get_account_status() - No action taken")
                return {'balance': 100000, 'equity': 100000, 'margin': 0, 'positions': {}}

            def evaluate_signal(self, signal: Dict[str, Any], time: datetime, status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                self.logger.debug(f"Stub call to {self.module_name}.evaluate_signal() - No action taken")
                return signal

            def execute_trade(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                self.logger.debug(f"Stub call to {self.module_name}.execute_trade() - No action taken")
                return {
                    "symbol": signal.get('symbol', 'UNKNOWN'),
                    "action": signal.get('action', 'buy'),
                    "position_size": signal.get('volume', 0.1),
                    "volume": signal.get('volume', 0.1),
                    "entry_price": signal.get('entry_price', 0.0),
                    "stop_loss": signal.get('stop_loss', 0.0),
                    "take_profit": signal.get('take_profit', 0.0),
                    "profit_loss": 0.0,
                    "timestamp": datetime.now(pytz.UTC).isoformat(),
                    "order_id": f"stub-{int(time.time())}"
                } if signal else None

            def send_notification(self, message: str, level: str = 'info') -> bool:
                self.logger.debug(f"Stub call to {self.module_name}.send_notification() - No action taken")
                return True

            def close_position(self, position_id: Union[int, str]) -> Optional[Dict[str, Any]]:
                self.logger.debug(f"Stub call to {self.module_name}.close_position() - No action taken")
                return None

            def get_status(self) -> Dict[str, str]:
                self.logger.debug(f"Stub call to {self.module_name}.get_status() - No action taken")
                return {"status": "stub", "module": self.module_name}

        return ModuleStub(self.logger, module_name)

    def _init_mt5_api(self) -> None:
        try:
            from MT5API import MT5API
            self.mt5_api = MT5API(
                self.config['mt5_settings']['server'],
                self.config['mt5_settings']['login'],
                self.config['mt5_settings']['password']
            )
            self.component_status['mt5_api'] = True
            self.logger.info("MT5 API initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MT5 API: {str(e)}")
            if self.config.get('central_trading_bot', {}).get('require_mt5', True):
                raise
            self.mt5_api = None

    def _init_data_ingestion(self) -> None:
        try:
            from DataIngestion import DataIngestion
            self.data_ingestion = DataIngestion(self.config, self.mt5_api)
            self.component_status['data_ingestion'] = True
            self.logger.info("Data Ingestion initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Data Ingestion: {str(e)}")
            self.data_ingestion = self._create_stub('data_ingestion')

    def _init_signal_generator(self) -> None:
        try:
            from SignalGenerator import SignalGenerator
            self.signal_generator = SignalGenerator(self.config, self.mt5_api)
            self.component_status['signal_generator'] = True
            self.logger.info("Signal Generator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Signal Generator: {str(e)}")
            self.signal_generator = self._create_stub('signal_generator')

    def _init_risk_management(self) -> None:
        try:
            from RiskManagement import RiskManagement
            self.risk_management = RiskManagement(self.config)
            self.component_status['risk_management'] = True
            self.logger.info("Risk Management initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Risk Management: {str(e)}")
            self.risk_management = self._create_stub('risk_management')

    def _init_trade_execution(self) -> None:
        try:
            from TradeExecution import TradeExecution
            self.trade_execution = TradeExecution(self.config, self.mt5_api)
            self.component_status['trade_execution'] = True
            self.logger.info("Trade Execution initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Trade Execution: {str(e)}")
            self.trade_execution = self._create_stub('trade_execution')

    def _init_notification(self) -> None:
        try:
            from Notification import Notification
            self.notification = Notification(self.config)
            self.component_status['notification'] = True
            self.logger.info("Notification initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Notification: {str(e)}")
            self.notification = self._create_stub('notification')

    def _init_api_server(self) -> None:
        try:
            from APIServer import APIServer
            self.api_server = APIServer(self.config.get('api_server', {}), self)
            self.component_status['api_server'] = True
            self.logger.info("API Server initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize API Server: {str(e)}")
            self.api_server = self._create_stub('api_server')

    def _init_trade_logger(self) -> None:
        try:
            from TradeLogger import TradeLogger
            self.trade_logger = TradeLogger(self.config.get('trade_logger', {}))
            self.component_status['trade_logger'] = True
            self.logger.info("Trade Logger initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Trade Logger: {str(e)}")
            self.trade_logger = self._create_stub('trade_logger')

    def _init_market_news(self) -> None:
        try:
            from MarketNewsAPI import MarketNewsAPI
            self.news_queue = queue.Queue()
            self.market_news = MarketNewsAPI(self.config, self.news_queue)
            self.component_status['market_news'] = True
            self.logger.info("Market News API initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Market News API: {str(e)}")
            self.market_news = self._create_stub('market_news')
            self.news_queue = queue.Queue()

    def _init_machine_learning(self) -> None:
        try:
            from MachineLearning import MachineLearning
            self.machine_learning = MachineLearning(self.config)
            self.component_status['machine_learning'] = True
            self.logger.info("Machine Learning initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Machine Learning: {str(e)}")
            self.machine_learning = self._create_stub('machine_learning')

    def _init_dashboard(self) -> None:
        try:
            from Dashboard import Dashboard
            self.dashboard = Dashboard(self.config, self, self.news_queue)
            self.component_status['dashboard'] = True
            self.logger.info("Dashboard initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Dashboard: {str(e)}")
            self.dashboard = self._create_stub('dashboard')

    def _setup_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.info("Signal handlers registered for graceful shutdown")
        except Exception as e:
            self.logger.error(f"Failed to set up signal handlers: {str(e)}")

    def _signal_handler(self, sig: int, frame: Any) -> None:
        self.logger.info(f"Received signal {sig}, initiating graceful shutdown")
        self.stop()

    def start(self) -> bool:
        if self.state != 'idle':
            self.logger.warning(f"Cannot start: Bot already in {self.state} state")
            return False

        if self.mode == 'live':
            return self.start_live_trading()
        else:
            return self.start_backtest()

    def start_live_trading(self) -> bool:
        if self.mode != 'live':
            self.logger.error("Cannot start live trading: Bot not in live mode")
            return False

        required_components = ['mt5_api', 'data_ingestion']
        if self.modules_enabled.get('SignalGenerator', True):
            required_components.append('signal_generator')
        if self.modules_enabled.get('RiskManagement', True):
            required_components.append('risk_management')
        if self.modules_enabled.get('TradeExecution', True):
            required_components.append('trade_execution')

        for component in required_components:
            if not self.component_status[component]:
                self.logger.error(f"Cannot start live trading: {component} not initialized")
                return False

        self.state = 'running'
        self.start_time = datetime.now(pytz.UTC)
        self.stop_event.clear()

        if self.api_server and self.component_status['api_server'] and self.modules_enabled.get('APIServer', True):
            try:
                self.api_server.start()
            except Exception as e:
                self.logger.error(f"Failed to start API server: {str(e)}")

        if self.market_news and self.component_status['market_news'] and self.modules_enabled.get('MarketNewsAPI', True):
            try:
                if hasattr(self.market_news, 'start_background_checks'):
                    self.market_news.start_background_checks()
            except Exception as e:
                self.logger.error(f"Failed to start news monitoring: {str(e)}")

        if self.dashboard and self.component_status['dashboard'] and self.modules_enabled.get('Dashboard', True):
            try:
                if hasattr(self.dashboard, 'start'):
                    dashboard_thread = threading.Thread(target=self.dashboard.start, daemon=True)
                    dashboard_thread.start()
            except Exception as e:
                self.logger.error(f"Failed to start dashboard: {str(e)}")

        self.logger.info(f"Live trading started with symbols: {self.config.get('symbols', [])}")

        self.trading_thread = threading.Thread(target=self._live_trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()

        return True

    def _live_trading_loop(self) -> None:
        try:
            self.logger.info("Trading loop started")
            observation_period = 300  # 5 minutes in seconds

            while not self.stop_event.is_set() and self.state == 'running':
                try:
                    current_time = datetime.now(pytz.UTC)
                    time_since_start = (current_time - self.start_time).total_seconds()

                    if time_since_start < observation_period:
                        self.logger.debug(f"Observing market: {int(observation_period - time_since_start)} seconds remaining")
                        time.sleep(1)
                        continue

                    if self.risk_management and self.modules_enabled.get('RiskManagement', True):
                        self.risk_management.reset_daily_metrics(current_time.date())

                    if self.news_queue and not self.news_queue.empty() and self.modules_enabled.get('MarketNewsAPI', True):
                        self._process_news_queue()

                    for symbol in self.config.get('symbols', []):
                        if self.stop_event.is_set():
                            break

                        if self._check_cooldown(symbol, current_time):
                            self.logger.debug(f"Cooldown active for {symbol}, skipping")
                            continue

                        self._process_symbol(symbol)
                        time.sleep(0.1)

                    self._update_performance_metrics()
                    time.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error in trading loop: {str(e)}")
                    self.performance_metrics['errors'] += 1
                    time.sleep(5)

            self.logger.info("Trading loop stopped")

        except Exception as e:
            self.logger.error(f"Fatal error in trading loop: {str(e)}")
            self.logger.error(traceback.format_exc())

        finally:
            if self.state == 'running':
                self.state = 'error'

    def _check_cooldown(self, symbol: str, current_time: datetime) -> bool:
        last_time = self.last_trade_time.get(symbol)
        if last_time and (current_time - last_time).total_seconds() < self.cool_down_period:
            return True
        return False

    def _process_symbol(self, symbol: str) -> None:
        try:
            tick = self.data_ingestion.fetch_live_data(symbol)
            if not tick:
                return

            signal = None
            if self.modules_enabled.get('SignalGenerator', True):
                signal = self.signal_generator.process_tick(tick)

            if signal:
                self.performance_metrics['signals_generated'] += 1
                self._process_signal(symbol, signal, tick)

            self._update_positions(symbol, tick)

        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {str(e)}")
            self.performance_metrics['errors'] += 1

    def _process_signal(self, symbol: str, signal: Dict[str, Any], tick: Dict[str, Any]) -> None:
        try:
            if self.current_positions[symbol]:
                self.logger.debug(f"Signal ignored for {symbol} - position already exists")
                self.performance_metrics['signals_filtered'] += 1
                return

            if signal['timeframe'] not in self.config['signal_generation']['trading_timeframes']:
                self.logger.debug(f"Signal ignored for {symbol} - unsupported timeframe: {signal['timeframe']}")
                self.performance_metrics['signals_filtered'] += 1
                return

            if self.modules_enabled.get('MachineLearning', True) and hasattr(self, 'machine_learning') and self.machine_learning:
                try:
                    recent_data = self._get_recent_data(symbol)
                    if recent_data is not None:
                        original_signal = signal.copy()
                        enhanced_signals = self.machine_learning.enhance_signals([signal], recent_data)
                        if not enhanced_signals:
                            self.logger.info(f"Signal rejected by ML for {symbol}")
                            self.performance_metrics['signals_filtered'] += 1
                            return
                        signal = enhanced_signals[0]
                        self.logger.debug(f"Signal enhanced by ML for {symbol}")
                except Exception as ml_error:
                    self.logger.error(f"Error in ML signal enhancement for {symbol}: {str(ml_error)}")
                    self.logger.debug(f"Continuing with original signal due to ML error")

            account_status: Optional[Dict[str, Any]] = None
            if self.modules_enabled.get('TradeExecution', True):
                account_status = self.trade_execution.get_account_status()
            if not account_status:
                account_status = {
                    'balance': self.risk_management.current_balance if hasattr(self.risk_management, 'current_balance') else 100000,
                    'equity': self.risk_management.current_balance if hasattr(self.risk_management, 'current_balance') else 100000,
                    'positions': {}
                }

            adjusted_signal: Optional[Dict[str, Any]] = signal
            if self.modules_enabled.get('RiskManagement', True):
                adjusted_signal = self.risk_management.evaluate_signal(signal, datetime.now(pytz.UTC), account_status)

            if not adjusted_signal:
                self.logger.debug(f"Signal rejected by risk management for {symbol}")
                self.performance_metrics['signals_filtered'] += 1
                return

            if self.modules_enabled.get('TradeExecution', True):
                self.logger.info(f"Executing trade for {symbol} based on {signal['timeframe']} signal")
                trade_result = self.trade_execution.execute_trade(adjusted_signal)
            else:
                trade_result = {
                    "symbol": symbol,
                    "action": adjusted_signal['action'],
                    "position_size": adjusted_signal.get('volume', 0.1),
                    "volume": adjusted_signal.get('volume', 0.1),
                    "entry_price": adjusted_signal['entry_price'],
                    "stop_loss": adjusted_signal.get('stop_loss'),
                    "take_profit": adjusted_signal.get('take_profit'),
                    "profit_loss": 0.0,
                    "timestamp": datetime.now(pytz.UTC).isoformat(),
                    "order_id": f"sim-{int(time.time())}"
                }
                self.logger.info(f"Simulated trade execution for {symbol} (module disabled)")

            if trade_result:
                self.current_positions[symbol] = trade_result
                self.last_trade_time[symbol] = datetime.now(pytz.UTC)

                if self.modules_enabled.get('RiskManagement', True) and hasattr(self.risk_management, 'track_position'):
                    self.risk_management.track_position(trade_result)

                if self.trade_logger:
                    self.trade_logger.log_trade(trade_result)

                self.performance_metrics['trades_executed'] += 1

                self.logger.info(f"Trade executed for {symbol}: {trade_result['action']} at {trade_result['entry_price']}")
            else:
                self.logger.warning(f"Trade execution failed for {symbol}")

        except Exception as e:
            self.logger.error(f"Error processing signal for {symbol}: {str(e)}")
            self.performance_metrics['errors'] += 1

    def _get_recent_data(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            timeframe = '1h'
            ohlc_data = self.data_ingestion.get_ohlc(symbol, self.data_ingestion.timeframe_map.get(timeframe))
            if not ohlc_data:
                self.logger.warning(f"No OHLC data available for {symbol} on {timeframe}")
                return None

            import pandas as pd
            df = pd.DataFrame(ohlc_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.columns = [col.lower() for col in df.columns]
            return df
        except Exception as e:
            self.logger.error(f"Error getting recent data for {symbol}: {str(e)}")
            return None

    def _update_positions(self, symbol: str, tick: Dict[str, Any]) -> None:
        if not self.current_positions[symbol]:
            return

        try:
            position_found = True
            if self.mt5_api and self.modules_enabled.get('TradeExecution', True):
                positions = self.mt5_api.positions_get(symbol=symbol)
                position_found = False

                if positions:
                    position_id = str(self.current_positions[symbol].get('order_id', ''))
                    for pos in positions:
                        if str(pos.get('ticket', '')) == position_id:
                            position_found = True
                            break

            if not position_found:
                self.logger.info(f"Position for {symbol} was closed")

                exit_price = tick.get('price')
                if exit_price is None:
                    self.logger.warning(f"No 'price' key in tick data for {symbol}, using last known price or 0")
                    exit_price = self.current_positions[symbol].get('entry_price', 0.0)

                profit_loss = 0.0

                if self.modules_enabled.get('RiskManagement', True) and hasattr(self.risk_management, 'update_pnl'):
                    profit_loss = self.risk_management.update_pnl(self.current_positions[symbol], exit_price)
                    if profit_loss > 0:
                        self.performance_metrics['profitable_trades'] += 1
                    else:
                        self.performance_metrics['losing_trades'] += 1

                if self.trade_logger:
                    closed_trade = self.current_positions[symbol].copy()
                    closed_trade['exit_price'] = exit_price
                    closed_trade['profit_loss'] = profit_loss
                    self.trade_logger.log_trade(closed_trade)

                self.current_positions[symbol] = None

        except Exception as e:
            self.logger.error(f"Error updating positions for {symbol}: {str(e)}")
            self.logger.debug(f"Tick data: {tick}")

    def _process_news_queue(self) -> None:
        try:
            max_items = 10
            for _ in range(max_items):
                if self.news_queue.empty():
                    break

                news_item = self.news_queue.get(block=False)
                if not isinstance(news_item, dict):
                    continue

                item_type = news_item.get('type')
                if item_type == 'risk_adjustment' and self.modules_enabled.get('RiskManagement', True):
                    if hasattr(self.risk_management, 'set_risk_level'):
                        risk_factor = news_item.get('risk_factor', 1.0)
                        duration = news_item.get('duration_minutes', 30)
                        self.risk_management.set_risk_level(risk_factor, duration)
                        self.logger.info(f"Risk adjusted to {risk_factor} based on news")

                self.logger.info(f"News: {news_item.get('message', 'No message')}")

        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing news queue: {str(e)}")

    def _update_performance_metrics(self) -> None:
        try:
            if self.start_time:
                uptime_seconds = (datetime.now(pytz.UTC) - self.start_time).total_seconds()
                self.performance_metrics['uptime'] = uptime_seconds

            if self.status_callback:
                try:
                    self.status_callback(self.get_status())
                except Exception as e:
                    self.logger.error(f"Error in status callback: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")

    def start_backtest(self) -> bool:
        if self.mode != 'backtest':
            self.logger.error("Cannot start backtest: Bot not in backtest mode")
            return False

        required_components = ['data_ingestion']
        for component in required_components:
            if not self.component_status[component]:
                self.logger.error(f"Cannot start backtest: {component} not initialized")
                return False

        self.logger.info("Starting backtest...")
        self.state = 'running'
        self.start_time = datetime.now(pytz.UTC)

        try:
            use_backtrader = self.config.get('backtesting', {}).get('use_backtrader', True)
            if use_backtrader:
                from Backtesting import Backtesting
                backtesting = Backtesting(self.config)
                backtesting.run()
            else:
                self._run_integrated_backtest()

            self.logger.info("Backtest completed")
            self.state = 'completed'
            return True

        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = 'error'
            return False

    def _run_integrated_backtest(self) -> None:
        self.logger.info("Running integrated backtest")
        historical_data = self.data_ingestion.load_historical_data()
        if not historical_data:
            self.logger.error("No historical data available for backtesting")
            return
        self.logger.info("Integrated backtest completed")

    def stop(self) -> bool:
        if self.state not in ['running', 'error']:
            self.logger.warning(f"Cannot stop: Bot not running (current state: {self.state})")
            return False

        self.logger.info("Stopping trading operations")
        self.shutdown_requested = True
        self.stop_event.set()

        if hasattr(self, 'trading_thread') and self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=30)
            if self.trading_thread.is_alive():
                self.logger.warning("Trading thread did not terminate cleanly within timeout")

        if self.api_server and hasattr(self.api_server, 'stop') and self.modules_enabled.get('APIServer', True):
            try:
                self.api_server.stop()
                self.logger.info("API server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping API server: {str(e)}")

        if self.market_news and hasattr(self.market_news, 'stop_background_checks') and self.modules_enabled.get('MarketNewsAPI', True):
            try:
                self.market_news.stop_background_checks()
                self.logger.info("Market news monitoring stopped")
            except Exception as e:
                self.logger.error(f"Error stopping market news monitoring: {str(e)}")

        if self.dashboard and hasattr(self.dashboard, 'stop') and self.modules_enabled.get('Dashboard', True):
            try:
                self.dashboard.stop()
                self.logger.info("Dashboard stopped")
            except Exception as e:
                self.logger.error(f"Error stopping dashboard: {str(e)}")

        if self.mt5_api and hasattr(self.mt5_api, 'shutdown'):
            try:
                self.mt5_api.shutdown()
                self.logger.info("MT5 connection closed")
            except Exception as e:
                self.logger.error(f"Error closing MT5 connection: {str(e)}")

        self._save_state()
        self.state = 'stopped'
        self.logger.info("Trading bot stopped")
        return True

    def get_status(self) -> Dict[str, Any]:
        uptime_seconds: float = 0.0
        if self.start_time:
            uptime_seconds = (datetime.now(pytz.UTC) - self.start_time).total_seconds()

        hours, remainder = divmod(int(uptime_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        positions: Dict[str, Optional[Dict[str, Any]]] = {}
        for symbol, pos in self.current_positions.items():
            if pos:
                positions[symbol] = {
                    'action': pos.get('action'),
                    'entry_price': pos.get('entry_price'),
                    'position_size': pos.get('position_size', pos.get('volume')),
                    'stop_loss': pos.get('stop_loss'),
                    'take_profit': pos.get('take_profit'),
                    'order_id': pos.get('order_id')
                }
            else:
                positions[symbol] = None

        risk_status: Dict[str, Any] = {}
        if self.risk_management and self.modules_enabled.get('RiskManagement', True) and hasattr(self.risk_management, 'get_status'):
            risk_status = self.risk_management.get_status()

        modules_status: Dict[str, bool] = {
            module: enabled
            for module, enabled in self.modules_enabled.items()
        }

        status: Dict[str, Any] = {
            'state': self.state,
            'mode': self.mode,
            'uptime': uptime_str,
            'uptime_seconds': uptime_seconds,
            'component_status': self.component_status,
            'modules_enabled': modules_status,
            'positions': positions,
            'risk_management': risk_status,
            'performance': self.performance_metrics,
            'timestamp': datetime.now(pytz.UTC).isoformat()
        }
        return status

    def _save_state(self) -> None:
        try:
            state_file = self.config.get('central_trading_bot', {}).get('state_file')
            if not state_file:
                return

            state_dir = os.path.dirname(state_file)
            if state_dir and not os.path.exists(state_dir):
                os.makedirs(state_dir, exist_ok=True)

            state_data = {
                'state': self.state,
                'mode': self.mode,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'component_status': self.component_status,
                'performance_metrics': self.performance_metrics,
                'last_trade_time': {
                    symbol: time.isoformat() if time else None
                    for symbol, time in self.last_trade_time.items()
                },
                'saved_at': datetime.now(pytz.UTC).isoformat()
            }

            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            self.logger.info(f"State saved to {state_file}")
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")

    def load_state(self) -> bool:
        try:
            state_file = self.config.get('central_trading_bot', {}).get('state_file')
            if not state_file or not os.path.exists(state_file):
                return False

            with open(state_file, 'r') as f:
                state_data = json.load(f)

            if not isinstance(state_data, dict):
                self.logger.error("Invalid state data format")
                return False

            if state_data.get('mode') != self.mode:
                self.logger.warning(f"State mode mismatch: {state_data.get('mode')} vs {self.mode}")
                return False

            if 'start_time' in state_data and state_data['start_time']:
                try:
                    self.start_time = datetime.fromisoformat(state_data['start_time'])
                except (ValueError, TypeError):
                    self.start_time = None

            if 'performance_metrics' in state_data and isinstance(state_data['performance_metrics'], dict):
                for key, value in state_data['performance_metrics'].items():
                    if key in self.performance_metrics:
                        self.performance_metrics[key] = value

            if 'last_trade_time' in state_data and isinstance(state_data['last_trade_time'], dict):
                for symbol, time_str in state_data['last_trade_time'].items():
                    if symbol in self.last_trade_time and time_str:
                        try:
                            self.last_trade_time[symbol] = datetime.fromisoformat(time_str)
                        except (ValueError, TypeError):
                            self.last_trade_time[symbol] = None

            self.logger.info(f"State loaded from {state_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            return False

    def register_status_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        if not callable(callback):
            self.logger.error("Status callback must be callable")
            return False

        self.status_callback = callback
        return True

    def close_position(self, symbol: str) -> bool:
        if self.state != 'running' or self.mode != 'live':
            self.logger.error("Cannot close position: Bot not running in live mode")
            return False

        if symbol not in self.current_positions or not self.current_positions[symbol]:
            self.logger.warning(f"No position to close for {symbol}")
            return False

        if not self.modules_enabled.get('TradeExecution', True):
            self.logger.warning(f"Cannot close position: Trade execution module disabled")
            return False

        try:
            position = self.current_positions[symbol]
            position_id = position.get('order_id')

            if not position_id:
                self.logger.error(f"Cannot close position for {symbol}: Missing order ID")
                return False

            result = self.trade_execution.close_position(position_id)

            if result:
                self.logger.info(f"Position closed for {symbol}: {result}")
                self.current_positions[symbol] = None
                profit_loss = result.get('profit', 0)
                if profit_loss > 0:
                    self.performance_metrics['profitable_trades'] += 1
                else:
                    self.performance_metrics['losing_trades'] += 1
                return True
            else:
                self.logger.error(f"Failed to close position for {symbol}")
                return False

        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {str(e)}")
            return False

    def generate_report(self) -> Optional[Dict[str, Any]]:
        if not self.trade_logger:
            self.logger.error("Cannot generate report: Trade logger not available")
            return None

        try:
            return self.trade_logger.generate_report()
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None