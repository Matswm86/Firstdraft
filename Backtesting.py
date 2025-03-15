import backtrader as bt
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
from BacktraderStrategy import BacktraderStrategy
from DataIngestion import DataIngestion
from SignalGenerator import SignalGenerator
import traceback


class Backtesting:
    """
    Enhanced Backtesting class that integrates with the modular trading system.
    Provides comprehensive backtesting capabilities using the MarketStructure and OrderFlow modules.
    """

    def __init__(self, config, signal_generator=None, data_ingestion=None):
        """
        Initialize Backtesting with configuration settings and optional external components.

        Args:
            config (dict): Configuration dictionary with backtesting settings
            signal_generator (SignalGenerator, optional): Existing SignalGenerator instance
            data_ingestion (DataIngestion, optional): Existing DataIngestion instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Use provided components or create new ones
        self.data_ingestion = data_ingestion if data_ingestion else DataIngestion(config)
        self.signal_generator = signal_generator

        # Parse configuration
        self.symbols = config['backtesting'].get('symbols', config.get('symbols', []))

        # Trading parameters
        self.commission = config['backtesting'].get('commission', 0.0002)  # Default: 0.02%
        self.slippage = config['backtesting'].get('slippage', 0.0001)  # Default: 0.01%
        self.initial_capital = config['backtesting'].get('initial_capital', 100000)

        # Risk parameters
        self.risk_per_trade = config['backtesting'].get('risk_per_trade', 0.01)  # 1% per trade
        self.position_sizing = config['backtesting'].get('position_sizing', 'risk')  # 'fixed' or 'risk'

        # Signal thresholds
        self.threshold = config['backtesting'].get('threshold', 6.0)  # Minimum combined score

        # Output settings
        self.enable_detailed_metrics = config['backtesting'].get('enable_detailed_metrics', True)
        self.plot_results = config['backtesting'].get('plot_results', True)
        self.save_results = config['backtesting'].get('save_results', True)
        self.results_path = config['backtesting'].get('results_path', 'backtest_results')

        # Timeframe mapping between main system and backtrader
        self.timeframe_mapping = {
            '1min': '1m', '5min': '5m', '15min': '15m', '30min': '30m',
            '1h': '1h', '4h': '4h', 'daily': '1d'
        }
        self.reverse_tf_mapping = {v: k for k, v in self.timeframe_mapping.items()}

        # Results storage
        self.results = None

        self.logger.info(f"Backtesting initialized for symbols: {', '.join(self.symbols)}")

    def initialize_modules(self):
        """Initialize signal generator and its analysis modules if not provided."""
        if not self.signal_generator:
            try:
                self.logger.info("Creating Signal Generator for backtesting")
                self.signal_generator = SignalGenerator(self.config)
                # Initialize history data
                for symbol in self.symbols:
                    self.logger.debug(f"Initializing history data for {symbol}")
            except Exception as e:
                self.logger.error(f"Error initializing Signal Generator: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise

    def run(self):
        """Run the backtest with the enhanced framework."""
        try:
            # Initialize modules
            self.initialize_modules()

            # Set output directory
            if self.save_results and not os.path.exists(self.results_path):
                os.makedirs(self.results_path)

            # Initialize Cerebro
            cerebro = bt.Cerebro()

            # Add strategy with integrated modules
            cerebro.addstrategy(
                BacktraderStrategy,
                signal_generator=self.signal_generator,
                market_structure=self.signal_generator.market_structure,
                order_flow=self.signal_generator.order_flow,
                threshold=self.threshold,
                position_sizing=self.position_sizing,
                risk_per_trade=self.risk_per_trade
            )

            # Load historical data
            historical_data = self.data_ingestion.load_historical_data()
            if not historical_data:
                self.logger.error("No historical data available for backtesting")
                return False

            # Add data feeds
            data_added = False
            for (symbol, orig_timeframe), df in historical_data.items():
                if symbol not in self.symbols:
                    continue

                if df is None or df.empty:
                    self.logger.warning(f"Empty data for {symbol} on {orig_timeframe}")
                    continue

                # Map timeframe naming convention
                bt_timeframe = self.timeframe_mapping.get(orig_timeframe, orig_timeframe)

                # Prepare the DataFrame for backtrader
                if 'datetime' in df.columns:
                    df = df.set_index('datetime')

                # Ensure column names are correct for backtrader
                column_map = {
                    'open': 'open', 'high': 'high', 'low': 'low',
                    'close': 'close', 'volume': 'volume'
                }

                # Rename columns to standardized names if needed
                df_renamed = df.rename(columns={
                    col: std_col for col, std_col in column_map.items()
                    if col in df.columns and col != std_col
                })

                # Ensure all required columns exist
                if not all(col in df_renamed.columns for col in ['open', 'high', 'low', 'close']):
                    self.logger.warning(f"Missing required columns for {symbol} on {bt_timeframe}")
                    continue

                # Add 'volume' column if missing
                if 'volume' not in df_renamed.columns:
                    df_renamed['volume'] = 0

                # Create backtrader data feed
                data = bt.feeds.PandasData(dataname=df_renamed)
                cerebro.adddata(data, name=f"{symbol}_{bt_timeframe}")
                self.logger.info(f"Added {symbol} {bt_timeframe} data to backtest")
                data_added = True

            if not data_added:
                self.logger.error("No valid data feeds added to backtest")
                return False

            # Set up broker
            cerebro.broker.set_cash(self.initial_capital)
            cerebro.broker.setcommission(commission=self.commission)
            cerebro.broker.set_slippage_perc(self.slippage)

            # Add performance analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
            cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='period_stats')

            # Add daily and monthly return analyzers
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='daily_returns', timeframe=bt.TimeFrame.Days)
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='monthly_returns', timeframe=bt.TimeFrame.Months)

            # Run the backtest
            self.logger.info("Starting backtest...")
            self.results = cerebro.run()

            # Extract and log performance metrics
            if self.results:
                self._log_performance_metrics(self.results[0])

                # Save results if enabled
                if self.save_results:
                    self._save_results(self.results[0])

                # Plot results if enabled
                if self.plot_results:
                    self._plot_results(cerebro)

            return True

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _log_performance_metrics(self, result):
        """Log comprehensive performance metrics."""
        # Access analyzers
        sharpe = result.analyzers.sharpe.get_analysis()
        drawdown = result.analyzers.drawdown.get_analysis()
        trades = result.analyzers.trades.get_analysis()
        returns = result.analyzers.returns.get_analysis()
        period_stats = result.analyzers.period_stats.get_analysis()

        # Calculate key metrics
        total_return = (result.broker.getvalue() / self.initial_capital - 1) * 100
        max_dd = drawdown.get('max', {}).get('drawdown', 0) * 100

        # Calculate win rate
        total_trades = trades.get('total', 0)
        if total_trades > 0:
            win_rate = (trades.get('won', 0) / total_trades) * 100
            avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
            avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0)
            profit_factor = abs(avg_win / avg_loss) if avg_loss else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Log results
        self.logger.info(f"\n{'=' * 50}\nBacktest Results Summary\n{'=' * 50}")
        self.logger.info(f"Initial capital: ${self.initial_capital:.2f}")
        self.logger.info(f"Final portfolio value: ${result.broker.getvalue():.2f}")
        self.logger.info(f"Total return: {total_return:.2f}%")
        self.logger.info(f"Sharpe ratio: {sharpe.get('sharperatio', 0):.4f}")
        self.logger.info(f"Maximum drawdown: {max_dd:.2f}%")
        self.logger.info(f"Win rate: {win_rate:.2f}% ({trades.get('won', 0)}/{total_trades})")
        self.logger.info(f"Average winning trade: ${avg_win:.2f}")
        self.logger.info(f"Average losing trade: ${abs(avg_loss):.2f}")
        self.logger.info(f"Profit factor: {profit_factor:.2f}")
        self.logger.info(f"CAGR: {returns.get('cagr', 0) * 100:.2f}%")
        self.logger.info(f"Annualized volatility: {period_stats.get('stddev_ann', 0) * 100:.2f}%")
        self.logger.info(f"{'=' * 50}")

    def _save_results(self, result):
        """Save backtest results to files."""
        try:
            # Create timestamp for the results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create directory for this backtest run
            result_dir = os.path.join(self.results_path, f"backtest_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)

            # Save overall metrics
            metrics = self._get_metrics_dict(result)
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(os.path.join(result_dir, "metrics.csv"), index=False)

            # Save trade list
            trades = result.analyzers.trades.get_analysis()
            if 'total' in trades and trades['total'] > 0:
                # Extract individual trades if available
                trade_list = self._extract_trade_list(result)
                if trade_list:
                    trade_df = pd.DataFrame(trade_list)
                    trade_df.to_csv(os.path.join(result_dir, "trades.csv"), index=False)

            # Save daily returns
            daily_returns = result.analyzers.daily_returns.get_analysis()
            if daily_returns:
                daily_returns_df = pd.DataFrame({
                    'date': [dt.strftime("%Y-%m-%d") for dt in daily_returns.keys()],
                    'return': [ret for ret in daily_returns.values()]
                })
                daily_returns_df.to_csv(os.path.join(result_dir, "daily_returns.csv"), index=False)

            # Save monthly returns
            monthly_returns = result.analyzers.monthly_returns.get_analysis()
            if monthly_returns:
                monthly_returns_df = pd.DataFrame({
                    'month': [dt.strftime("%Y-%m") for dt in monthly_returns.keys()],
                    'return': [ret for ret in monthly_returns.values()]
                })
                monthly_returns_df.to_csv(os.path.join(result_dir, "monthly_returns.csv"), index=False)

            # Save configuration
            config_df = pd.DataFrame([{
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'slippage': self.slippage,
                'risk_per_trade': self.risk_per_trade,
                'position_sizing': self.position_sizing,
                'threshold': self.threshold,
                'symbols': ', '.join(self.symbols)
            }])
            config_df.to_csv(os.path.join(result_dir, "config.csv"), index=False)

            self.logger.info(f"Results saved to {result_dir}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _get_metrics_dict(self, result):
        """Convert backtest results to a dictionary of metrics."""
        # Access analyzers
        sharpe = result.analyzers.sharpe.get_analysis()
        drawdown = result.analyzers.drawdown.get_analysis()
        trades = result.analyzers.trades.get_analysis()
        returns = result.analyzers.returns.get_analysis()
        period_stats = result.analyzers.period_stats.get_analysis()

        # Calculate metrics
        total_return = (result.broker.getvalue() / self.initial_capital - 1) * 100
        max_dd = drawdown.get('max', {}).get('drawdown', 0) * 100

        # Calculate win rate and other trade metrics
        total_trades = trades.get('total', 0)
        if total_trades > 0:
            win_rate = (trades.get('won', 0) / total_trades) * 100
            avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
            avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0)
            profit_factor = abs(avg_win / avg_loss) if avg_loss else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Create metrics dictionary
        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': result.broker.getvalue(),
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe.get('sharperatio', 0),
            'max_drawdown_pct': max_dd,
            'total_trades': total_trades,
            'winning_trades': trades.get('won', 0),
            'losing_trades': trades.get('lost', 0),
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'cagr_pct': returns.get('cagr', 0) * 100,
            'annualized_volatility_pct': period_stats.get('stddev_ann', 0) * 100,
            'backtest_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return metrics

    def _extract_trade_list(self, result):
        """Extract list of individual trades if available."""
        # This is more complex as backtrader doesn't provide direct access to individual trades
        # We would need to implement a custom analyzer or extract from broker's history
        # For now, returning an empty list
        return []

    def _plot_results(self, cerebro):
        """Plot backtest results with enhanced visualizations."""
        try:
            # Default backtrader plot with some customization
            figure = cerebro.plot(style='candlestick', barup='green', bardown='red',
                                  volup='green', voldown='red', grid=True, returnfig=True)

            if figure and len(figure) > 0 and len(figure[0]) > 0:
                fig = figure[0][0]

                # Add title and adjust layout
                fig.suptitle(f"Backtest Results - {', '.join(self.symbols)}", fontsize=14)
                fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title

                # Save figure if enabled
                if self.save_results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fig.savefig(os.path.join(self.results_path, f"backtest_plot_{timestamp}.png"))

                plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")

            # Fallback to basic plot
            try:
                cerebro.plot(style='candlestick')
            except:
                self.logger.error("Failed to create even basic plot")


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = {
        "central_trading_bot": {"mode": "backtest"},
        "data_ingestion": {
            "historical_data_path": "./data",
            "timeframe_files": {
                "daily": "daily.csv",
                "4h": "4h.csv",
                "1h": "1h.csv",
                "30min": "30min.csv",
                "15min": "15min.csv",
                "5min": "5min.csv",
                "1min": "1min.csv"
            },
            "delimiter": ",",
            "column_names": ["datetime", "open", "high", "low", "close", "volume"]
        },
        "backtesting": {
            "symbols": ["NQ 03-25", "EURUSD", "GBPUSD", "GBPJPY"],
            "commission": 0.0002,  # 0.02%
            "slippage": 0.0001,  # 0.01%
            "initial_capital": 100000,
            "risk_per_trade": 0.01,  # 1% per trade
            "position_sizing": "risk",
            "threshold": 6.0,
            "plot_results": True,
            "save_results": True,
            "results_path": "./backtest_results"
        }
    }

    backtesting = Backtesting(config)
    backtesting.run()