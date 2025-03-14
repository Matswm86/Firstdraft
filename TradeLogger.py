import logging
import csv
import os
import json
import pytz
from datetime import datetime, date, timedelta
import threading
import shutil
import time


class TradeLogger:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.log_file = config.get('trade_log_file', 'logs/trades.csv')
        self.log_level = config.get('log_level', 'INFO').upper()
        self.real_time_log_level = config.get('real_time_log_level', 'DEBUG').upper()

        self.backup_dir = config.get('backup_dir', 'logs/backups')
        self.max_log_size = config.get('max_log_size', 10 * 1024 * 1024)  # 10 MB default
        self.max_log_age = config.get('max_log_age', 30)  # 30 days default
        self.performance_file = config.get('performance_file', 'logs/performance.json')
        self.include_execution_time = config.get('include_execution_time', True)
        self.include_slippage = config.get('include_slippage', True)

        self.csv_fields = [
            'timestamp', 'symbol', 'action', 'entry_price', 'exit_price',
            'position_size', 'stop_loss', 'take_profit', 'profit_loss', 'pips',
            'execution_time', 'slippage', 'commission', 'ticket'
        ]

        self.trade_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_trade': 0.0,
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'first_trade_date': None,
            'last_trade_date': None,
            'symbols_traded': set(),  # Initialize as set
            'by_symbol': {},
            'by_day': {},
            'updated_at': None
        }

        self.file_lock = threading.RLock()

        self.setup_trade_logging()
        self._load_performance_data()

    def setup_trade_logging(self):
        try:
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            os.makedirs(self.backup_dir, exist_ok=True)

            write_headers = not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0

            with self.file_lock:
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                    if write_headers:
                        writer.writeheader()

            self._check_log_rotation()

            self.logger.info(f"Trade log file set up at {self.log_file}")
        except Exception as e:
            self.logger.error(f"Failed to set up trade log file: {str(e)}")

    def _check_log_rotation(self):
        try:
            if not os.path.exists(self.log_file):
                return

            file_size = os.path.getsize(self.log_file)
            if file_size > self.max_log_size:
                self._rotate_log_file()

            self._purge_old_backups()
        except Exception as e:
            self.logger.error(f"Error checking log rotation: {str(e)}")

    def _rotate_log_file(self):
        try:
            if not os.path.exists(self.log_file):
                return

            timestamp = datetime.now(pytz.UTC).strftime('%Y%m%d_%H%M%S')
            backup_name = f"trades_{timestamp}.csv"
            backup_path = os.path.join(self.backup_dir, backup_name)

            with self.file_lock:
                shutil.copy2(self.log_file, backup_path)

                with open(self.log_file, 'r', encoding='utf-8') as source:
                    header = source.readline()

                with open(self.log_file, 'w', encoding='utf-8') as target:
                    target.write(header)

            self.logger.info(f"Log file rotated to {backup_path}")
        except Exception as e:
            self.logger.error(f"Error rotating log file: {str(e)}")

    def _purge_old_backups(self):
        try:
            if not os.path.exists(self.backup_dir):
                return

            now = datetime.now(pytz.UTC)
            cutoff = now - timedelta(days=self.max_log_age)

            for filename in os.listdir(self.backup_dir):
                if not filename.startswith('trades_') or not filename.endswith('.csv'):
                    continue

                file_path = os.path.join(self.backup_dir, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path), tz=pytz.UTC)

                if file_mtime < cutoff:
                    os.remove(file_path)
                    self.logger.info(f"Deleted old backup log: {filename}")
        except Exception as e:
            self.logger.error(f"Error purging old backups: {str(e)}")

    def _load_performance_data(self):
        """Load performance data from file if available"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert date strings back to datetime objects
                if 'first_trade_date' in data and data['first_trade_date']:
                    data['first_trade_date'] = datetime.fromisoformat(data['first_trade_date'])
                if 'last_trade_date' in data and data['last_trade_date']:
                    data['last_trade_date'] = datetime.fromisoformat(data['last_trade_date'])

                # Ensure symbols_traded is a set
                if 'symbols_traded' in data:
                    if isinstance(data['symbols_traded'], list):
                        data['symbols_traded'] = set(data['symbols_traded'])
                    elif not isinstance(data['symbols_traded'], set):
                        self.logger.warning(f"Unexpected type for symbols_traded: {type(data['symbols_traded'])}. Resetting to empty set.")
                        data['symbols_traded'] = set()

                # Ensure by_day symbols are sets
                if 'by_day' in data:
                    for day in data['by_day']:
                        if 'symbols' in data['by_day'][day]:
                            if isinstance(data['by_day'][day]['symbols'], list):
                                data['by_day'][day]['symbols'] = set(data['by_day'][day]['symbols'])
                            elif not isinstance(data['by_day'][day]['symbols'], set):
                                self.logger.warning(f"Unexpected type for by_day[{day}]['symbols']: {type(data['by_day'][day]['symbols'])}. Resetting to empty set.")
                                data['by_day'][day]['symbols'] = set()

                # Update metrics with type safety
                for key in self.trade_metrics:
                    if key in data:
                        if key == 'symbols_traded' or key == 'by_day':
                            self.trade_metrics[key] = data[key]  # Already converted to set where needed
                        elif isinstance(self.trade_metrics[key], (int, float)):
                            self.trade_metrics[key] = float(data[key]) if key in ['total_profit', 'total_loss', 'largest_win', 'largest_loss', 'average_win', 'average_loss', 'win_rate', 'profit_factor', 'average_trade', 'total_commission', 'total_slippage'] else int(data[key])
                        else:
                            self.trade_metrics[key] = data[key]

                self.logger.info(f"Loaded performance data: {self.trade_metrics['total_trades']} trades")
            else:
                self.logger.info("No performance data file found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading performance data: {str(e)}. Starting fresh.")
            # Reset to defaults, ensuring symbols_traded is a set
            self.trade_metrics = {k: v if k != 'symbols_traded' else set() for k, v in self.trade_metrics.items()}

    def _save_performance_data(self):
        try:
            perf_dir = os.path.dirname(self.performance_file)
            if perf_dir and not os.path.exists(perf_dir):
                os.makedirs(perf_dir, exist_ok=True)

            self.trade_metrics['updated_at'] = datetime.now(pytz.UTC).isoformat()

            save_data = self.trade_metrics.copy()

            if save_data['first_trade_date']:
                save_data['first_trade_date'] = save_data['first_trade_date'].isoformat()
            if save_data['last_trade_date']:
                save_data['last_trade_date'] = save_data['last_trade_date'].isoformat()

            # Convert sets to lists for JSON serialization
            if isinstance(save_data['symbols_traded'], set):
                save_data['symbols_traded'] = list(save_data['symbols_traded'])

            for day in save_data['by_day']:
                if isinstance(save_data['by_day'][day]['symbols'], set):
                    save_data['by_day'][day]['symbols'] = list(save_data['by_day'][day]['symbols'])

            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)

            self.logger.info(f"Saved performance data to {self.performance_file}")
        except Exception as e:
            self.logger.error(f"Error saving performance data: {str(e)}")

    def _validate_trade_data(self, trade):
        required_fields = ['symbol', 'action', 'entry_price']
        for field in required_fields:
            if field not in trade or trade[field] is None:
                return False, f"Missing required field: {field}"

        valid_actions = ['buy', 'sell']
        if trade.get('action', '').lower() not in valid_actions:
            return False, f"Invalid action: {trade.get('action')}"

        numeric_fields = ['entry_price', 'exit_price', 'position_size', 'stop_loss', 'take_profit', 'profit_loss']
        for field in numeric_fields:
            if field in trade and trade[field] is not None:
                try:
                    float(trade[field])
                except (ValueError, TypeError):
                    return False, f"Invalid numeric value for {field}: {trade[field]}"

        return True, ""

    def _prepare_trade_data(self, trade):
        trade_data = {}

        trade_data['timestamp'] = trade.get('timestamp', datetime.now(pytz.UTC).isoformat())
        trade_data['symbol'] = trade.get('symbol', 'Unknown')
        trade_data['action'] = trade.get('action', 'Unknown').lower()

        numeric_fields = [
            'entry_price', 'exit_price', 'position_size', 'stop_loss',
            'take_profit', 'profit_loss', 'commission', 'slippage', 'pips'
        ]
        for field in numeric_fields:
            if field in trade and trade[field] is not None:
                try:
                    trade_data[field] = float(trade[field])
                except (ValueError, TypeError):
                    trade_data[field] = 0.0
            else:
                trade_data[field] = 0.0

        if 'position_size' not in trade and 'volume' in trade and trade['volume'] is not None:
            try:
                trade_data['position_size'] = float(trade['volume'])
            except (ValueError, TypeError):
                trade_data['position_size'] = 0.0

        trade_data['ticket'] = str(trade.get('ticket', 'N/A'))

        if self.include_execution_time:
            trade_data['execution_time'] = float(trade.get('execution_time', 0.0))

        for field in self.csv_fields:
            if field not in trade_data:
                trade_data[field] = '' if field == 'ticket' else 0.0

        return trade_data

    def log_trade(self, trade):
        try:
            is_valid, error = self._validate_trade_data(trade)
            if not is_valid:
                self.logger.error(f"Invalid trade data: {error}")
                return False

            trade_data = self._prepare_trade_data(trade)

            profit_loss = trade_data.get('profit_loss', 0.0)
            log_message = (
                f"Trade: {trade_data['symbol']} {trade_data['action'].upper()}, "
                f"Size: {trade_data['position_size']:.2f}, "
                f"Entry: {trade_data['entry_price']:.5f}, "
                f"Exit: {trade_data['exit_price']:.5f}, "
                f"P/L: {profit_loss:.2f}, "
                f"Ticket: {trade_data['ticket']}"
            )

            if profit_loss >= 0:
                self.logger.info(log_message)
            else:
                self.logger.warning(log_message)

            with self.file_lock:
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                    writer.writerow(trade_data)

            self._update_metrics(trade_data)
            self._check_log_rotation()

            return True
        except Exception as e:
            self.logger.error(f"Failed to log trade: {str(e)}")
            return False

    def log_trades(self, trades):
        if not trades:
            self.logger.info("No trades to log")
            return 0

        success_count = 0
        for trade in trades:
            if self.log_trade(trade):
                success_count += 1

        self.logger.info(f"Logged {success_count}/{len(trades)} trades")
        return success_count

    def _update_metrics(self, trade):
        try:
            profit_loss = float(trade.get('profit_loss', 0.0))
            symbol = trade.get('symbol', 'Unknown')
            timestamp = trade.get('timestamp')
            commission = float(trade.get('commission', 0.0))
            slippage = float(trade.get('slippage', 0.0))

            if timestamp:
                try:
                    trade_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00') if timestamp.endswith('Z') else timestamp)
                except ValueError:
                    trade_date = datetime.now(pytz.UTC)
            else:
                trade_date = datetime.now(pytz.UTC)

            if not self.trade_metrics['first_trade_date'] or trade_date < self.trade_metrics['first_trade_date']:
                self.trade_metrics['first_trade_date'] = trade_date
            if not self.trade_metrics['last_trade_date'] or trade_date > self.trade_metrics['last_trade_date']:
                self.trade_metrics['last_trade_date'] = trade_date

            self.trade_metrics['total_trades'] = int(self.trade_metrics['total_trades'] + 1)
            # Ensure symbols_traded is a set before adding
            if not isinstance(self.trade_metrics['symbols_traded'], set):
                self.logger.warning(f"symbols_traded is not a set: {type(self.trade_metrics['symbols_traded'])}. Converting to set.")
                self.trade_metrics['symbols_traded'] = set()
            self.trade_metrics['symbols_traded'].add(symbol)
            self.trade_metrics['total_commission'] = float(self.trade_metrics['total_commission'] + commission)
            self.trade_metrics['total_slippage'] = float(self.trade_metrics['total_slippage'] + slippage)

            if profit_loss > 0:
                self.trade_metrics['winning_trades'] = int(self.trade_metrics['winning_trades'] + 1)
                self.trade_metrics['total_profit'] = float(self.trade_metrics['total_profit'] + profit_loss)
                self.trade_metrics['largest_win'] = float(max(self.trade_metrics['largest_win'], profit_loss))
            elif profit_loss < 0:
                self.trade_metrics['losing_trades'] = int(self.trade_metrics['losing_trades'] + 1)
                self.trade_metrics['total_loss'] = float(self.trade_metrics['total_loss'] + abs(profit_loss))
                self.trade_metrics['largest_loss'] = float(max(self.trade_metrics['largest_loss'], abs(profit_loss)))

            if self.trade_metrics['winning_trades'] > 0:
                self.trade_metrics['average_win'] = float(self.trade_metrics['total_profit'] / self.trade_metrics['winning_trades'])
            if self.trade_metrics['losing_trades'] > 0:
                self.trade_metrics['average_loss'] = float(self.trade_metrics['total_loss'] / self.trade_metrics['losing_trades'])

            if self.trade_metrics['total_trades'] > 0:
                self.trade_metrics['win_rate'] = float(self.trade_metrics['winning_trades'] / self.trade_metrics['total_trades'])

            if self.trade_metrics['total_loss'] > 0:
                self.trade_metrics['profit_factor'] = float(self.trade_metrics['total_profit'] / self.trade_metrics['total_loss'])

            net_profit = self.trade_metrics['total_profit'] - self.trade_metrics['total_loss']
            self.trade_metrics['average_trade'] = float(net_profit / self.trade_metrics['total_trades']) if self.trade_metrics['total_trades'] > 0 else 0.0

            if symbol not in self.trade_metrics['by_symbol']:
                self.trade_metrics['by_symbol'][symbol] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'total_loss': 0.0,
                    'win_rate': 0.0
                }

            symbol_metrics = self.trade_metrics['by_symbol'][symbol]
            symbol_metrics['total_trades'] = int(symbol_metrics['total_trades'] + 1)
            if profit_loss > 0:
                symbol_metrics['winning_trades'] = int(symbol_metrics['winning_trades'] + 1)
                symbol_metrics['total_profit'] = float(symbol_metrics['total_profit'] + profit_loss)
            elif profit_loss < 0:
                symbol_metrics['losing_trades'] = int(symbol_metrics['losing_trades'] + 1)
                symbol_metrics['total_loss'] = float(symbol_metrics['total_loss'] + abs(profit_loss))
            symbol_metrics['win_rate'] = float(symbol_metrics['winning_trades'] / symbol_metrics['total_trades']) if symbol_metrics['total_trades'] > 0 else 0.0

            day_key = trade_date.strftime('%Y-%m-%d')
            if day_key not in self.trade_metrics['by_day']:
                self.trade_metrics['by_day'][day_key] = {
                    'total_trades': 0,
                    'net_profit': 0.0,
                    'symbols': set()  # Initialize as set
                }

            day_metrics = self.trade_metrics['by_day'][day_key]
            day_metrics['total_trades'] = int(day_metrics['total_trades'] + 1)
            day_metrics['net_profit'] = float(day_metrics['net_profit'] + profit_loss)
            # Ensure day_metrics['symbols'] is a set
            if not isinstance(day_metrics['symbols'], set):
                self.logger.warning(f"by_day[{day_key}]['symbols'] is not a set: {type(day_metrics['symbols'])}. Converting to set.")
                day_metrics['symbols'] = set()
            day_metrics['symbols'].add(symbol)

            self._save_performance_data()
        except Exception as e:
            self.logger.error(f"Error updating trade metrics: {str(e)}")

    def get_performance_summary(self):
        try:
            net_profit = self.trade_metrics['total_profit'] - self.trade_metrics['total_loss']

            summary = {
                'total_trades': int(self.trade_metrics['total_trades']),
                'winning_trades': int(self.trade_metrics['winning_trades']),
                'losing_trades': int(self.trade_metrics['losing_trades']),
                'win_rate': float(self.trade_metrics['win_rate']),
                'profit_factor': float(self.trade_metrics['profit_factor']),
                'net_profit': float(net_profit),
                'total_profit': float(self.trade_metrics['total_profit']),
                'total_loss': float(self.trade_metrics['total_loss']),
                'largest_win': float(self.trade_metrics['largest_win']),
                'largest_loss': float(self.trade_metrics['largest_loss']),
                'average_win': float(self.trade_metrics['average_win']),
                'average_loss': float(self.trade_metrics['average_loss']),
                'average_trade': float(self.trade_metrics['average_trade']),
                'total_commission': float(self.trade_metrics['total_commission']),
                'total_slippage': float(self.trade_metrics['total_slippage']),
                'symbols_traded': len(self.trade_metrics['symbols_traded']),
                'first_trade_date': self.trade_metrics['first_trade_date'].isoformat() if self.trade_metrics['first_trade_date'] else None,
                'last_trade_date': self.trade_metrics['last_trade_date'].isoformat() if self.trade_metrics['last_trade_date'] else None,
                'trading_days': len(self.trade_metrics['by_day'])
            }
            return summary
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {str(e)}")
            return {}

    def get_trades_by_symbol(self, symbol=None):
        trades = []
        try:
            if not os.path.exists(self.log_file):
                return trades

            with self.file_lock:
                with open(self.log_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if symbol is None or row['symbol'] == symbol:
                            numeric_fields = ['entry_price', 'exit_price', 'position_size', 'profit_loss', 'stop_loss', 'take_profit', 'pips', 'commission', 'slippage', 'execution_time']
                            for field in numeric_fields:
                                if field in row and row[field]:
                                    try:
                                        row[field] = float(row[field])
                                    except (ValueError, TypeError):
                                        row[field] = 0.0
                            trades.append(row)
        except Exception as e:
            self.logger.error(f"Error getting trades by symbol: {str(e)}")
        return trades

    def get_trades_by_date_range(self, start_date=None, end_date=None):
        trades = []
        try:
            if not os.path.exists(self.log_file):
                return trades

            if end_date is None:
                end_date = datetime.now(pytz.UTC)
            if start_date is None:
                start_date = end_date - timedelta(days=30)

            start_date = start_date.replace(tzinfo=pytz.UTC) if start_date.tzinfo is None else start_date
            end_date = end_date.replace(tzinfo=pytz.UTC) if end_date.tzinfo is None else end_date

            with self.file_lock:
                with open(self.log_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            trade_date = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00') if row['timestamp'].endswith('Z') else row['timestamp'])
                            if start_date <= trade_date <= end_date:
                                numeric_fields = ['entry_price', 'exit_price', 'position_size', 'profit_loss', 'stop_loss', 'take_profit', 'pips', 'commission', 'slippage', 'execution_time']
                                for field in numeric_fields:
                                    if field in row and row[field]:
                                        try:
                                            row[field] = float(row[field])
                                        except (ValueError, TypeError):
                                            row[field] = 0.0
                                trades.append(row)
                        except (ValueError, KeyError):
                            continue
        except Exception as e:
            self.logger.error(f"Error getting trades by date range: {str(e)}")
        return trades

    def calculate_daily_performance(self, days=30):
        daily_performance = []
        try:
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=days)
            trades = self.get_trades_by_date_range(start_date, end_date)

            trades_by_day = {}
            for trade in trades:
                try:
                    trade_date = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00') if trade['timestamp'].endswith('Z') else trade['timestamp'])
                    day_key = trade_date.strftime('%Y-%m-%d')
                    if day_key not in trades_by_day:
                        trades_by_day[day_key] = []
                    trades_by_day[day_key].append(trade)
                except (ValueError, KeyError):
                    continue

            for day_key, day_trades in trades_by_day.items():
                total_trades = len(day_trades)
                winning_trades = sum(1 for t in day_trades if float(t.get('profit_loss', 0)) > 0)
                losing_trades = sum(1 for t in day_trades if float(t.get('profit_loss', 0)) < 0)
                net_profit = sum(float(t.get('profit_loss', 0)) for t in day_trades)

                daily_performance.append({
                    'date': day_key,
                    'total_trades': int(total_trades),
                    'winning_trades': int(winning_trades),
                    'losing_trades': int(losing_trades),
                    'win_rate': float(winning_trades / total_trades) if total_trades > 0 else 0.0,
                    'net_profit': float(net_profit),
                    'symbols': list(set(t.get('symbol', 'Unknown') for t in day_trades))
                })

            daily_performance.sort(key=lambda x: x['date'])
        except Exception as e:
            self.logger.error(f"Error calculating daily performance: {str(e)}")
        return daily_performance

    def analyze_trades(self, trades=None):
        try:
            if trades is None:
                trades = self.get_trades_by_symbol()

            if not trades:
                return {"status": "no_data", "message": "No trades to analyze"}

            hours = {i: {'count': 0, 'profit_loss': 0.0} for i in range(24)}
            days = {i: {'count': 0, 'profit_loss': 0.0} for i in range(7)}  # 0 = Monday, 6 = Sunday
            symbols = {}
            trades_chronological = sorted(trades, key=lambda x: x.get('timestamp', ''), reverse=False)
            cumulative_pl = []
            running_total = 0.0

            for trade in trades_chronological:
                try:
                    trade_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00') if trade['timestamp'].endswith('Z') else trade['timestamp'])
                    hour = trade_time.hour
                    day = trade_time.weekday()
                    profit_loss = float(trade.get('profit_loss', 0.0))
                    symbol = trade.get('symbol', 'Unknown')

                    hours[hour]['count'] = int(hours[hour]['count'] + 1)
                    hours[hour]['profit_loss'] = float(hours[hour]['profit_loss'] + profit_loss)

                    days[day]['count'] = int(days[day]['count'] + 1)
                    days[day]['profit_loss'] = float(days[day]['profit_loss'] + profit_loss)

                    if symbol not in symbols:
                        symbols[symbol] = {
                            'count': 0,
                            'profit_loss': 0.0,
                            'winning': 0,
                            'losing': 0
                        }
                    symbols[symbol]['count'] = int(symbols[symbol]['count'] + 1)
                    symbols[symbol]['profit_loss'] = float(symbols[symbol]['profit_loss'] + profit_loss)
                    if profit_loss > 0:
                        symbols[symbol]['winning'] = int(symbols[symbol]['winning'] + 1)
                    elif profit_loss < 0:
                        symbols[symbol]['losing'] = int(symbols[symbol]['losing'] + 1)

                    running_total = float(running_total + profit_loss)
                    cumulative_pl.append({
                        'timestamp': trade.get('timestamp'),
                        'profit_loss': profit_loss,
                        'cumulative': running_total
                    })
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Skipping trade in analysis due to error: {str(e)}")
                    continue

            for symbol in symbols:
                total = symbols[symbol]['count']
                symbols[symbol]['win_rate'] = float(symbols[symbol]['winning'] / total) if total > 0 else 0.0

            best_hour = max(hours.items(), key=lambda x: x[1]['profit_loss'])[0] if hours else None
            worst_hour = min(hours.items(), key=lambda x: x[1]['profit_loss'])[0] if hours else None
            best_day = max(days.items(), key=lambda x: x[1]['profit_loss'])[0] if days else None
            worst_day = min(days.items(), key=lambda x: x[1]['profit_loss'])[0] if days else None

            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            return {
                "status": "success",
                "total_trades": len(trades),
                "hours": {str(h): {"count": int(hours[h]['count']), "profit_loss": float(hours[h]['profit_loss'])} for h in hours},
                "best_hour": best_hour,
                "worst_hour": worst_hour,
                "days": {day_names[d]: {"count": int(days[d]['count']), "profit_loss": float(days[d]['profit_loss'])} for d in days},
                "best_day": day_names[best_day] if best_day is not None else None,
                "worst_day": day_names[worst_day] if worst_day is not None else None,
                "symbols": {s: {k: float(v) if k in ['profit_loss', 'win_rate'] else int(v) for k, v in stats.items()} for s, stats in symbols.items()},
                "performance_trend": cumulative_pl
            }
        except Exception as e:
            self.logger.error(f"Error analyzing trades: {str(e)}")
            return {"status": "error", "message": str(e)}

    def log_error(self, error_message, context=None):
        try:
            error_log_file = self.config.get('error_log_file', 'logs/trade_errors.csv')

            error_log_dir = os.path.dirname(error_log_file)
            if error_log_dir and not os.path.exists(error_log_dir):
                os.makedirs(error_log_dir, exist_ok=True)

            context_str = json.dumps(context) if context else ""

            error_data = {
                'timestamp': datetime.now(pytz.UTC).isoformat(),
                'error': error_message,
                'context': context_str
            }

            write_headers = not os.path.exists(error_log_file) or os.path.getsize(error_log_file) == 0

            with open(error_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'error', 'context'])
                if write_headers:
                    writer.writeheader()
                writer.writerow(error_data)

            self.logger.error(f"Trade Error: {error_message}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to log error: {str(e)}")
            return False

    def export_trades(self, file_path, format='csv', filter_func=None):
        try:
            trades = self.get_trades_by_symbol()

            if filter_func and callable(filter_func):
                trades = [t for t in trades if filter_func(t)]

            if not trades:
                self.logger.warning("No trades to export")
                return False

            export_dir = os.path.dirname(file_path)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir, exist_ok=True)

            if format.lower() == 'csv':
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                    writer.writeheader()
                    writer.writerows(trades)
            elif format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(trades, f, indent=2)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False

            self.logger.info(f"Exported {len(trades)} trades to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export trades: {str(e)}")
            return False

    def calculate_drawdown(self):
        try:
            trades = self.get_trades_by_symbol()
            trades.sort(key=lambda x: x.get('timestamp', ''))

            equity_curve = []
            running_total = 0.0
            peak = 0.0
            current_drawdown = 0.0
            max_drawdown = 0.0
            drawdown_start = None
            drawdown_end = None
            drawdown_start_value = 0.0
            max_drawdown_start = None
            max_drawdown_end = None

            for trade in trades:
                profit_loss = float(trade.get('profit_loss', 0.0))
                timestamp = trade.get('timestamp', datetime.now(pytz.UTC).isoformat())

                running_total += profit_loss
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': running_total
                })

                if running_total > peak:
                    peak = running_total
                    current_drawdown = 0.0
                    drawdown_start = None
                else:
                    drawdown = peak - running_total
                    if drawdown > current_drawdown:
                        current_drawdown = drawdown
                        if drawdown_start is None:
                            drawdown_start = timestamp
                            drawdown_start_value = peak
                        drawdown_end = timestamp

                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                        max_drawdown_start = drawdown_start
                        max_drawdown_end = drawdown_end

            max_drawdown_pct = float(max_drawdown / drawdown_start_value) if drawdown_start_value > 0 else 0.0

            return {
                "max_drawdown": float(max_drawdown),
                "max_drawdown_pct": max_drawdown_pct,
                "drawdown_start": max_drawdown_start,
                "drawdown_end": max_drawdown_end,
                "equity_curve": equity_curve
            }
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {str(e)}")
            return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}

    def backup_logs(self, backup_dir=None):
        try:
            if backup_dir is None:
                backup_dir = self.backup_dir

            os.makedirs(backup_dir, exist_ok=True)

            timestamp = datetime.now(pytz.UTC).strftime('%Y%m%d_%H%M%S')

            if os.path.exists(self.log_file):
                log_name = os.path.basename(self.log_file)
                backup_path = os.path.join(backup_dir, f"{log_name}_{timestamp}")
                shutil.copy2(self.log_file, backup_path)

            if os.path.exists(self.performance_file):
                perf_name = os.path.basename(self.performance_file)
                backup_path = os.path.join(backup_dir, f"{perf_name}_{timestamp}")
                shutil.copy2(self.performance_file, backup_path)

            self.logger.info(f"Created log backup at {backup_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup logs: {str(e)}")
            return False

    def generate_report(self):
        try:
            summary = self.get_performance_summary()
            daily = self.calculate_daily_performance()
            drawdown = self.calculate_drawdown()
            trades = self.get_trades_by_symbol()
            analysis = self.analyze_trades(trades)

            report = {
                "summary": summary,
                "daily_performance": daily,
                "drawdown": drawdown,
                "analysis": analysis,
                "generated_at": datetime.now(pytz.UTC).isoformat(),
                "recent_trades": trades[-10:] if len(trades) > 10 else trades
            }

            return report
        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            return {"error": str(e), "generated_at": datetime.now(pytz.UTC).isoformat()}