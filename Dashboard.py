import logging
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from queue import Queue
import time
from datetime import datetime
import pytz


class Dashboard:
    def __init__(self, config, bot, news_queue):
        """Initialize the Trading Dashboard."""
        self.config = config
        self.bot = bot
        self.news_queue = news_queue
        self.logger = logging.getLogger(__name__)

        # Dashboard settings
        dashboard_config = config.get('dashboard', {})
        self.enable_web_dashboard = dashboard_config.get('enable_web_dashboard', False)
        self.web_dashboard_port = dashboard_config.get('web_dashboard_port', 5000)
        self.refresh_interval = dashboard_config.get('refresh_interval', 1000)  # ms

        # Initialize UI
        self._init_ui()

        # Thread control
        self.running = False
        self.update_thread = None

        self.logger.info("Dashboard initialized")

    def _init_ui(self):
        """Initialize the UI components."""
        try:
            self.root = tk.Tk()
            self.root.title("Trading Bot Dashboard")
            self.root.geometry("900x700")
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

            # Create main frame with padding
            self.main_frame = ttk.Frame(self.root, padding="10")
            self.main_frame.pack(fill=tk.BOTH, expand=True)

            # Create status frame
            status_frame = ttk.LabelFrame(self.main_frame, text="Bot Status")
            status_frame.pack(fill=tk.X, pady=5)

            # Status labels
            self.status_labels = {}
            labels = ["State", "Mode", "Balance", "Trades Today", "Daily Profit", "Daily Loss"]
            for i, label in enumerate(labels):
                ttk.Label(status_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
                self.status_labels[label] = ttk.Label(status_frame, text="Unknown")
                self.status_labels[label].grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)

            # Create positions frame
            positions_frame = ttk.LabelFrame(self.main_frame, text="Open Positions")
            positions_frame.pack(fill=tk.BOTH, expand=True, pady=5)

            # Use ScrolledText for positions
            self.positions_text = scrolledtext.ScrolledText(
                positions_frame, height=8, width=80, wrap=tk.WORD)
            self.positions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Create news frame
            news_frame = ttk.LabelFrame(self.main_frame, text="Market News")
            news_frame.pack(fill=tk.BOTH, expand=True, pady=5)

            # Use ScrolledText for news
            self.news_text = scrolledtext.ScrolledText(
                news_frame, height=10, width=80, wrap=tk.WORD)
            self.news_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Control buttons
            btn_frame = ttk.Frame(self.main_frame)
            btn_frame.pack(fill=tk.X, pady=10)

            self.stop_button = ttk.Button(btn_frame, text="Stop Bot", command=self.stop_bot)
            self.stop_button.pack(side=tk.LEFT, padx=5)

            self.clear_news_button = ttk.Button(btn_frame, text="Clear News",
                                                command=lambda: self.news_text.delete(1.0, tk.END))
            self.clear_news_button.pack(side=tk.LEFT, padx=5)
        except Exception as e:
            self.logger.error(f"Error initializing UI: {str(e)}")
            raise

    def on_closing(self):
        """Handle window closing event."""
        try:
            self.stop()
            self.root.destroy()
            self.logger.info("Dashboard closed by user")
        except Exception as e:
            self.logger.error(f"Error during dashboard closing: {str(e)}")

    def update_dashboard(self):
        """Update dashboard in a separate thread."""
        while self.running:
            try:
                # Get bot status safely
                bot_status = self._get_bot_status()

                # Schedule UI update in the main thread
                self.root.after(0, lambda: self._update_ui(bot_status))

                # Process news queue
                self._process_news_queue()

                # Sleep based on refresh interval
                time.sleep(self.refresh_interval / 1000)
            except Exception as e:
                self.logger.error(f"Error updating dashboard: {str(e)}")
                time.sleep(1)  # Delay before retry

    def _get_bot_status(self):
        """Safely get bot status data."""
        try:
            bot_status = self.bot.get_status()
            risk_status = bot_status.get('risk_management', {})

            return {
                'state': bot_status.get('state', 'Unknown'),
                'mode': bot_status.get('mode', 'Unknown'),
                'balance': float(risk_status.get('current_balance', 0.0)),
                'trades_today': int(risk_status.get('trades_today', 0)),
                'daily_profit': float(risk_status.get('daily_profit', 0.0)),
                'daily_loss': float(risk_status.get('daily_loss', 0.0)),
                'positions': bot_status.get('positions', {})
            }
        except Exception as e:
            self.logger.error(f"Error getting bot status: {str(e)}")
            return {
                'state': 'Error',
                'mode': 'Unknown',
                'balance': 0.0,
                'trades_today': 0,
                'daily_profit': 0.0,
                'daily_loss': 0.0,
                'positions': {}
            }

    def _update_ui(self, status):
        """Update UI with bot status (called in main thread)."""
        try:
            # Update status labels
            self.status_labels["State"].config(text=status.get('state', 'Unknown'))
            self.status_labels["Mode"].config(text=status.get('mode', 'Unknown'))
            self.status_labels["Balance"].config(text=f"${status.get('balance', 0.0):.2f}")
            self.status_labels["Trades Today"].config(text=str(status.get('trades_today', 0)))
            self.status_labels["Daily Profit"].config(text=f"${status.get('daily_profit', 0.0):.2f}")
            self.status_labels["Daily Loss"].config(text=f"${status.get('daily_loss', 0.0):.2f}")

            # Update positions
            self.positions_text.delete(1.0, tk.END)
            for symbol, position in status.get('positions', {}).items():
                if position:
                    pos_info = (
                        f"{symbol}: {position.get('action', 'Unknown')} @ "
                        f"{position.get('entry_price', 0):.5f}, "
                        f"Size: {position.get('position_size', position.get('volume', 0)):.2f}, "
                        f"Ticket: {position.get('ticket', 'N/A')}\n"
                    )
                    self.positions_text.insert(tk.END, pos_info)
        except Exception as e:
            self.logger.error(f"Error updating UI: {str(e)}")

    def _process_news_queue(self):
        """Process news items from the queue."""
        try:
            items_to_process = []
            while not self.news_queue.empty():
                try:
                    items_to_process.append(self.news_queue.get_nowait())
                except Queue.Empty:
                    break

            if items_to_process:
                self.root.after(0, lambda: self._update_news(items_to_process))
        except Exception as e:
            self.logger.error(f"Error processing news queue: {str(e)}")

    def _update_news(self, news_items):
        """Update news display (called in main thread)."""
        try:
            for item in news_items:
                timestamp = datetime.now(pytz.UTC).strftime('%H:%M:%S')
                if isinstance(item, dict) and 'message' in item:
                    msg = f"{timestamp}: {item['message']}\n"
                else:
                    msg = f"{timestamp}: {str(item)}\n"
                self.news_text.insert(tk.END, msg)
                self.news_text.see(tk.END)  # Scroll to bottom
        except Exception as e:
            self.logger.error(f"Error updating news: {str(e)}")

    def stop_bot(self):
        """Stop the trading bot."""
        try:
            if hasattr(self.bot, 'stop') and callable(self.bot.stop):
                success = self.bot.stop()
                if success:
                    self.logger.info("Bot stop requested from dashboard")
                else:
                    self.logger.error("Failed to stop bot from dashboard")
            else:
                self.logger.error("Bot instance lacks stop method")
        except Exception as e:
            self.logger.error(f"Error stopping bot from dashboard: {str(e)}")

    def start(self):
        """Start the dashboard."""
        if self.running:
            self.logger.warning("Dashboard already running")
            return

        try:
            self.running = True
            self.update_thread = threading.Thread(
                target=self.update_dashboard, daemon=True)
            self.update_thread.start()
            self.logger.info("Dashboard started")

            # Start Tkinter main loop
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {str(e)}")
            self.stop()

    def stop(self):
        """Stop the dashboard cleanly."""
        try:
            self.running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2)
                if self.update_thread.is_alive():
                    self.logger.warning("Update thread did not terminate cleanly")
            self.logger.info("Dashboard stopped")
        except Exception as e:
            self.logger.error(f"Error stopping dashboard: {str(e)}")