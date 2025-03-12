import logging
import tkinter as tk
from tkinter import ttk
import threading
from queue import Queue
import time


class Dashboard:
    def __init__(self, config, bot, news_queue):
        """
        Initialize the Dashboard for The 5%ers MT5 trading with EURUSD and GBPJPY.

        Args:
            config (dict): Configuration dictionary with dashboard settings.
            bot (CentralTradingBot): Instance of the trading bot for status updates.
            news_queue (Queue): Queue for receiving news updates from MarketNewsAPI.
        """
        self.config = config
        self.bot = bot
        self.news_queue = news_queue
        self.logger = logging.getLogger(__name__)

        # Dashboard settings from config
        self.enable_web_dashboard = config.get('dashboard', {}).get('enable_web_dashboard', False)
        self.web_dashboard_port = config.get('dashboard', {}).get('web_dashboard_port', 5000)
        self.refresh_interval = config.get('dashboard', {}).get('refresh_interval', 1000)  # ms

        # Tkinter GUI setup
        self.root = tk.Tk()
        self.root.title("Trading Bot Dashboard - The 5%ers MT5")
        self.root.geometry("800x600")

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Labels and displays
        self.state_label = ttk.Label(self.main_frame, text="State: Unknown")
        self.state_label.grid(row=0, column=0, sticky=tk.W)

        self.mode_label = ttk.Label(self.main_frame, text="Mode: Unknown")
        self.mode_label.grid(row=1, column=0, sticky=tk.W)

        self.balance_label = ttk.Label(self.main_frame, text="Balance: $0.00")
        self.balance_label.grid(row=2, column=0, sticky=tk.W)

        self.trades_today_label = ttk.Label(self.main_frame, text="Trades Today: 0")
        self.trades_today_label.grid(row=3, column=0, sticky=tk.W)

        self.daily_profit_label = ttk.Label(self.main_frame, text="Daily Profit: $0.00")
        self.daily_profit_label.grid(row=4, column=0, sticky=tk.W)

        self.daily_loss_label = ttk.Label(self.main_frame, text="Daily Loss: $0.00")
        self.daily_loss_label.grid(row=5, column=0, sticky=tk.W)

        # Positions display
        self.positions_label = ttk.Label(self.main_frame, text="Open Positions:")
        self.positions_label.grid(row=6, column=0, sticky=tk.W)
        self.positions_text = tk.Text(self.main_frame, height=5, width=50)
        self.positions_text.grid(row=7, column=0, columnspan=2, pady=5)

        # News display
        self.news_label = ttk.Label(self.main_frame, text="Market News:")
        self.news_label.grid(row=8, column=0, sticky=tk.W)
        self.news_text = tk.Text(self.main_frame, height=10, width=50)
        self.news_text.grid(row=9, column=0, columnspan=2, pady=5)

        # Buttons
        self.stop_button = ttk.Button(self.main_frame, text="Stop Bot", command=self.stop_bot)
        self.stop_button.grid(row=10, column=0, pady=10)

        # Threading for updates
        self.running = True
        self.update_thread = threading.Thread(target=self.update_dashboard, daemon=True)
        self.update_thread.start()

    def update_dashboard(self):
        """Continuously update the dashboard with bot status and news."""
        while self.running:
            try:
                # Update bot status
                self.state_label.config(text=f"State: {self.bot.state}")
                self.mode_label.config(text=f"Mode: {self.bot.mode}")
                self.balance_label.config(text=f"Balance: ${self.bot.risk_management.current_balance:.2f}")
                self.trades_today_label.config(text=f"Trades Today: {self.bot.risk_management.trades_today}")
                self.daily_profit_label.config(text=f"Daily Profit: ${self.bot.risk_management.daily_profit:.2f}")
                self.daily_loss_label.config(text=f"Daily Loss: ${self.bot.risk_management.daily_loss:.2f}")

                # Update positions
                self.positions_text.delete(1.0, tk.END)
                for symbol, position in self.bot.current_positions.items():
                    if position:
                        pos_info = (f"{symbol}: {position['action']} @ {position['entry_price']:.5f}, "
                                    f"Size: {position['position_size']}, Order ID: {position['order_id']}\n")
                        self.positions_text.insert(tk.END, pos_info)

                # Update news from queue
                while not self.news_queue.empty():
                    news_item = self.news_queue.get()
                    self.news_text.insert(tk.END, f"{datetime.utcnow().strftime('%H:%M:%S')}: {news_item}\n")
                    self.news_text.see(tk.END)

                time.sleep(self.refresh_interval / 1000)  # Convert ms to seconds
            except Exception as e:
                self.logger.error(f"Error updating dashboard: {str(e)}")
                time.sleep(1)  # Wait before retrying

    def stop_bot(self):
        """Stop the trading bot."""
        try:
            self.bot.stop()
            self.logger.info("Bot stop requested from dashboard")
        except Exception as e:
            self.logger.error(f"Error stopping bot from dashboard: {str(e)}")

    def start(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()

    def stop(self):
        """Stop the dashboard and clean up."""
        self.running = False
        self.update_thread.join(timeout=2)
        self.root.quit()
        self.logger.info("Dashboard stopped")


# Example usage (for testing)
if __name__ == "__main__":
    from queue import Queue

    logging.basicConfig(level=logging.INFO)

    # Dummy config and bot for testing
    config = {
        "dashboard": {
            "enable_web_dashboard": False,
            "web_dashboard_port": 5000,
            "refresh_interval": 1000
        },
        "central_trading_bot": {"mode": "live"},
        "symbols": ["EURUSD", "GBPJPY"],
        "risk_management": {
            "current_balance": 100000,
            "trades_today": 2,
            "daily_profit": 500,
            "daily_loss": 200
        }
    }


    class DummyBot:
        def __init__(self):
            self.state = "running"
            self.mode = config["central_trading_bot"]["mode"]
            self.current_positions = {
                "EURUSD": {"action": "buy", "entry_price": 1.0900, "position_size": 0.1, "order_id": "12345"},
                "GBPJPY": None
            }
            self.risk_management = type('Risk', (), {
                "current_balance": config["risk_management"]["current_balance"],
                "trades_today": config["risk_management"]["trades_today"],
                "daily_profit": config["risk_management"]["daily_profit"],
                "daily_loss": config["risk_management"]["daily_loss"]
            })()

        def stop(self):
            self.state = "stopped"


    news_queue = Queue()
    bot = DummyBot()
    dashboard = Dashboard(config, bot, news_queue)
    dashboard.start()