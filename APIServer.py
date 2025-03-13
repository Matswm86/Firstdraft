import logging
from fastapi import FastAPI, HTTPException
import uvicorn
import threading
import time

class APIServer:
    def __init__(self, config, bot):
        """
        Initialize the APIServer for The 5%ers MT5 trading with EURUSD and GBPJPY.

        Args:
            config (dict): Configuration dictionary with API server settings.
            bot (CentralTradingBot): Instance of CentralTradingBot for status and control.
        """
        self.config = config
        self.bot = bot
        self.logger = logging.getLogger(__name__)

        # API server settings from config
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 8080)

        # Initialize FastAPI app
        self.app = FastAPI(title="Trading Bot API Server - The 5%ers MT5")
        self.running = False
        self.server_thread = None

        # Define API endpoints
        self._setup_endpoints()

    def _setup_endpoints(self):
        """Set up FastAPI endpoints for bot status and control."""

        @self.app.get("/status")
        async def get_status():
            """
            Get the current status of the trading bot.

            Returns:
                dict: Status data including state, mode, risk metrics, and positions.
            """
            try:
                status = {
                    "state": self.bot.state,
                    "mode": self.bot.mode,
                    "balance": self.bot.risk_management.current_balance,
                    "trades_today": self.bot.risk_management.trades_today,
                    "daily_profit": self.bot.risk_management.daily_profit,
                    "daily_loss": self.bot.risk_management.daily_loss,
                    "positions": {
                        symbol: {
                            "action": pos["action"],
                            "entry_price": pos["entry_price"],
                            "size": pos["position_size"],
                            "order_id": pos["order_id"]
                        } if pos else None
                        for symbol, pos in self.bot.current_positions.items()
                    }
                }
                self.logger.info("API request: Retrieved bot status")
                return status
            except AttributeError:
                self.logger.error("Bot not properly initialized")
                raise HTTPException(status_code=503, detail="Bot not initialized")
            except Exception as e:
                self.logger.error(f"Error retrieving status: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/stop")
        async def stop_bot():
            """
            Stop the trading bot.

            Returns:
                dict: Confirmation message.
            """
            try:
                self.bot.stop()
                self.logger.info("API request: Bot stopped")
                return {"message": "Bot stopping"}
            except AttributeError:
                self.logger.error("Bot not properly initialized")
                raise HTTPException(status_code=503, detail="Bot not initialized")
            except Exception as e:
                self.logger.error(f"Error stopping bot: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    def start(self):
        """Start the API server in a separate thread."""
        if self.running:
            self.logger.warning("API server is already running")
            return

        self.running = True
        self.server_thread = threading.Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={"host": self.host, "port": self.port, "log_level": "info"},
            daemon=True
        )
        self.server_thread.start()
        self.logger.info(f"API server started at http://{self.host}:{self.port}")

    def stop(self):
        """Stop the API server."""
        if not self.running:
            self.logger.warning("API server is not running")
            return

        self.running = False
        # Note: FastAPI/uvicorn doesn't have a clean shutdown method from Python;
        # typically, you'd terminate the process or use a signal handler in production
        # For this implementation, we let the daemon thread terminate with the main app
        self.logger.info("API server stop requested; will terminate with main process")


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Dummy config and bot for testing
    config = {
        "api_server": {
            "host": "127.0.0.1",
            "port": 8080
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


    bot = DummyBot()
    api_server = APIServer(config["api_server"], bot)
    api_server.start()

    # Keep the main thread alive for testing
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        api_server.stop()