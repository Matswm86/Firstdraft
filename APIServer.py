from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading


class APIServer:
    def __init__(self, config, bot):
        """
        Initialize the APIServer with configuration and bot instance.

        Args:
            config (dict): Configuration dictionary with API server settings.
            bot (CentralTradingBot): Instance of CentralTradingBot for status and control.
        """
        self.bot = bot
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 8080)
        self.app = FastAPI()

        # Enable CORS middleware for web clients
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "https://your-dashboard.com"],  # Update with actual origins
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"]
        )

        # Define API routes
        @self.app.get("/status")
        async def get_status():
            """
            Get the current status of the trading bot.

            Returns:
                dict: Status data including state, mode, and risk metrics.
            """
            if not self.bot:
                raise HTTPException(status_code=503, detail="Bot not initialized")
            return {
                "state": self.bot.state,
                "mode": self.bot.mode,
                "trade_count": self.bot.risk_management.trade_count,
                "daily_profit": self.bot.risk_management.daily_profit,
                "daily_loss": self.bot.risk_management.daily_loss
            }

        @self.app.post("/stop")
        async def stop_bot():
            """
            Stop the trading bot.

            Returns:
                dict: Confirmation message.
            """
            if not self.bot:
                raise HTTPException(status_code=503, detail="Bot not initialized")
            self.bot.stop()
            return {"message": "Bot stopping"}

    def start(self):
        """
        Start the API server in a separate thread.
        """
        self.server_thread = threading.Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={"host": self.host, "port": self.port},
            daemon=True
        )
        self.server_thread.start()

    def stop(self):
        """
        Stop the API server.

        Note: FastAPI doesn't provide a direct stop method; relies on bot state or process termination.
        """
        # Placeholder for future graceful shutdown implementation
        pass

# Example usage (for testing, commented out)
# if __name__ == "__main__":
#     from central_trading_bot import CentralTradingBot
#     config = {"api_server": {"host": "127.0.0.1", "port": 8080}}
#     bot = CentralTradingBot({"central_trading_bot": {"mode": "live"}})  # Dummy config
#     server = APIServer(config["api_server"], bot)
#     server.start()