from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

class APIServer:
    def __init__(self, config, bot):
        self.bot = bot
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 8080)
        self.app = FastAPI()

        # Enable CORS middleware
        # Update 'allow_origins' with actual domains or ports used by your web applications
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "https://your-dashboard.com"],  # Replace with actual origins
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],  # Allowed HTTP methods
            allow_headers=["*"],  # Allow all headers
        )

        # Define API routes
        @self.app.get("/status")
        async def get_status():
            """
            Get the current status of the trading bot.
            Returns state, mode, trade count, daily profit, and daily loss.
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
            """
            if not self.bot:
                raise HTTPException(status_code=503, detail="Bot not initialized")
            self.bot.stop()
            return {"message": "Bot stopping"}

        # Optional: Example endpoint for starting the bot (uncomment if needed)
        # @self.app.post("/start")
        # async def start_bot():
        #     """
        #     Start the trading bot.
        #     """
        #     if not self.bot:
        #         raise HTTPException(status_code=503, detail="Bot not initialized")
        #     self.bot.start()
        #     return {"message": "Bot starting"}

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
        Note: FastAPI doesn't provide a direct stop method; rely on bot state or process termination.
        """
        # Placeholder: Implement graceful shutdown if needed
        pass