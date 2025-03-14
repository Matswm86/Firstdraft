import logging
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import threading
import asyncio
import time
import uuid
import secrets
from datetime import datetime, timedelta
import pytz
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import signal
import os
import sys


# Define data models for requests and responses
class StatusResponse(BaseModel):
    state: str
    mode: str
    balance: float
    trades_today: int
    daily_profit: float
    daily_loss: float
    positions: Dict[str, Optional[Dict[str, Any]]]
    timestamp: str


class ModifyRiskRequest(BaseModel):
    risk_level: float = Field(..., gt=0, le=1, description="Risk level between 0 and 1")
    duration_minutes: int = Field(60, ge=5, le=1440, description="Duration in minutes")


class ClosePositionRequest(BaseModel):
    ticket: str = Field(..., description="Position ticket to close")


class ModifyPositionRequest(BaseModel):
    ticket: str = Field(..., description="Position ticket to modify")
    stop_loss: Optional[float] = Field(None, description="New stop loss level")
    take_profit: Optional[float] = Field(None, description="New take profit level")


class ServerStatus(BaseModel):
    uptime: str
    api_version: str
    clients_connected: int
    requests_processed: int
    start_time: str


class APIServer:
    """
    API Server for trading bot control and monitoring.
    Provides RESTful endpoints for status, control, and configuration.
    """

    def __init__(self, config, bot):
        """
        Initialize APIServer with configuration settings.

        Args:
            config (dict): Configuration dictionary with API server settings
            bot (CentralTradingBot): Instance of CentralTradingBot for status and control
        """
        self.config = config
        self.bot = bot
        self.logger = logging.getLogger(__name__)

        # API server settings
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 8080)
        self.api_version = "1.0.0"
        self.api_title = "Trading Bot API"

        # Security settings
        self.enable_auth = config.get('enable_auth', True)
        self.api_key = config.get('api_key', os.environ.get('TRADING_API_KEY', self._generate_api_key()))
        self.cors_origins = config.get('cors_origins', ["*"])

        # Monitoring settings
        self.max_requests_per_minute = config.get('max_requests_per_minute', 60)
        self.rate_limit_window = timedelta(minutes=1)

        # State tracking
        self.running = False
        self.server_start_time = None
        self.server_thread = None
        self.shutdown_event = threading.Event()
        self.clients_connected = 0
        self.requests_processed = 0
        self.request_log = {}  # To track request rates by IP

        # Initialize FastAPI app
        self.app = self._create_app()

        self.logger.info(f"API Server initialized with host={self.host}, port={self.port}")

    def _create_app(self):
        """Create and configure the FastAPI application"""
        app = FastAPI(
            title=self.api_title,
            description="API for trading bot control and monitoring",
            version=self.api_version
        )

        # Add security
        if self.enable_auth:
            self.api_key_header = APIKeyHeader(name="X-API-Key")

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add rate limiting middleware
        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            client_ip = request.client.host
            now = datetime.now(pytz.UTC)

            # Initialize or clean up request log for this client
            if client_ip not in self.request_log:
                self.request_log[client_ip] = []
            else:
                # Remove old requests outside the window
                self.request_log[client_ip] = [
                    timestamp for timestamp in self.request_log[client_ip]
                    if now - timestamp < self.rate_limit_window
                ]

            # Check if rate limit exceeded
            if len(self.request_log[client_ip]) >= self.max_requests_per_minute:
                self.logger.warning(f"Rate limit exceeded for IP {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded. Please try again later."}
                )

            # Add current request to log
            self.request_log[client_ip].append(now)
            self.requests_processed += 1
            self.clients_connected = len(set(self.request_log.keys()))  # Update connected clients

            # Process the request
            response = await call_next(request)
            return response

        # Set up routes
        self._setup_endpoints(app)

        return app

    def _api_key_auth(self, api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
        """Validate API key for protected endpoints"""
        if not self.enable_auth:
            return True

        if not api_key or api_key != self.api_key:
            self.logger.warning(f"Invalid API key attempt: {api_key}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return True

    def _generate_api_key(self):
        """Generate a secure API key"""
        key = secrets.token_hex(16)  # 32-character hex string
        self.logger.info(f"Generated new API key: {key}")
        return key

    def _setup_endpoints(self, app):
        """Set up API endpoints"""

        # Status endpoint (public)
        @app.get("/status", response_model=StatusResponse)
        async def get_status():
            """Get current trading bot status"""
            try:
                bot_status = self.bot.get_status()
                risk_status = bot_status.get('risk_management', {})

                response = {
                    "state": bot_status.get('state', 'unknown'),
                    "mode": bot_status.get('mode', 'unknown'),
                    "balance": float(risk_status.get('current_balance', 0.0)),
                    "trades_today": int(risk_status.get('trades_today', 0)),
                    "daily_profit": float(risk_status.get('daily_profit', 0.0)),
                    "daily_loss": float(risk_status.get('daily_loss', 0.0)),
                    "positions": bot_status.get('positions', {}),
                    "timestamp": datetime.now(pytz.UTC).isoformat()
                }
                self.logger.debug(f"Status requested: {response}")
                return response
            except Exception as e:
                self.logger.error(f"Error retrieving status: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error retrieving status: {str(e)}"
                )

        # Server health endpoint (public)
        @app.get("/health")
        async def health_check():
            """Simple health check endpoint"""
            return {"status": "ok", "timestamp": datetime.now(pytz.UTC).isoformat()}

        # Server status endpoint (authenticated)
        @app.get("/server-status", response_model=ServerStatus)
        async def server_status(authorized: bool = Depends(self._api_key_auth)):
            """Get server status information"""
            try:
                if not self.server_start_time:
                    uptime = "Not started"
                else:
                    uptime_seconds = (datetime.now(pytz.UTC) - self.server_start_time).total_seconds()
                    hours, remainder = divmod(uptime_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    uptime = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

                response = {
                    "uptime": uptime,
                    "api_version": self.api_version,
                    "clients_connected": self.clients_connected,
                    "requests_processed": self.requests_processed,
                    "start_time": self.server_start_time.isoformat() if self.server_start_time else None
                }
                self.logger.debug(f"Server status requested: {response}")
                return response
            except Exception as e:
                self.logger.error(f"Error retrieving server status: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error retrieving server status: {str(e)}"
                )

        # Stop bot endpoint (authenticated)
        @app.post("/stop")
        async def stop_bot(authorized: bool = Depends(self._api_key_auth)):
            """Stop the trading bot"""
            try:
                success = self.bot.stop()
                if success:
                    self.logger.info("API request: Bot stopped")
                    return {"message": "Bot stopped", "timestamp": datetime.now(pytz.UTC).isoformat()}
                else:
                    self.logger.error("Failed to stop bot via API")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to stop bot"
                    )
            except Exception as e:
                self.logger.error(f"Error stopping bot: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error stopping bot: {str(e)}"
                )

        # Start bot endpoint (authenticated)
        @app.post("/start")
        async def start_bot(authorized: bool = Depends(self._api_key_auth)):
            """Start the trading bot"""
            try:
                success = self.bot.start()
                if success:
                    self.logger.info("API request: Bot started")
                    return {"message": "Bot started", "timestamp": datetime.now(pytz.UTC).isoformat()}
                else:
                    self.logger.error("Failed to start bot via API")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to start bot"
                    )
            except Exception as e:
                self.logger.error(f"Error starting bot: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error starting bot: {str(e)}"
                )

        # Modify risk level endpoint (authenticated)
        @app.post("/risk")
        async def modify_risk(request: ModifyRiskRequest, authorized: bool = Depends(self._api_key_auth)):
            """Modify risk level for trading"""
            try:
                if not hasattr(self.bot, 'risk_management') or not self.bot.risk_management:
                    raise HTTPException(
                        status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="Risk management module not available"
                    )

                success = self.bot.risk_management.set_risk_level(
                    request.risk_level,
                    request.duration_minutes
                )

                if success:
                    self.logger.info(
                        f"Risk level modified to {request.risk_level} for {request.duration_minutes} minutes via API")
                    return {
                        "message": f"Risk level set to {request.risk_level} for {request.duration_minutes} minutes",
                        "timestamp": datetime.now(pytz.UTC).isoformat()
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to set risk level"
                    )
            except HTTPException as e:
                raise e
            except Exception as e:
                self.logger.error(f"Error modifying risk level: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error modifying risk level: {str(e)}"
                )

        # Close position endpoint (authenticated)
        @app.post("/close-position")
        async def close_position(request: ClosePositionRequest, authorized: bool = Depends(self._api_key_auth)):
            """Close a specific trading position"""
            try:
                if not hasattr(self.bot, 'trade_execution') or not self.bot.trade_execution:
                    raise HTTPException(
                        status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="Trade execution module not available"
                    )

                result = self.bot.trade_execution.close_position(request.ticket)
                if result:
                    self.logger.info(f"Position {request.ticket} closed via API")
                    return {
                        "message": f"Position {request.ticket} closed",
                        "result": result,
                        "timestamp": datetime.now(pytz.UTC).isoformat()
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to close position {request.ticket}"
                    )
            except HTTPException as e:
                raise e
            except Exception as e:
                self.logger.error(f"Error closing position {request.ticket}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error closing position: {str(e)}"
                )

        # Modify position endpoint (authenticated)
        @app.post("/modify-position")
        async def modify_position(request: ModifyPositionRequest, authorized: bool = Depends(self._api_key_auth)):
            """Modify stop loss or take profit for a position"""
            try:
                if not hasattr(self.bot, 'trade_execution') or not self.bot.trade_execution:
                    raise HTTPException(
                        status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="Trade execution module not available"
                    )

                if request.stop_loss is None and request.take_profit is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="At least one of stop_loss or take_profit must be provided"
                    )

                result = self.bot.trade_execution.modify_position(
                    request.ticket,
                    stop_loss=request.stop_loss,
                    take_profit=request.take_profit
                )

                if result:
                    self.logger.info(f"Position {request.ticket} modified via API: SL={request.stop_loss}, TP={request.take_profit}")
                    return {
                        "message": f"Position {request.ticket} modified",
                        "stop_loss": request.stop_loss,
                        "take_profit": request.take_profit,
                        "timestamp": datetime.now(pytz.UTC).isoformat()
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to modify position {request.ticket}"
                    )
            except HTTPException as e:
                raise e
            except Exception as e:
                self.logger.error(f"Error modifying position {request.ticket}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error modifying position: {str(e)}"
                )

        # Reset daily metrics endpoint (authenticated)
        @app.post("/reset-daily")
        async def reset_daily_metrics(authorized: bool = Depends(self._api_key_auth)):
            """Reset daily trading metrics"""
            try:
                if not hasattr(self.bot, 'risk_management') or not self.bot.risk_management:
                    raise HTTPException(
                        status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="Risk management module not available"
                    )

                self.bot.risk_management.reset_daily_metrics(datetime.now(pytz.UTC).date())
                self.logger.info("Daily metrics reset via API")
                return {
                    "message": "Daily metrics reset",
                    "timestamp": datetime.now(pytz.UTC).isoformat()
                }
            except HTTPException as e:
                raise e
            except Exception as e:
                self.logger.error(f"Error resetting daily metrics: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error resetting daily metrics: {str(e)}"
                )

    def start(self):
        """Start the API server in a separate thread with graceful shutdown handling"""
        if self.running:
            self.logger.warning("API server is already running")
            return

        # Reset shutdown event
        self.shutdown_event.clear()

        # Record start time
        self.server_start_time = datetime.now(pytz.UTC)

        # Create a new loop for the thread
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create a config for uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                loop="asyncio"
            )

            # Create and start server
            server = uvicorn.Server(config)

            # Override server's install_signal_handlers to do nothing (handled in main thread)
            server.install_signal_handlers = lambda: None

            try:
                self.running = True
                self.logger.info(f"API server started at http://{self.host}:{self.port}")
                loop.run_until_complete(server.serve())
            except Exception as e:
                self.logger.error(f"API server error: {str(e)}")
            finally:
                self.running = False
                loop.close()
                self.logger.info("API server stopped")

        # Start the server in a new thread
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def stop(self, timeout=5):
        """
        Stop the API server gracefully.

        Args:
            timeout (int): Maximum time to wait for shutdown in seconds
        """
        if not self.running or not self.server_thread:
            self.logger.warning("API server is not running")
            return

        self.logger.info("Stopping API server...")
        self.running = False
        self.shutdown_event.set()

        # Wait for server thread to terminate
        self.server_thread.join(timeout=timeout)

        if self.server_thread.is_alive():
            self.logger.warning(f"API server did not shut down cleanly within {timeout} seconds")
        else:
            self.logger.info("API server stopped successfully")

        self.server_thread = None

    def is_running(self):
        """Check if API server is running"""
        return self.running and self.server_thread and self.server_thread.is_alive()

    def get_api_key(self):
        """Get the current API key"""
        return self.api_key

    def regenerate_api_key(self):
        """Generate a new API key"""
        self.api_key = self._generate_api_key()
        self.logger.info("API key regenerated")
        return self.api_key


# For testing purposes (if run directly)
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create a mock bot for testing
    class MockBot:
        def __init__(self):
            self.state = "running"
            self.mode = "test"
            self.current_positions = {
                "EURUSD": {"action": "buy", "entry_price": 1.0900, "volume": 0.1, "ticket": "12345"},
                "GBPJPY": None
            }
            self.risk_management = type('Risk', (), {
                "current_balance": 100000.0,
                "trades_today": 2,
                "daily_profit": 500.0,
                "daily_loss": 200.0,
                "set_risk_level": lambda level, duration: True,
                "reset_daily_metrics": lambda date: None,
                "get_status": lambda: {
                    "current_balance": 100000.0,
                    "trades_today": 2,
                    "daily_profit": 500.0,
                    "daily_loss": 200.0
                }
            })()
            self.trade_execution = type('TradeExecution', (), {
                "close_position": lambda ticket: {"ticket": ticket, "closed": True, "profit": 100.0},
                "modify_position": lambda ticket, stop_loss=None, take_profit=None: True
            })()

        def stop(self):
            self.state = "stopped"
            return True

        def start(self):
            self.state = "running"
            return True

        def get_status(self):
            return {
                "state": self.state,
                "mode": self.mode,
                "positions": self.current_positions,
                "risk_management": self.risk_management.get_status()
            }

    # Create and start API server
    bot = MockBot()
    api_config = {
        "host": "127.0.0.1",
        "port": 8080,
        "enable_auth": True,
        "max_requests_per_minute": 60
    }

    api_server = APIServer(api_config, bot)

    # Print API key for testing
    print(f"API Key: {api_server.get_api_key()}")
    print(f"API server starting at http://{api_config['host']}:{api_config['port']}")
    print("Available endpoints:")
    print("  GET  /status           - Get bot status (public)")
    print("  GET  /health           - Simple health check (public)")
    print("  GET  /server-status    - Get server status (authenticated)")
    print("  POST /stop             - Stop the bot (authenticated)")
    print("  POST /start            - Start the bot (authenticated)")
    print("  POST /risk             - Modify risk level (authenticated)")
    print("  POST /close-position   - Close a position (authenticated)")
    print("  POST /modify-position  - Modify a position (authenticated)")
    print("  POST /reset-daily      - Reset daily metrics (authenticated)")

    # Start server
    api_server.start()

    # Handle Ctrl+C for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down API server...")
        api_server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Keep main thread alive
    try:
        while api_server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        api_server.stop()