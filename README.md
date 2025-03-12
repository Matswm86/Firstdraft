Trading Bot System
This project is an automated trading bot system designed to trade forex pairs like EURUSD and GBPJPY in live mode using MetaTrader 5 (MT5) with The 5%ers proprietary firm. It also supports backtesting for NASDAQ futures using historical data. The system is modular, with components for data ingestion, signal generation, risk management, trade execution, notifications, and more.

Table of Contents
Features
Installation
Configuration
Usage
Modules
API Documentation
Contributing
License
Disclaimer
Features
Live Trading: Executes trades on MT5 for forex pairs (EURUSD, GBPJPY) with The 5%ers.
Backtesting: Runs simulations on historical NASDAQ futures data using Backtrader.
Modular Design: Separate modules for data handling, signal generation, risk management, and trade execution.
Risk Management: Enforces drawdown limits, position sizing, and daily loss controls.
Notifications: Sends alerts via email for key events (e.g., trade executions, errors).
API & Dashboard: Provides a FastAPI server and Tkinter GUI for monitoring and control.
Machine Learning: Optionally enhances signals using machine learning models.
News Integration: Adjusts risk based on market news sentiment.
Installation
Prerequisites
Python 3.8+
MetaTrader 5 (for live trading)
Git (to clone the repository)
Steps
Clone the repository:
bash

Collapse

Wrap

Copy
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
Install dependencies:
bash

Collapse

Wrap

Copy
pip install -r requirements.txt
Install MT5 Python Library:
bash

Collapse

Wrap

Copy
pip install MetaTrader5
Download NLTK Data (for sentiment analysis):
python

Collapse

Wrap

Copy
import nltk
nltk.download('vader_lexicon')
Configuration
The bot is configured via the config.json file. Below is an overview of key sections:

mt5_settings: MT5 server, login, and password (for live trading).
symbols: Forex pairs for live trading (e.g., ["EURUSD", "GBPJPY"]).
backtesting: Symbols for backtesting (e.g., ["NQ 03-25"]).
risk_management: Risk parameters like max_drawdown, risk_per_trade, etc.
notification: Email settings for alerts (e.g., email, smtp_server).
api_server: Host and port for the FastAPI server.
dashboard: Settings for the Tkinter GUI dashboard.
Example config.json
json

Collapse

Wrap

Copy
{
  "central_trading_bot": {
    "mode": "live"
  },
  "mt5_settings": {
    "server": "The5ers-Server",
    "login": "your_login",
    "password": "your_password"
  },
  "symbols": ["EURUSD", "GBPJPY"],
  "risk_management": {
    "max_drawdown": 0.04,
    "risk_per_trade": 0.01
  },
  "notification": {
    "email_enabled": true,
    "email": "your_email@example.com",
    "smtp_server": "smtp.example.com",
    "smtp_port": 587,
    "smtp_user": "your_email@example.com",
    "smtp_password": "your_password"
  },
  "api_server": {
    "host": "127.0.0.1",
    "port": 8080
  },
  "dashboard": {
    "enable_web_dashboard": false,
    "refresh_interval": 1000
  }
}
Important: Replace placeholders (e.g., your_login, your_password) with actual values.

Usage
Running the Bot
Live Trading Mode:
Set "mode": "live" in config.json.
Run the bot:
bash

Collapse

Wrap

Copy
python main.py
Backtesting Mode:
Set "mode": "backtest" in config.json.
Ensure historical data is in the specified path (e.g., ./data/backtrader_15m.csv).
Run the backtest:
bash

Collapse

Wrap

Copy
python main.py
Monitoring and Control
API Server: Access bot status and control via http://127.0.0.1:8080/status and http://127.0.0.1:8080/stop.
Dashboard: Launch the Tkinter GUI to monitor metrics and positions in real-time.
Modules
The system is composed of the following modules:

main.py: Entry point for starting the bot.
CentralTradingBot.py: Core logic for trading and backtesting.
DataIngestion.py: Handles data loading for live and backtest modes.
SignalGenerator.py: Generates trading signals based on market data.
RiskManagement.py: Enforces risk rules and calculates position sizes.
TradeExecution.py: Executes trades in live or backtest modes.
TradeLogger.py: Logs trade details to a CSV file.
Notification.py: Sends notifications for key events.
APIServer.py: FastAPI server for remote monitoring and control.
Dashboard.py: Tkinter-based GUI for real-time monitoring.
MachineLearning.py: Enhances signals using machine learning models.
MarketNewsAPI.py: Fetches and analyzes market news for risk adjustments.
BacktraderStrategy.py: Defines the backtesting strategy.
Backtesting.py: Runs the backtest using Backtrader.
MT5API.py: Interface for MT5 communication.
API Documentation
The API server provides the following endpoints:

GET /status: Retrieve the current status of the bot, including state, mode, balance, and open positions.
POST /stop: Stop the trading bot.
Example Request:

bash

Collapse

Wrap

Copy
curl http://127.0.0.1:8080/status
Example Response:

json

Collapse

Wrap

Copy
{
  "state": "running",
  "mode": "live",
  "balance": 100000.0,
  "trades_today": 2,
  "daily_profit": 500.0,
  "daily_loss": 200.0,
  "positions": {
    "EURUSD": {
      "action": "buy",
      "entry_price": 1.0900,
      "size": 0.1,
      "order_id": "12345"
    },
    "GBPJPY": null
  }
}
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch for your feature or fix.
Submit a pull request with a clear description of your changes.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Disclaimer
Trading involves significant risk and may result in the loss of your invested capital. This bot is provided for educational purposes only and should not be used for live trading without thorough testing and validation. The developers are not responsible for any financial losses incurred from using this software.
