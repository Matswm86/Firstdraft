import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import time
from datetime import datetime
import pytz


class Notification:
    def __init__(self, config):
        """
        Initialize the Notification module with multiple notification channels.

        Args:
            config (dict): Configuration dictionary with notification settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Get notification config section
        notification_config = config.get('notification', {})

        # Email settings
        self.email_enabled = notification_config.get('email_enabled', False)
        self.email = notification_config.get('email')
        self.smtp_server = notification_config.get('smtp_server')
        self.smtp_port = notification_config.get('smtp_port', 587)
        self.smtp_user = notification_config.get('smtp_user')
        self.smtp_password = notification_config.get('smtp_password')

        # Telegram settings
        self.telegram_enabled = notification_config.get('telegram_enabled', False)
        self.telegram_bot_token = notification_config.get('telegram_bot_token')
        self.telegram_chat_id = notification_config.get('telegram_chat_id')

        # General settings
        self.max_retries = notification_config.get('max_retries', 3)
        self.retry_delay = notification_config.get('retry_delay', 5)  # seconds
        self.notification_levels = notification_config.get('notification_levels', ['error', 'trade', 'warning'])

        # Validate configurations
        if self.email_enabled:
            missing = self._validate_email_config()
            if missing:
                self.logger.warning(f"Email notifications enabled but missing: {', '.join(missing)}")
                self.email_enabled = False  # Disable if config incomplete

        if self.telegram_enabled:
            missing = self._validate_telegram_config()
            if missing:
                self.logger.warning(f"Telegram notifications enabled but missing: {', '.join(missing)}")
                self.telegram_enabled = False  # Disable if config incomplete

        self.logger.info(
            f"Notification initialized: Email={self.email_enabled}, Telegram={self.telegram_enabled}"
        )

    def _validate_email_config(self):
        """Validate email configuration and return missing fields."""
        required = ['email', 'smtp_server', 'smtp_port', 'smtp_user', 'smtp_password']
        missing = [field for field in required if not getattr(self, field) or str(getattr(self, field)).strip() == '']
        return missing

    def _validate_telegram_config(self):
        """Validate Telegram configuration and return missing fields."""
        required = ['telegram_bot_token', 'telegram_chat_id']
        missing = [field for field in required if not getattr(self, field) or str(getattr(self, field)).strip() == '']
        return missing

    def send_notification(self, message, details=None, level='info'):
        """
        Send a notification through configured channels.

        Args:
            message (str): The main notification message.
            details (dict, optional): Additional details to include.
            level (str): Notification level (error, trade, warning, info)
        """
        # Skip if level is not in configured notification levels (always send errors)
        if level not in self.notification_levels and level != 'error':
            return

        # Always log the notification
        details_str = f" - Details: {details}" if details else ""
        self.logger.info(f"Notification ({level}): {message}{details_str}")

        # Send via enabled channels
        if self.email_enabled:
            self._send_email_with_retry(message, details, level)

        if self.telegram_enabled:
            self._send_telegram_with_retry(message, details, level)

    def _send_email_with_retry(self, message, details=None, level='info'):
        """Send email with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                self._send_email(message, details, level)
                self.logger.debug(f"Email sent successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                self.logger.warning(f"Email attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to send email after {self.max_retries} attempts: {str(e)}")

    def _send_telegram_with_retry(self, message, details=None, level='info'):
        """Send Telegram message with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                self._send_telegram(message, details, level)
                self.logger.debug(f"Telegram message sent successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                self.logger.warning(f"Telegram attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to send Telegram message after {self.max_retries} attempts: {str(e)}")

    def _send_email(self, message, details=None, level='info'):
        """Send an email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.email
            msg['Subject'] = f"Trading Bot Notification ({level.upper()}): {message}"

            body = f"Timestamp: {datetime.now(pytz.UTC).isoformat()}\n"
            body += f"Level: {level.upper()}\n"
            body += f"Message: {message}\n"
            if details:
                body += "\nDetails:\n"
                for key, value in details.items():
                    body += f"  {key}: {value}\n"
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            self.logger.info("Email notification sent successfully")
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            raise

    def _send_telegram(self, message, details=None, level='info'):
        """Send a Telegram notification."""
        try:
            text = f"*{level.upper()} Notification*\n"
            text += f"Timestamp: {datetime.now(pytz.UTC).isoformat()}\n"
            text += f"Message: {message}\n"
            if details:
                text += "\n*Details:*\n"
                for key, value in details.items():
                    text += f"  {key}: {value}\n"

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }

            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            self.logger.info("Telegram notification sent successfully")
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {str(e)}")
            raise