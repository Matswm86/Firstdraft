import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class Notification:
    def __init__(self, config):
        """
        Initialize the Notification module with configuration settings.

        Args:
            config (dict): Configuration dictionary with notification settings.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Extract email settings from config
        self.email_enabled = config.get('email_enabled', False)
        self.email = config.get('email')
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.smtp_user = config.get('smtp_user')
        self.smtp_password = config.get('smtp_password')

    def send_notification(self, message, details=None):
        """
        Send a notification based on configuration settings (e.g., email, log).

        Args:
            message (str): The main notification message.
            details (dict, optional): Additional details to include in the notification.
        """
        # Log the notification regardless of other methods
        details_str = f" - Details: {details}" if details else ""
        self.logger.info(f"Notification: {message}{details_str}")

        # Send email if enabled and configured
        if self.email_enabled and self.email and self.smtp_server:
            try:
                self._send_email(message, details)
            except Exception as e:
                self.logger.error(f"Failed to send email notification: {str(e)}")

    def _send_email(self, message, details=None):
        """
        Send an email notification using SMTP settings from config.

        Args:
            message (str): The main notification message.
            details (dict, optional): Additional details to include in the email body.
        """
        # Construct the email
        msg = MIMEMultipart()
        msg['From'] = self.smtp_user
        msg['To'] = self.email
        msg['Subject'] = f"Trading Bot Notification: {message}"

        body = message
        if details:
            body += "\n\nDetails:\n"
            for key, value in details.items():
                body += f"{key}: {value}\n"
        msg.attach(MIMEText(body, 'plain'))

        # Connect to SMTP server and send email
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()  # Enable TLS
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            self.logger.info("Email notification sent successfully")


# Example usage (for testing purposes, commented out)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "email_enabled": True,
        "email": "matswm86@yahoo.no",
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "smtp_user": "user@example.com",
        "smtp_password": "password"
    }
    notifier = Notification(config)
    notifier.send_notification("Trade Executed", {"symbol": "NQ", "profit": 100.50})