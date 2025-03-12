import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class Notification:
    def __init__(self, config):
        """
        Initialize the Notification module for The 5%ers MT5 trading with EURUSD and GBPJPY.

        Args:
            config (dict): Configuration dictionary with notification settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract email settings from config
        self.email_enabled = config.get('notification', {}).get('email_enabled', False)
        self.email = config.get('notification', {}).get('email', 'matswm86@yahoo.no')
        self.smtp_server = config.get('notification', {}).get('smtp_server', 'smtp.yahoo.com')
        self.smtp_port = config.get('notification', {}).get('smtp_port', 587)
        self.smtp_user = config.get('notification', {}).get('smtp_user', 'matswm86@yahoo.no')
        self.smtp_password = config.get('notification', {}).get('smtp_password', 'your_password')  # Placeholder

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

        # Send email if enabled and fully configured
        if self.email_enabled and self.email and self.smtp_server and self.smtp_port and self.smtp_user and self.smtp_password:
            try:
                self._send_email(message, details)
            except Exception as e:
                self.logger.error(f"Failed to send email notification: {str(e)}")
        elif self.email_enabled:
            self.logger.warning("Email notification enabled but incomplete configuration provided")

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
        msg['Subject'] = f"Trading Bot Notification - The 5%ers MT5: {message}"

        body = message
        if details:
            body += "\n\nDetails:\n"
            for key, value in details.items():
                body += f"{key}: {value}\n"
        msg.attach(MIMEText(body, 'plain'))

        # Connect to SMTP server and send email
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Enable TLS
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
                self.logger.info("Email notification sent successfully")
        except smtplib.SMTPException as e:
            self.logger.error(f"SMTP error sending email: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error sending email: {str(e)}")
            raise


# Example usage (for testing purposes, commented out)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "notification": {
            "email_enabled": True,
            "email": "matswm86@yahoo.no",
            "smtp_server": "smtp.yahoo.com",
            "smtp_port": 587,
            "smtp_user": "matswm86@yahoo.no",
            "smtp_password": "your_password"  # Replace with actual SMTP password
        }
    }
    notifier = Notification(config)
    notifier.send_notification("Trade Executed", {"symbol": "EURUSD", "profit": 100.50})