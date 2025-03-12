import os
import pandas as pd

# Define the data directory where your files are located
data_dir = r"C:\Users\matsw\PycharmProjects\Firstdraft\data"

# Original file name for the 1m file that needs to be edited
input_filename = "nq-1m_bk.csv"
input_path = os.path.join(data_dir, input_filename)

# Read the original 1m file (semicolon-separated)
df = pd.read_csv(input_path, sep=";", header=None,
                 names=["date", "time", "open", "high", "low", "close", "volume"])

# Combine 'date' and 'time' into a single datetime column.
# Adjust the format string if needed (here we assume dd/mm/YYYY format).
df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'], format="%d/%m/%Y %H:%M")

# Drop the original 'date' and 'time' columns.
df.drop(columns=["date", "time"], inplace=True)

# Filter the data to only include rows between the start and end periods of your 5m file.
start_period = pd.to_datetime("2009-01-20 18:00:00")
end_period = pd.to_datetime("2025-02-06 17:55:00")
df = df[(df['datetime'] >= start_period) & (df['datetime'] <= end_period)]

# Reorder columns to match the standard format: datetime, Open, High, Low, Close, Volume.
df = df[["datetime", "open", "high", "low", "close", "volume"]]

# Convert datetime to ISO format (YYYY-MM-DD HH:MM:SS)
df['datetime'] = df['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")

# Save the converted data as a new CSV file in the same folder.
output_filename = "backtrader_1m_converted.csv"
output_path = os.path.join(data_dir, output_filename)
df.to_csv(output_path, index=False)

print(f"Conversion complete. Converted file saved as {output_path}")
