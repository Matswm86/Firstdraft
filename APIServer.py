import sys
import clr

# Add the NinjaTrader DLL directory to Python’s path
ninja_bin_path = r"C:\Program Files\NinjaTrader 8\bin"  # Adjust if your path is different
sys.path.append(ninja_bin_path)

# Load the NinjaTrader assembly
clr.AddReference("NinjaTrader.Core")  # This is the DLL name—check below if it doesn’t work

# Import specific classes (example namespace, adjust as needed)
from NinjaTrader.Cbi import Order, OrderType, OrderAction