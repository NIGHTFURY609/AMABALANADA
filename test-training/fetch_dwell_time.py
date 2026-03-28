import sys
import os
from dwell_time_engine import calculate_expected_dwell_time

# 1. Get the area_size from Node.js (or default to 100 if missing)
try:
    area_size = float(sys.argv[1])
except IndexError:
    area_size = 100.0

# 2. Define the exact DB path you requested
# Since this script runs inside the test-training directory via Node's 'cwd', 
# we just need the filename.
db_path = "ground_state.db"

# 3. Run the function with use_projected=False
try:
    current_dwell_time = calculate_expected_dwell_time(area_size, db_path, use_projected=False)
    # Print it so Node.js can capture it
    print(round(current_dwell_time, 2))
except Exception as e:
    # If the database fails or is missing, print 0 to prevent the server from crashing
    print("0.0")