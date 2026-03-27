import time
import os
from dwell_time_engine import calculate_expected_dwell_time, get_ground_data

def run_terminal_dashboard(db_path, area_size):
    print("Starting Live Dwell Time Monitor... (Press Ctrl+C to stop)")
    time.sleep(2)

    while True:
        # Clear the terminal screen so it looks like a static dashboard
        # 'cls' is for Windows, 'clear' is for Mac/Linux
        os.system('cls' if os.name == 'nt' else 'clear') 
        
        try:
            # 1. Fetch raw data just to show on the screen
            current_crowd, projected_crowd, _ = get_ground_data(db_path)
            
            # 2. Run your algorithm to get the LIVE Dwell Time
            current_dwell_time = calculate_expected_dwell_time(area_size, db_path, use_projected=False)
            projected_dwell_time = calculate_expected_dwell_time(area_size, db_path, use_projected=True)
            
            # 3. Print the Dashboard
            print("=========================================")
            print("         LIVE VENUE STATUS               ")
            print("=========================================")
            print(f" Current Crowd      : {current_crowd} people")
            print(f" Projected Crowd    : {projected_crowd} people")
            print("-----------------------------------------")
            print(f" Current Dwell Time : {current_dwell_time} minutes")
            print(f" Projected Dwell Time: {projected_dwell_time} minutes")
            print("=========================================")
            print("Updating every 2 seconds...")
            
        except Exception as e:
            print(f"Error reading database: {e}. Waiting for setup...")
            
        # Wait 2 seconds before calculating again
        time.sleep(2) 

if __name__ == "__main__":
    DB_FILE = "test-training/ground_state.db"
    TOTAL_AREA = 5000.0  # Make sure this matches your simulation
    
    run_terminal_dashboard(DB_FILE, TOTAL_AREA)