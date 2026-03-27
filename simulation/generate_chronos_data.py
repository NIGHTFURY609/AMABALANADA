import simpy
import csv
import random

class VenueSimulation:
    def __init__(self, env):
        self.env = env
        # Initial state at 10:00 AM
        self.crowd_count = 100
        self.hotspot_count = 4
        self.data_log = []
        
        # Start the two independent processes
        self.env.process(self.crowd_dynamics())
        self.env.process(self.data_logger(max_rows=50))

    def crowd_dynamics(self):
        """Simulates the continuous movement of the crowd every minute."""
        while True:
            # 1. Simulate people arriving/leaving
            # Add a slight upward trend for the first few hours, then downward
            if self.env.now < (10 * 60) + 120:  # Before 12:00 PM
                change = random.randint(-5, 12) # Biased towards entering
            else:
                change = random.randint(-15, 5) # Biased towards leaving
                
            self.crowd_count = max(10, self.crowd_count + change)

            # 2. Dynamic Hotspot Management (Your Covariate)
            # Management opens more hotspots when the crowd gets too large
            if self.crowd_count < 120:
                self.hotspot_count = random.randint(3, 4)
            elif self.crowd_count < 180:
                self.hotspot_count = random.randint(5, 6)
            else:
                self.hotspot_count = random.randint(7, 8)
                
            # Wait 1 virtual minute before updating state again
            yield self.env.timeout(1)

    def data_logger(self, max_rows):
        """Takes a snapshot of the venue state every 5 minutes and formats for CSV."""
        rows_generated = 0
        
        while rows_generated < max_rows:
            # Convert SimPy's raw minutes into HH:MM format
            current_minutes = int(self.env.now)
            hours = current_minutes // 60
            minutes = current_minutes % 60
            
            timestamp = f"{hours:02d}:{minutes:02d}"
            time_of_day = hours
            
            # Log the exact format required for Chronos
            self.data_log.append([timestamp, self.crowd_count, time_of_day, self.hotspot_count])
            
            rows_generated += 1
            
            # Wait 5 virtual minutes before logging again
            yield self.env.timeout(5)

# --- Execution Block ---
if __name__ == "__main__":
    # Initialize the SimPy environment
    # We start the clock at 600 minutes (which is 10:00 AM)
    start_time_minutes = 10 * 60 
    env = simpy.Environment(initial_time=start_time_minutes)
    
    # Instantiate the venue simulation
    venue = VenueSimulation(env)
    
    # Run the simulation until 50 rows of 5-minute intervals are collected (250 minutes)
    env.run(until=start_time_minutes + (50 * 5))
    
    # Write the collected data to CSV
    output_filename = "chronos_covariate_data.csv"
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header exactly as requested
        writer.writerow(["timestamp", "crowd_count", "time_of_day", "hotspot_count"])
        # Write the simulated data
        writer.writerows(venue.data_log)

    print(f"Successfully generated 50 simulation rows in '{output_filename}'")
    
    # Preview the first 5 rows in the terminal
    print("\nPreview:")
    print("timestamp,crowd_count,time_of_day,hotspot_count")
    for row in venue.data_log[:5]:
        print(",".join(map(str, row)))