import simpy
import sqlite3
import random

# --- Global Simulation Trackers ---
current_crowd_size = 0
total_visitors_entered = 0

# --- 1. Database Setup ---
def setup_database(db_path="ground_state.db"):
    """Creates tables and sets up a time-series log for multiple days."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create attractions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attractions (
            id INTEGER PRIMARY KEY, 
            name TEXT, 
            base_duration_mins REAL, 
            pull_factor INTEGER
        )
    """)
    
    # NEW: Create a Time-Series log table instead of a single state row
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crowd_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day_number INTEGER,
            sim_minute REAL,
            current_crowd INTEGER,
            is_weekend BOOLEAN
        )
    """)
    
    # Clear old data for a fresh simulation run
    cursor.execute("DELETE FROM attractions")
    cursor.execute("DELETE FROM crowd_log")
    
    # Insert Sample Attractions
    attractions = [
        ('Main Stage', 45.0, 10),
        ('Food Court', 20.0, 8),
        ('Merch Tent', 10.0, 4),
        ('Restrooms', 5.0, 2)
    ]
    cursor.executemany("INSERT INTO attractions (name, base_duration_mins, pull_factor) VALUES (?, ?, ?)", attractions)
    
    conn.commit()
    conn.close()

# --- 2. The SimPy Processes ---

def visitor(env, name, db_path):
    """Models a single person's journey through the ground."""
    global current_crowd_size
    
    # 1. ENTER
    current_crowd_size += 1
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, base_duration_mins, pull_factor FROM attractions")
    attractions = cursor.fetchall()
    conn.close()

    num_visits = random.randint(1, 3)

    for _ in range(num_visits):
        # 2. TRANSIT
        yield env.timeout(random.uniform(2.0, 5.0)) 
        
        # 3. ATTRACTION
        pulls = [attr[2] for attr in attractions]
        durations = [attr[1] for attr in attractions]
        
        chosen_index = random.choices(range(len(attractions)), weights=pulls, k=1)[0]
        actual_duration = durations[chosen_index] * random.uniform(0.8, 1.2)
        yield env.timeout(actual_duration)

    # 4. EXIT
    yield env.timeout(random.uniform(2.0, 5.0))
    current_crowd_size -= 1


def crowd_generator(env, db_path, arrival_rate_per_min):
    """Continuously generates new visitors based on an arrival rate."""
    global total_visitors_entered
    while True:
        time_between_arrivals = random.expovariate(arrival_rate_per_min)
        yield env.timeout(time_between_arrivals)
        
        total_visitors_entered += 1
        env.process(visitor(env, f"Visitor-{total_visitors_entered}", db_path))


def vision_camera_simulator(env, db_path, update_interval_mins, day_number, is_weekend):
    """Logs the current crowd size to the database every few minutes."""
    while True:
        yield env.timeout(update_interval_mins)
        
        # INSERT a new record instead of overwriting an old one
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO crowd_log (day_number, sim_minute, current_crowd, is_weekend) VALUES (?, ?, ?, ?)", 
            (day_number, round(env.now, 2), current_crowd_size, is_weekend)
        )
        conn.commit()
        conn.close()
        
        # Print a clean log to the terminal
        day_type = "Weekend" if is_weekend else "Weekday"
        print(f"[Day {day_number} - {day_type} | Time {env.now:06.2f}m] Crowd: {current_crowd_size}")

# --- 3. Run the Multi-Day Simulation ---
if __name__ == "__main__":
    DB_FILE = "ground_state.db"
    TOTAL_DAYS_TO_SIMULATE = 7 # Simulating one full week
    MINUTES_PER_DAY = 720      # e.g., Open for 12 hours (12 * 60)
    
    print("Setting up database...")
    setup_database(DB_FILE)
    
    print(f"\nStarting {TOTAL_DAYS_TO_SIMULATE}-Day Simulation...")
    
    for day in range(1, TOTAL_DAYS_TO_SIMULATE + 1):
        # Reset the global crowd size at the start of every new day (gates open)
        current_crowd_size = 0 
        
        # Determine if it's a weekend (Days 6 and 7 of our cycle)
        is_weekend = True if (day % 7 == 6) or (day % 7 == 0) else False
        
        # Weekends get 3x more traffic! (15 people/min vs 5 people/min)
        daily_arrival_rate = 15.0 if is_weekend else 5.0 
        
        print(f"\n--- STARTING DAY {day} (Arrival Rate: {daily_arrival_rate}/min) ---")
        
        # Initialize a fresh SimPy environment for the new day
        env = simpy.Environment()
        
        # 1. Start the camera logger (logs to DB every 15 virtual minutes)
        env.process(vision_camera_simulator(env, DB_FILE, update_interval_mins=15.0, day_number=day, is_weekend=is_weekend))
        
        # 2. Start people arriving
        env.process(crowd_generator(env, DB_FILE, arrival_rate_per_min=daily_arrival_rate))
        
        # Run the simulation for the day's total minutes
        env.run(until=MINUTES_PER_DAY)
        
    print(f"\n Simulation Complete!")
    print(f"Total visitors processed across {TOTAL_DAYS_TO_SIMULATE} days: {total_visitors_entered}")
    print("Your SQLite database 'crowd_log' table is now populated with multi-day time-series data!")