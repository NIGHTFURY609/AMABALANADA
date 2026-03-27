import simpy
import sqlite3
import random

# --- Global Simulation Trackers ---
current_crowd_size = 0
total_visitors_entered = 0

# --- 1. Database Setup ---
def setup_database(db_path="ground_state.db"):
    """Creates tables and inserts initial data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("CREATE TABLE IF NOT EXISTS attractions (id INTEGER PRIMARY KEY, name TEXT, base_duration_mins REAL, pull_factor INTEGER)")
    cursor.execute("CREATE TABLE IF NOT EXISTS venue_state (id INTEGER PRIMARY KEY CHECK (id = 1), last_updated DATETIME DEFAULT CURRENT_TIMESTAMP, current_crowd INTEGER NOT NULL DEFAULT 0, projected_crowd_size INTEGER NOT NULL DEFAULT 0)")
    
    # Clear old data
    cursor.execute("DELETE FROM attractions")
    cursor.execute("DELETE FROM venue_state")
    
    # Insert Sample Attractions
    attractions = [
        ('Main Stage', 45.0, 10),
        ('Food Court', 20.0, 8),
        ('Merch Tent', 10.0, 4),
        ('Restrooms', 5.0, 2)
    ]
    cursor.executemany("INSERT INTO attractions (name, base_duration_mins, pull_factor) VALUES (?, ?, ?)", attractions)
    
    # Initialize Venue State (Starts at 0)
    cursor.execute("INSERT INTO venue_state (id, current_crowd, projected_crowd_size) VALUES (1, 0, 0)")
    
    conn.commit()
    conn.close()

# --- 2. The SimPy Processes ---

def visitor(env, name, db_path):
    """Models a single person's journey through the ground."""
    global current_crowd_size
    
    # 1. ENTER
    current_crowd_size += 1
    
    # Connect to DB to see what attractions are available
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, base_duration_mins, pull_factor FROM attractions")
    attractions = cursor.fetchall()
    conn.close()

    # Determine how many things they will visit (1 to 3 things)
    num_visits = random.randint(1, 3)

    for _ in range(num_visits):
        # 2. TRANSIT (takes 2 to 5 minutes to walk somewhere)
        walking_time = random.uniform(2.0, 5.0)
        yield env.timeout(walking_time) 
        
        # 3. ATTRACTION
        # Pick an attraction weighted by its pull factor
        names = [attr[0] for attr in attractions]
        durations = [attr[1] for attr in attractions]
        pulls = [attr[2] for attr in attractions]
        
        chosen_index = random.choices(range(len(attractions)), weights=pulls, k=1)[0]
        visit_duration = durations[chosen_index]
        
        # Add a little randomness to the duration (+/- 20%)
        actual_duration = visit_duration * random.uniform(0.8, 1.2)
        yield env.timeout(actual_duration)

    # 4. EXIT (transit to exit)
    exit_walk_time = random.uniform(2.0, 5.0)
    yield env.timeout(exit_walk_time)
    
    current_crowd_size -= 1


def crowd_generator(env, db_path, arrival_rate_per_min):
    """Continuously generates new visitors based on an arrival rate."""
    global total_visitors_entered
    while True:
        # Exponential distribution is standard for modeling random arrivals
        time_between_arrivals = random.expovariate(arrival_rate_per_min)
        yield env.timeout(time_between_arrivals)
        
        total_visitors_entered += 1
        env.process(visitor(env, f"Visitor-{total_visitors_entered}", db_path))


def vision_camera_simulator(env, db_path, update_interval_mins):
    """Mimics your ML script by periodically updating the SQLite database."""
    while True:
        # Wait for the update interval (e.g., every 1 minute)
        yield env.timeout(update_interval_mins)
        
        # Update the database with the current global state
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE venue_state SET current_crowd = ?, last_updated = CURRENT_TIMESTAMP WHERE id = 1", 
            (current_crowd_size,)
        )
        conn.commit()
        conn.close()
        
        # Print to console so you can watch the simulation run
        print(f"[Time {env.now:.2f} mins] Vision Camera Update: {current_crowd_size} people currently in ground.")

# --- 3. Run the Simulation ---
if __name__ == "__main__":
    DB_FILE = "ground_state.db"
    
    print("Setting up database...")
    setup_database(DB_FILE)
    
    print("Starting Simulation...")
    # Initialize the SimPy environment
    env = simpy.Environment()
    
    # 1. Start the camera updater (updates DB every 5 virtual minutes)
    env.process(vision_camera_simulator(env, DB_FILE, update_interval_mins=5.0))
    
    # 2. Start people arriving (e.g., 5 people entering per minute)
    env.process(crowd_generator(env, DB_FILE, arrival_rate_per_min=5.0))
    
    # Run the simulation for 120 virtual minutes (2 hours)
    env.run(until=120)
    
    print(f"\nSimulation Complete! Total visitors processed: {total_visitors_entered}")
    print("Your SQLite database is now populated and ready for you to test your dwell time algorithm against the final state.")