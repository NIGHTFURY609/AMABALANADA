import sqlite3
import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Attraction:
    id: int
    name: str
    base_duration_mins: float
    pull_factor: int

def get_ground_data(db_path: str) -> Tuple[int, int, List[Attraction]]:
    """Fetches the latest venue state and attractions from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Get the LATEST crowd size from the new crowd_log table
    try:
        # ORDER BY id DESC LIMIT 1 grabs the absolute newest row logged by the camera
        cursor.execute("SELECT current_crowd FROM crowd_log ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        current_crowd = row[0] if row else 0 
    except sqlite3.OperationalError:
        current_crowd = 0 # Fallback if table doesn't exist

    # We set projected to 0 because your React dashboard now uses XGBoost for predictions!
    projected_crowd = 0

    # 2. Get Attractions (This table hasn't changed!)
    try:
        cursor.execute("SELECT id, name, base_duration_mins, pull_factor FROM attractions")
        attractions = [Attraction(*r) for r in cursor.fetchall()]
    except sqlite3.OperationalError:
        attractions = []
        
    conn.close()
    return current_crowd, projected_crowd, attractions

def calculate_expected_dwell_time(total_area_sqm: float, db_path: str, use_projected: bool = False) -> float:
    """Calculates deterministic expected dwell time without randomness."""
    current_crowd, projected_crowd, attractions = get_ground_data(db_path)
    
    if not attractions:
        return 0.0 
        
    active_crowd_size = projected_crowd if use_projected else current_crowd
    
    # Calculate Walking Speed
    v_max = 1.34  
    rho_jam = 4.0 
    
    current_density = active_crowd_size / total_area_sqm
    walking_speed_mps = max(0.2, v_max * (1 - (current_density / rho_jam)))
    walking_speed_mpm = walking_speed_mps * 60 

    total_dwell_mins = 0.0
    avg_distance_m = math.sqrt(total_area_sqm) / 2.0
    
    # Estimate visits
    total_ground_pull = sum(attr.pull_factor for attr in attractions)
    estimated_visits = max(1, int(total_ground_pull / 5)) 
    estimated_visits = min(estimated_visits, len(attractions))

    # --- THE FIX: Calculate the Expected Value of an attraction visit ---
    # Formula: Sum of (Probability of visiting * Duration) for all attractions
    expected_attraction_duration = sum(
        (attr.pull_factor / total_ground_pull) * attr.base_duration_mins 
        for attr in attractions
    )

    # Process the Journey deterministically
    for _ in range(estimated_visits):
        transit_time_mins = avg_distance_m / walking_speed_mpm
        
        total_dwell_mins += transit_time_mins
        total_dwell_mins += expected_attraction_duration # Add the average time
        
    exit_transit_mins = avg_distance_m / walking_speed_mpm
    total_dwell_mins += exit_transit_mins
    
    return round(total_dwell_mins, 2)