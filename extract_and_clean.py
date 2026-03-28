import sqlite3
import pandas as pd

print("1. Connecting to the database...")
# Connect to your SQLite database
conn = sqlite3.connect("ground_state.db")

# Read the entire crowd_log table directly into a Pandas DataFrame
df = pd.read_sql_query("SELECT * FROM crowd_log", conn)
conn.close()

print(f" Successfully loaded {len(df)} rows from the database.")

# Optional: Save the raw data just so you have a backup to look at
df.to_csv("raw_simulated_data.csv", index=False)


print("\n2. Cleaning and Engineering Features...")
# --- DATA CLEANING & FEATURE ENGINEERING ---

# 1. Convert 'sim_minute' (0 to 720) into an 'Hour' column (0 to 12)
# This mimics real-world clock hours which AI models prefer
df['Hour'] = (df['sim_minute'] // 60).astype(int)

# 2. Ensure 'is_weekend' is a strict 1 or 0 (Machine Learning hates True/False text)
df['is_weekend'] = df['is_weekend'].astype(int)

# 3. Drop columns that the AI doesn't need to learn from
# We drop 'id' because it's just a row counter, and 'sim_minute' because we have 'Hour' now
df = df.drop(columns=['id', 'sim_minute'])

# 4. Rearrange columns so 'current_crowd' (our Target) is at the end for easy viewing
columns_order = ['day_number', 'Hour', 'is_weekend', 'current_crowd']
df = df[columns_order]

print("Data cleaned and formatted for Machine Learning.")
print("\nHere is a preview of your clean data:")
print(df.head())


print("\n3. Saving Final AI-Ready Dataset...")
# --- SAVE THE CLEAN DATA ---
clean_filename = "ai_ready_crowd_data.csv"
df.to_csv(clean_filename, index=False)

print(f" Success! Your clean data is saved as '{clean_filename}'")