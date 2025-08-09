import fastf1 as f1

# Pick the first completed 2025 race (replace round number if needed)
session = f1.get_session(2025, 1, 'R')  # 'R' = Race
session.load(telemetry=False, weather=False, messages=False)

print("Available columns in session.results:")
print(session.results.columns.tolist())

print("\nFirst few rows:")
print(session.results.head())
