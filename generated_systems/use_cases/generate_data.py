import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

START_DATE = datetime(2025, 1, 1)
END_DATE   = datetime(2025, 1, 31)

GENRES = [
    "Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance",
    "Documentary", "Animation", "Thriller", "Crime", "Fantasy", "Kids"
]

CONTENT_TYPES = ["Movie", "TV Show"]
GENRE_WEIGHTS = [18, 25, 20, 12, 15, 14, 8, 16, 13, 12, 11, 22]
rows = []

current_id = 1

for single_date in pd.date_range(START_DATE, END_DATE):
    day_of_week = single_date.weekday()          # 0=Mon, 6=Sun
    is_weekend = day_of_week >= 5
    
    base_sessions = random.randint(30, 50)
    if is_weekend:
        daily_sessions = int(base_sessions * random.uniform(1.4, 1.8))
    else:
        daily_sessions = int(base_sessions * random.uniform(0.9, 1.2))
    
    for _ in range(daily_sessions):
        # Time of day – strong peaks at 20:00–23:00, smaller at lunch & late night
        hour = random.choices(
            population=range(24),
            weights=[1,0.5,0.3,0.2,0.2,0.4,1,2,4,4,3,3,3,3,4,5,7,10,14,20,22,18,12,6],
            k=1
        )[0]
        
        session_start = single_date + timedelta(
            hours=hour,
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Genre & type selection
        genre = random.choices(GENRES, weights=GENRE_WEIGHTS, k=1)[0]
        content_type = random.choices(CONTENT_TYPES, weights=[40, 60], k=1)[0]  # more TV shows
        
        # Duration watched in minutes
        if content_type == "Movie":
            full_length = random.randint(80, 160)
        else:  # TV Show episode
            full_length = random.randint(20, 70)
        
        # How much did the user actually watch? (many abandon early)
        watch_percentage = random.choices(
            [1.0, 0.9, 0.75, 0.5, 0.25, 0.1],
            weights=[40, 20, 15, 12, 8, 5], k=1
        )[0]
        
        minutes_watched = int(full_length * watch_percentage)
        if minutes_watched == 0:
            minutes_watched = random.randint(1, 5)  # at least a few seconds
            
        rows.append({
            "session_id": current_id,
            "date": single_date.date(),
            "datetime": session_start,
            "day_of_week": single_date.strftime("%A"),
            "hour": hour,
            "genre": genre,
            "content_type": content_type,
            "minutes_watched": minutes_watched
        })
        current_id += 1

df = pd.DataFrame(rows)
df = df.sort_values("datetime").reset_index(drop=True)
df.to_csv("data/input/streaming_sessions.csv", index=False)
print(f"Generated {len(df):,} streaming sessions")