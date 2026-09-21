import os
import pandas as pd
from openai import OpenAI

CSV_FILE = "weather_data.csv"

def load_weather_data():
    """Load weather data from CSV or create file if missing."""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if not df.empty and "Date" in df.columns:
            df["Date_Parsed"] = pd.to_datetime(df["Date"], errors='coerce')
        return df
    df = pd.DataFrame(columns=["Date", "Temperature", "Condition", "Humidity", "Wind_Speed_kmh", "Wind_Speed_kmmin"])
    df.to_csv(CSV_FILE, index=False)
    return df

def save_observation(date_str, temp, condition, humidity, wind_kmh):
    """Save a new weather observation to CSV."""
    df = load_weather_data().drop(columns=["Date_Parsed"], errors="ignore")
    new_data = {
        "Date": date_str, "Temperature": float(temp), "Condition": str(condition),
        "Humidity": float(humidity), "Wind_Speed_kmh": float(wind_kmh),
        "Wind_Speed_kmmin": round(float(wind_kmh) / 60, 4)
    }
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return True

def get_weather_statistics():
    """Calculate basic temperature stats and common condition."""
    df = load_weather_data()
    if df.empty:
        return None
    return {
        "avg_temp": round(df["Temperature"].mean(), 2),
        "min_temp": df["Temperature"].min(),
        "max_temp": df["Temperature"].max(),
        "common_condition": df["Condition"].mode()[0] if not df["Condition"].empty else "N/A"
    }

def search_by_date(search_date):
    """Search weather entries by date string."""
    df = load_weather_data()
    return pd.DataFrame() if df.empty else df[df["Date"] == search_date]

# --- Stretch Goals ---

def filter_by_month_or_season(month=None, season=None):
    """Filter weather logs by month integer or season name."""
    df = load_weather_data()
    if df.empty or "Date_Parsed" not in df.columns:
        return pd.DataFrame()
    if month:
        return df[df["Date_Parsed"].dt.month == month]
    seasons = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Autumn": [9, 10, 11]}
    return df[df["Date_Parsed"].dt.month.isin(seasons.get(season, []))]

def predict_tomorrow_weather():
    """Simple prediction based on historical trends."""
    df = load_weather_data()
    if df.empty:
        return "No data available to make a prediction."
    latest, avg = df["Temperature"].iloc[-1], df["Temperature"].mean()
    return f"Predicted Temp: ~{round(latest - 1 if latest > avg else latest + 1, 1)}°C"

def compare_yearly_data():
    """Aggregate temperature statistics grouped by year."""
    df = load_weather_data()
    if df.empty or "Date_Parsed" not in df.columns:
        return None
    df["Year"] = df["Date_Parsed"].dt.year
    res = df.groupby("Year")["Temperature"].agg(["mean", "min", "max"]).reset_index()
    res.columns = ["Year", "Avg Temp (°C)", "Min Temp (°C)", "Max Temp (°C)"]
    return res

def get_records():
    """Find highest and lowest temperature records."""
    df = load_weather_data()
    if df.empty:
        return None
    max_row, min_row = df.loc[df["Temperature"].idxmax()], df.loc[df["Temperature"].idxmin()]
    return {
        "highest_temp": max_row["Temperature"], "highest_date": max_row["Date"],
        "lowest_temp": min_row["Temperature"], "lowest_date": min_row["Date"]
    }


def generate_ai_summary(api_key):
    """Generate LLM summary, outfit tips, and activity suggestions via OpenRouter."""
    df = load_weather_data()
    if df.empty:
        return "No data available for AI analysis."
    
    recent_data = df.tail(7)[["Date", "Temperature", "Condition", "Humidity", "Wind_Speed_kmh"]].to_dict(orient="records")
    prompt = f"Analyze the past 7 days weather data: {recent_data}. Provide a short summary, outfit suggestions, and outdoor activity recommendations."
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        completion = client.chat.completions.create(
            model="cohere/north-mini-code:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI Meteorologist. Provide helpful, concise weather summaries and recommendations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error connecting to OpenRouter API: {e}"