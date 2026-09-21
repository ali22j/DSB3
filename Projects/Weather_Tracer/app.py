import streamlit as st
import myutils

st.title("Weather Tracker Application")

menu = st.sidebar.selectbox(
    "Choose an Option",
    [
        "1. Record new weather observation",
        "2. View weather statistics",
        "3. Search observations by date",
        "4. View all observations",
        "5. Temperature Trends (Graph)",
        "6. Filter by Month or Season",
        "7. Predict Tomorrow's Weather",
        "8. Compare Yearly Data",
        "9. View Record-Breaking Records",
        "10. AI Meteorologist Summary (LLM API)"
    ]
)

if menu == "1. Record new weather observation":
    st.subheader("Record Weather Data")
    date_str = st.date_input("Select Date").strftime("%m/%d/%Y")
    temp = st.number_input("Temperature (°C)", value=25.0, step=0.5)
    condition = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"])
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind_kmh = st.number_input("Wind Speed (km/h)", value=10.0, step=1.0)
    st.info(f"Calculated Wind Speed: **{round(wind_kmh / 60, 4)} km/min**")
    
    if st.button("Save Observation"):
        myutils.save_observation(date_str, temp, condition, humidity, wind_kmh)
        st.success("Observation saved successfully!")

elif menu == "2. View weather statistics":
    st.subheader("Weather Statistics")
    stats = myutils.get_weather_statistics()
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Temp", f"{stats['avg_temp']} °C")
        c2.metric("Minimum Temp", f"{stats['min_temp']} °C")
        c3.metric("Maximum Temp", f"{stats['max_temp']} °C")
        st.info(f"**Most Common Condition:** {stats['common_condition']}")
    else:
        st.warning("No weather observations found.")

elif menu == "3. Search observations by date":
    st.subheader("Search Observations")
    s_date = st.date_input("Select Date").strftime("%m/%d/%Y")
    if st.button("Search"):
        res = myutils.search_by_date(s_date)
        st.dataframe(res, use_container_width=True) if not res.empty else st.warning("No entries found.")

elif menu == "4. View all observations":
    st.subheader("All Recorded Weather Observations")
    df = myutils.load_weather_data()
    clean_df = df.drop(columns=["Date_Parsed"], errors="ignore")
    if not clean_df.empty:
        st.dataframe(clean_df, use_container_width=True)
    else:
        st.info("No data recorded.")

elif menu == "5. Temperature Trends (Graph)":
    st.subheader("Temperature Trends Over Time")
    df = myutils.load_weather_data()
    if not df.empty:
        st.line_chart(df.set_index("Date")["Temperature"])
    else:
        st.warning("No data available.")

elif menu == "6. Filter by Month or Season":
    st.subheader("Filter Observations")
    f_type = st.radio("Filter By", ["Month", "Season"])
    
    if f_type == "Month":
        selected_month = st.slider("Month", 1, 12, 1)
        res = myutils.filter_by_month_or_season(month=selected_month)
    else:
        selected_season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
        res = myutils.filter_by_month_or_season(season=selected_season)
        
    clean_res = res.drop(columns=["Date_Parsed"], errors="ignore")
    if not clean_res.empty:
        st.dataframe(clean_res, use_container_width=True)
    else:
        st.warning("No matching data.")

elif menu == "7. Predict Tomorrow's Weather":
    st.subheader("Weather Prediction")
    st.success(myutils.predict_tomorrow_weather())

elif menu == "8. Compare Yearly Data":
    st.subheader("Yearly Comparison")
    df = myutils.compare_yearly_data()
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No data available.")

elif menu == "9. View Record-Breaking Records":
    st.subheader("Record-Breaking Temperatures")
    rec = myutils.get_records()
    if rec:
        st.error(f"Highest: {rec['highest_temp']}°C on {rec['highest_date']}")
        st.info(f"Lowest: {rec['lowest_temp']}°C on {rec['lowest_date']}")
    else:
        st.warning("No data available.")

elif menu == "10. AI Meteorologist Summary (LLM API)":
    st.subheader("AI Meteorologist Summary")
    st.write("Generates weather summary and recommendations from your past 7 days of logs.")
    
    if st.button("Generate AI Summary"):
        if "OPENROUTER_API_KEY" in st.secrets:
            api_key = st.secrets["OPENROUTER_API_KEY"]
            with st.spinner("Analyzing data via OpenRouter..."):
                summary = myutils.generate_ai_summary(api_key)
                st.write(summary)
        else:
            st.error("OPENROUTER_API_KEY not found in .streamlit/secrets.toml")

            