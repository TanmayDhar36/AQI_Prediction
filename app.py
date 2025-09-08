import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import folium_static
import plotly.graph_objects as go
from folium.plugins import HeatMap
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="AQI Prediction App", layout="wide")

# Title and description
st.title("INDIA's Air Quality Index (AQI) Prediction")
st.write("Enter a date, state, and city to predict AQI and pollutant levels. View an all-India AQI heatmap for the selected date.")

# Indian AQI guidelines
AQI_GUIDELINES = [
    {"AQI Range": "0–50", "Category": "Good", "Color": "#00E400", "Health Implications": "Minimal impact"},
    {"AQI Range": "51–100", "Category": "Satisfactory", "Color": "#90EE90", "Health Implications": "Minor breathing discomfort to sensitive people"},
    {"AQI Range": "101–200", "Category": "Moderate", "Color": "#FFFF00", "Health Implications": "Breathing discomfort to people with lung diseases, children, and older adults"},
    {"AQI Range": "201–300", "Category": "Poor", "Color": "#FFA500", "Health Implications": "Breathing discomfort to most people on prolonged exposure"},
    {"AQI Range": "301–400", "Category": "Very Poor", "Color": "#FF0000", "Health Implications": "Respiratory illness on prolonged exposure; severe effects on people with lung/heart diseases"},
    {"AQI Range": "401–500", "Category": "Severe", "Color": "#8B0000", "Health Implications": "Serious impact on health; affects even healthy people, severe for sensitive groups"}
]

# Load locations from CSV
try:
    locations_df = pd.read_csv('locations.csv')
    # Clean invalid entries
    locations_df = locations_df[
        (locations_df['Latitude'] != 0) &
        (locations_df['Longitude'] != 0) &
        (locations_df['Latitude'] != '#N/A')
    ]
    # Convert Latitude and Longitude to numeric, coercing errors to NaN
    locations_df['Latitude'] = pd.to_numeric(locations_df['Latitude'], errors='coerce')
    locations_df['Longitude'] = pd.to_numeric(locations_df['Longitude'], errors='coerce')
    # Drop rows with NaN coordinates or missing City/State
    locations_df = locations_df.dropna(subset=['Latitude', 'Longitude', 'City', 'State'])
    # Ensure City and State are strings
    locations_df['City'] = locations_df['City'].astype(str)
    locations_df['State'] = locations_df['State'].astype(str)
except FileNotFoundError:
    st.error("File 'locations.csv' not found. Please ensure it is in the same directory.")
    st.stop()

# Create encoders from locations.csv
state_encoder = LabelEncoder()
city_encoder = LabelEncoder()
state_encoder.fit(locations_df['State'])
city_encoder.fit(locations_df['City'])
state_encoding = {state: idx for idx, state in enumerate(state_encoder.classes_)}
city_encoding = {city: idx for idx, city in enumerate(city_encoder.classes_)}

# Debug: Display number of encoded states and cities
st.write(f"Encoded {len(state_encoding)} states and {len(city_encoding)} cities from locations.csv")

# Load the scaler
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("File 'scaler.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Load the saved model
try:
    rf = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    st.error("Model file 'random_forest_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Display AQI guidelines chart
st.subheader("Indian AQI Guidelines")
guideline_df = pd.DataFrame(AQI_GUIDELINES)
st.dataframe(
    guideline_df[["AQI Range", "Category", "Health Implications"]].style.set_table_styles(
        [
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ]
    ),
    use_container_width=True,
    column_config={
        "Health Implications": st.column_config.TextColumn(width="large")
    }
)

# Input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        date_input = st.date_input("Select Date", value=datetime.today())
    with col2:
        available_states = sorted(list(state_encoding.keys()))
        if not available_states:
            st.error("No valid states found in locations.csv.")
            st.stop()
        state_input = st.selectbox("Select State", available_states)
    with col3:
        available_cities = sorted([city for city in locations_df[locations_df['State'] == state_input]['City'].tolist() if city in city_encoding])
        if not available_cities:
            st.error("No valid cities found for the selected state in locations.csv.")
            st.stop()
        city_input = st.selectbox("Select City", available_cities)
    submit_button = st.form_submit_button("Predict")

# Process prediction when form is submitted
if submit_button:
    st.write("Form submitted successfully!")  # Debug statement
    try:
        # Convert date to components
        date = pd.to_datetime(date_input)
        Year, Month, Day = date.year, date.month, date.day

        # Prepare input data for selected city
        input_data = pd.DataFrame({
            'Year': [Year],
            'Month': [Month],
            'Day': [Day],
            'State_encoded': [state_encoding[str(state_input)]],
            'City_encoded': [city_encoding[str(city_input)]]
        })

        # Scale input data
        X_scaled = scaler.transform(input_data)

        # Predict for selected city
        prediction = rf.predict(X_scaled)
        targets = ['Ozone', 'CO', 'SO2', 'NO2', 'PM10', 'PM2.5', 'AQI']
        aqi_pred = prediction[0, 6]  # AQI is index 6

        # Get AQI category and color
        aqi_category = "Unknown"
        aqi_color = "#808080"
        for guideline in AQI_GUIDELINES:
            min_val, max_val = map(int, guideline["AQI Range"].split("–"))
            if min_val <= aqi_pred <= max_val:
                aqi_category = guideline["Category"]
                aqi_color = guideline["Color"]
                break

        # Update guideline chart to highlight predicted AQI category
        st.subheader("Indian AQI Guidelines (Updated)")
        guideline_df["Highlight"] = guideline_df["Category"].apply(lambda x: "✔" if x == aqi_category else "")
        st.dataframe(
            guideline_df[["AQI Range", "Category", "Highlight", "Health Implications"]].style.apply(
                lambda x: ['background-color: {}'.format(aqi_color) if x["Category"] == aqi_category else '' for _ in x],
                axis=1
            ).set_table_styles(
                [
                    {'selector': 'th', 'props': [('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ]
            ),
            use_container_width=True,
            column_config={
                "Highlight": st.column_config.TextColumn(label=""),
                "Health Implications": st.column_config.TextColumn(width="large")
            }
        )

        # Display predictions
        st.subheader(f"Predictions for {date_input} in {city_input}, {state_input}")
        fig = go.Figure(data=[
            go.Bar(
                x=targets,
                y=prediction[0],
                marker_color=[aqi_color if target == 'AQI' else '#1f77b4' for target in targets],
                text=[f"{value:.2f}" for value in prediction[0]],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Predicted Pollutant Levels and AQI",
            xaxis_title="Pollutants",
            yaxis_title="Concentration",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display predicted AQI with larger font
        st.markdown(
            f"""
            <style>
            .big-font {{
                font-size: 24px !important;
                font-weight: bold;
                color: {aqi_color};
            }}
            </style>
            <div class="big-font">Predicted AQI: {aqi_pred:.2f} ({aqi_category})</div>
            """,
            unsafe_allow_html=True
        )

        # Create and display maps
        st.subheader("Location and All-India AQI Heatmap")
        col1, col2 = st.columns(2)

        # City-specific map
        with col1:
            st.write("Selected City Map")
            city_data = locations_df[locations_df['City'] == city_input]
            lat, lon = city_data['Latitude'].iloc[0], city_data['Longitude'].iloc[0]
            m_city = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB Positron")
            folium.Marker(
                [lat, lon],
                popup=f"{city_input}, {state_input}<br>AQI: {aqi_pred:.2f} ({aqi_category})",
                tooltip=city_input,
                icon=folium.Icon(color='blue' if aqi_pred <= 200 else 'red')
            ).add_to(m_city)
            # Add AQI value label
            folium.Marker(
                [lat + 0.01, lon + 0.01],  # Slightly offset for visibility
                icon=folium.DivIcon(html=f'<div style="font-size: 12px; font-weight: bold; color: black;">{aqi_pred:.2f}</div>')
            ).add_to(m_city)
            aqi_intensity = min(aqi_pred / 500, 1.0)  # Normalize AQI to [0,1]
            HeatMap(
                [[lat, lon, aqi_intensity]],
                gradient={0.2: 'green', 0.4: 'yellow', 0.6: 'orange', 1.0: 'red'},
                radius=20,
                blur=15
            ).add_to(m_city)
            folium_static(m_city, width=600, height=400)

        # All-India heatmap
        with col2:
            st.write("All-India AQI Heatmap")
            # Prepare input data for all cities
            all_cities_data = []
            for _, row in locations_df.iterrows():
                city = str(row['City'])
                state = str(row['State'])
                all_cities_data.append({
                    'Year': Year,
                    'Month': Month,
                    'Day': Day,
                    'State_encoded': state_encoding[state],
                    'City_encoded': city_encoding[city],
                    'Latitude': row['Latitude'],
                    'Longitude': row['Longitude'],
                    'City': city,
                    'State': state
                })
            all_cities_df = pd.DataFrame(all_cities_data)
            
            # Scale and predict for all cities
            X_all_scaled = scaler.transform(all_cities_df[['Year', 'Month', 'Day', 'State_encoded', 'City_encoded']])
            all_predictions = rf.predict(X_all_scaled)
            all_cities_df['AQI'] = all_predictions[:, 6]  # AQI is index 6

            # Create all-India map
            m_india = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB Positron")
            heat_data = [[row['Latitude'], row['Longitude'], min(row['AQI'] / 500, 1.0)] for _, row in all_cities_df.iterrows()]
            HeatMap(
                heat_data,
                gradient={0.2: 'green', 0.4: 'yellow', 0.6: 'orange', 1.0: 'red'},
                radius=20,
                blur=15
            ).add_to(m_india)
            for _, row in all_cities_df.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5,
                    popup=f"{row['City']}, {row['State']}<br>AQI: {row['AQI']:.2f}",
                    tooltip=row['City'],
                    fill=True,
                    fill_color=aqi_color if row['City'] == city_input else '#1f77b4',
                    color=None,
                    fill_opacity=0.7
                ).add_to(m_india)
                # Add AQI value label
                folium.Marker(
                    [row['Latitude'] + 0.1, row['Longitude'] + 0.1],  # Slightly offset for visibility
                    icon=folium.DivIcon(html=f'<div style="font-size: 10px; font-weight: bold; color: black;">{row["AQI"]:.2f}</div>')
                ).add_to(m_india)
            folium_static(m_india, width=600, height=400)

    except KeyError as e:
        st.error(f"KeyError: {str(e)}. Please ensure the selected state and city are valid.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Note about the app
st.markdown("""
---
**Note**: This app uses a pre-trained Random Forest model (`random_forest_model.pkl`), location data from `locations.csv`, and a scaler from `scaler.pkl`. The State and City encoders are generated directly from `locations.csv`. Ensure the model and scaler were trained with consistent encodings for accurate predictions.
""")