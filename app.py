import streamlit as st
import requests

# Streamlit UI
st.title("Player Performance Prediction")

# Input fields
position_encoded = st.number_input("Position Encoded", min_value=0)
winger = st.number_input("Winger (0 or 1)", min_value=0, max_value=1)
appearance = st.number_input("Appearances", min_value=0)
award = st.number_input("Awards", min_value=0)  # This may not be used in your model
current_value = st.number_input("Current Value", min_value=0)
goals = st.number_input("Goals", min_value=0.0)
assists = st.number_input("Assists", min_value=0.0)
goals_conceded = st.number_input("Goals Conceded", min_value=0.0)

# Button to trigger prediction
if st.button("Predict"):
    # Prepare the input data
    input_data = {
        "position_encoded": position_encoded,
        "winger": winger,
        "appearance": appearance,
        "award": award,
        "current_value": current_value,
        "goals": goals,
        "assists": assists,
        "goals_conceded": goals_conceded
    }
    
    # Make a request to the FastAPI endpoint
    response = requests.post("https://deploy-ml-r29i.onrender.com/predict", json=input_data)
    
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"Prediction: {prediction}")
    else:
        st.error("Error in prediction. Please check your inputs.")