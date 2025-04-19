# Nama: Felicia Audrey Tanujaya
# NIM: 2702217610
# Kelas: LE09 Model Deployment
# UTS Model Deployment No 3: Code Inference
# !pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

with gzip.open('rfmodel.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('oh_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

categorical = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
numerical = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month', 'arrival_date', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']


def main():
    st.title("Hotel Booking Prediction")

    no_of_adults = st.slider("Number of Adults", 0, 4, 1)
    no_of_children = st.slider("Number of Children", 0, 10, 0)
    no_of_weekend_nights = st.slider("Weekend Nights", 0, 7, 1)
    no_of_week_nights = st.slider("Week Nights", 0, 17, 2)
    type_of_meal_plan = st.selectbox("Meal Plan", ['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
    required_car_parking_space = st.selectbox("Car Parking Space", [0, 1])
    room_type_reserved = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4'])
    lead_time = st.slider("Lead Time", 0, 443, 0)
    arrival_year = st.selectbox("Arrival Year", [2017, 2018])
    arrival_month = st.slider("Arrival Month", 1, 12, 1)
    arrival_date = st.slider("Arrival Date", 1, 31, 1)
    market_segment_type = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
    repeated_guest = st.selectbox("Repeated Guest", [0, 1])
    no_of_previous_cancellations = st.slider("Previous Cancellations", 0, 13, 0)
    no_of_previous_bookings_not_canceled = st.slider("Previous Bookings Not Canceled", 0, 58, 0)
    avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, step=0.01)
    no_of_special_requests = st.slider("Special Requests", 0, 5, 0)

    input_dict = {
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'type_of_meal_plan': [type_of_meal_plan],
        'required_car_parking_space': [required_car_parking_space],
        'room_type_reserved': [room_type_reserved],
        'lead_time': [lead_time],
        'arrival_year': [arrival_year],
        'arrival_month': [arrival_month],
        'arrival_date': [arrival_date],
        'market_segment_type': [market_segment_type],
        'repeated_guest': [repeated_guest],
        'no_of_previous_cancellations': [no_of_previous_cancellations],
        'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
        'avg_price_per_room': [avg_price_per_room],
        'no_of_special_requests': [no_of_special_requests]
    }

    df = pd.DataFrame(input_dict)

    if st.button("Predict Booking Status"):
        df_scaled = df.copy()
        df_scaled[numerical] = scaler.transform(df[numerical])
        ohe_data = ohe.transform(df[categorical])
        ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(categorical))

        final_df = pd.concat([df_scaled.drop(categorical, axis=1).reset_index(drop=True), ohe_df], axis=1)

        pred = model.predict(final_df)
        label = le.inverse_transform(pred)[0]

        st.success(f"Prediction: **{label}**")

if __name__ == "__main__":
    main()
