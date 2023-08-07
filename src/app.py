import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle

# Load the best model
with open('best_rf_model_components.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

# Load the feature engineering steps
def preprocess_data(data):
    if 'date' in data.columns:
        # Extracting Date Components
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day

        # Dropping Unnecessary Columns
        columns_to_drop = ['date', 'id', 'locale', 'locale_name', 'description', 'store_type', 'transferred', 'state']
        data = data.drop(columns=columns_to_drop)

        # Product Categorization Based on Families
        food_families = ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI', 'PRODUCE', 'DAIRY', 'POULTRY', 'EGGS', 'SEAFOOD']
        home_families = ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES']
        clothing_families = ['LINGERIE', 'LADYSWARE']
        grocery_families = ['GROCERY I', 'GROCERY II']
        stationery_families = ['BOOKS', 'MAGAZINES', 'SCHOOL AND OFFICE SUPPLIES']
        cleaning_families = ['HOME CARE', 'BABY CARE', 'PERSONAL CARE']
        hardware_families = ['PLAYERS AND ELECTRONICS', 'HARDWARE']

        data['family'] = np.where(data['family'].isin(food_families), 'FOODS', data['family'])
        data['family'] = np.where(data['family'].isin(home_families), 'HOME', data['family'])
        data['family'] = np.where(data['family'].isin(clothing_families), 'CLOTHING', data['family'])
        data['family'] = np.where(data['family'].isin(grocery_families), 'GROCERY', data['family'])
        data['family'] = np.where(data['family'].isin(stationery_families), 'STATIONERY', data['family'])
        data['family'] = np.where(data['family'].isin(cleaning_families), 'CLEANING', data['family'])
        data['family'] = np.where(data['family'].isin(hardware_families), 'HARDWARE', data['family'])

    # Feature Scaling
    scaler = StandardScaler()
    num_cols = ['sales', 'transactions', 'dcoilwtico']
    data[num_cols] = scaler.fit_transform(data[num_cols])

    # Encoding The Categorical Variables
    categorical_columns = ["family", "city", "holiday_type"]
    encoder = OneHotEncoder()
    one_hot_encoded_data = encoder.fit_transform(data[categorical_columns])

    column_names = encoder.get_feature_names_out(categorical_columns)
    data_encoded = pd.DataFrame(one_hot_encoded_data.toarray(), columns=column_names)

    data_encoded = pd.concat([data, data_encoded], axis=1)
    data_encoded.drop(categorical_columns, axis=1, inplace=True)

    return data_encoded

# Streamlit App
def main():
    st.title("Sales Prediction App")
    st.write("This app predicts sales based on input features.")

    # User inputs
    year = st.slider("Year", 2013, 2017)
    month = st.slider("Month", 1, 12)
    day = st.slider("Day", 1, 31)
    family = st.selectbox("Product Family", ["FOODS", "HOME", "CLOTHING", "GROCERY", "STATIONERY", "CLEANING", "HARDWARE"])
    city = st.selectbox("City", ["Quito", "Santo Domingo", "Cuenca", "Guayaquil", "Manta", "Ambato", "Puyo", "Machala", "Salinas", "Esmeraldas", "Libertad", "Babahoyo", "Quevedo", "Latacunga", "Loja", "Riobamba"])
    holiday_type = st.selectbox("Holiday Type", ["None", "Holiday", "Event", "Bridge", "Transfer", "Additional"])
    transactions = st.slider("Transactions", 0, 10000)
    dcoilwtico = st.slider("Oil Price", -1.5, 2.5)
    
    input_data = pd.DataFrame({
        'year': [year],
        'month': [month],
        'day': [day],
        'family': [family],
        'city': [city],
        'holiday_type': [holiday_type],
        'transactions': [transactions],
        'dcoilwtico': [dcoilwtico]
    })

    # Preprocess the input data
    processed_input = preprocess_data(input_data)

    # Make prediction
    prediction = best_model.predict(processed_input)

    st.write("Predicted Sales:", prediction)

if __name__ == "__main__":
    main()
