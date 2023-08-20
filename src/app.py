import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
from tabulate import tabulate

# Load the best model and relevant data
with open('model_and_data.pkl', 'rb') as model_file:
    loaded_data = pickle.load(model_file)
    best_model = loaded_data['model']
    unique_category_values = loaded_data['unique_category_values']
    min_oil_price = loaded_data['min_oil_price']
    max_oil_price = loaded_data['max_oil_price']
    best_hyperparameters = loaded_data['best_hyperparameters']

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

    # Encode categorical variables using drop='first' and fit with training data
    encoder = OneHotEncoder(categories=[unique_category_values['family'], unique_category_values['city'], unique_category_values['holiday_type']], drop='first', sparse=False)
    encoder.fit(data[["family", "city", "holiday_type"]])
    encoded_data = encoder.transform(data[["family", "city", "holiday_type"]])
    column_names = encoder.get_feature_names_out(["family", "city", "holiday_type"])
    encoded_df = pd.DataFrame(encoded_data, columns=column_names)
    
    # Combine encoded features with numerical features
    data_encoded = pd.concat([data.drop(["family", "city", "holiday_type"], axis=1), encoded_df], axis=1)
    
    # Feature Scaling
    scaler = StandardScaler()
    num_cols = ['transactions', 'dcoilwtico']
    data_encoded[num_cols] = scaler.fit_transform(data_encoded[num_cols])

    return data_encoded

# Streamlit App
def main():
    st.title("Sales Prediction App")
    st.write("This app predicts sales based on input features.")

    # User inputs
    year = st.number_input("Year", min_value=2013, max_value=2017, step=1, value=2017)
    month = st.slider("Month", 1, 12)
    day = st.slider("Day", 1, 31)
    day_of_week = st.number_input("Day of Week")
    family = st.selectbox("Product Family", unique_category_values['family'])
    city = st.selectbox("City", unique_category_values['city'])
    holiday_type = st.selectbox("Holiday Type", unique_category_values['holiday_type'])
    transactions = st.number_input("Number of Transactions", min_value=0, max_value=10000, value=0)
    onpromotion = st.number_input("Number of Items on Promotion")
    dcoilwtico = st.number_input("Crude Oil Price (dcoilwtico)")

# Add "Predict Sales" Button
    if st.button("Predict Sales"):
        # Create a dictionary with input data
        input_df = pd.DataFrame({
            "family": [family],
            "onpromotion": [onpromotion],
            "city": [city],
            "transactions": [transactions],
            "holiday_type": [holiday_type],
            "dcoilwtico": [dcoilwtico],
            "year": [year],
            "month": [month],
            "day": [day],
            "day_of_week": [day_of_week]
        })

        # Preprocess the input data
        processed_input = preprocess_data(input_df)


        # Reorder columns of processed input to match expected order
        column_names = processed_input.columns.tolist()  # Get column names in current order
        column_names.remove('year')  # Remove 'year' as it's not needed for prediction
        final_processed_input = processed_input[['year'] + column_names]

        # Print the processed input DataFrame
        print("Processed Input DataFrame:")
        print(tabulate(final_processed_input, headers='keys', tablefmt='grid'))


        # Make prediction
        prediction = best_model.predict(final_processed_input)


        st.write("Predicted Sales:", prediction)

if __name__ == "__main__":
    main()