import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
from tabulate import tabulate

# Load the best model and relevant data
with open('model_and_key_components.pkl', 'rb') as model_file:
    loaded_data = pickle.load(model_file)
    model = loaded_data['model']
    unique_category_values = loaded_data['unique_category_values']
    min_oil_price = loaded_data['min_oil_price']
    max_oil_price = loaded_data['max_oil_price']
    best_hyperparameters = loaded_data['best_hyperparameters']
    encoder = loaded_data['encoder']
    scaler = loaded_data['scaler']


# Create a Streamlit app
st.title('Sales Prediction App')

# Add a sidebar with widgets for the input features
st.sidebar.title('Input Features')

store_nbr = st.sidebar.number_input('Store Number', min_value=1, max_value=9999)
family = st.sidebar.selectbox('Product Family', unique_category_values['family'])
city = st.sidebar.selectbox('City', unique_category_values['city'])
holiday_type = st.sidebar.selectbox('Holiday Type', unique_category_values['holiday_type'])

onpromotion = st.sidebar.checkbox('On Promotion')
if onpromotion:
    onpromotion_str = 'True'
else:
    onpromotion_str = 'False'
cluster = st.sidebar.number_input('Customer Cluster', min_value=1, max_value=10)
transactions = st.sidebar.number_input('Number of Transactions', min_value=0)
dcoilwtico = st.sidebar.number_input('Daily Oil Price', min_value=0)

# Use the model to predict the sales
input_data = {
    'store_nbr': store_nbr,
    'family': family,
    'city': city,
    'holiday_type': holiday_type,
    'onpromotion': onpromotion_str,
    'cluster': cluster,
    'transactions': transactions,
    'dcoilwtico': dcoilwtico
}

# Convert the input data to a Pandas DataFrame
input_data_df = pd.DataFrame(input_data)

# Check if the input data is empty
if not input_data_df.empty:

    # Get the feature importances
    feature_importances = model.feature_importances_

    # Keep only the features with importance above a threshold
    seen_features = set(feature_importances[feature_importances > 0.0])
    seen_features = [feature for feature in seen_features if feature in input_data_df.columns]

    # Filter the input data to keep only the seen features
    input_data_df = input_data_df[seen_features]

    # Make predictions
    prediction = model.predict(input_data_df)

    # Display the prediction
    st.write({'Predicted Sales': prediction})

else:
    st.write('Please enter some input data.')
