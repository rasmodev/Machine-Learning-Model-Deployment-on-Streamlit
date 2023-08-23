import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
from tabulate import tabulate


# Loading the trained model and components
encoder = pickle.load(open('categorical_encoder.pkl', 'rb'))
model = pickle.load(open('best_rf_model.pkl', 'rb'))
unique_category_values = pickle.load(open('unique_category_values.pkl', 'rb'))

st.image("https://ultimahoraec.com/wp-content/uploads/2021/03/dt.common.streams.StreamServer-768x426.jpg")
st.title("Sales Prediction App")

st.caption("This app predicts sales patterns of Corporation Favorita over time in different stores in Ecuador based on the inputs.")

# Create the input fields
input_df = {}
col1, col2 = st.columns(2)
with col1:
    input_df['store_nbr'] = st.slider("Store Number", 0, 54)
    input_df['family'] = st.selectbox("Product Family", ['AUTOMOTIVE', 'BEAUTY', 'CELEBRATION', 'CLEANING', 'CLOTHING', 'FOODS', 
                                            'GROCERY', 'HARDWARE', 'HOME', 'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER', 
                                            'PET SUPPLIES', 'STATIONERY'])
    input_df['onpromotion'] = st.number_input("Number of Items on Promotion", step=1)
    input_df['city'] = st.selectbox("City", unique_category_values['city'])
    input_df['cluster'] = st.selectbox("Cluster", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    input_df['transactions'] = st.number_input("Number of Transactions")
    input_df['holiday_type'] = st.selectbox("Holiday Type", ['Additional', 'Bridge', 'Event', 'Holiday', 'Transfer'])    

with col2:
    input_df['dcoilwtico'] = st.slider("Crude Oil Price", min_value=1.00, max_value=100.00, step=0.1)  
    input_df['year'] = st.number_input("Year", step=1)
    input_df['month'] = st.slider("Month", 1, 12)
    input_df['day'] = st.slider("Day", 1, 31)
    input_df['day_of_week'] = st.number_input("Day of Week, 0=Sun and 6=Sat", step=1)

# Create a button to make a prediction
if st.button("Predict Sales"):
    # Convert the input data to a pandas DataFrame
    input_data = pd.DataFrame([input_df])


    # Scale the Numerical Columns(Min-Max Scaling)
    # create an instance of StandardScaler
    scaler = StandardScaler()
    
    #select the numerical columns
    num_cols = ['transactions', 'dcoilwtico']
    
    # Scale the numerical columns
    input_data[num_cols] = scaler.fit_transform(input_data[num_cols])

    # Encode the categorical columns
    cat_cols = ['family', 'city', 'holiday_type']

    # Transform the categorical columns using the fitted encoder    
    one_hot_encoded_data = encoder.fit_transform(input_data[cat_cols])

    # Create column names for the one-hot encoded data
    column_names = encoder.get_feature_names_out(cat_cols)
    
    # Convert the one-hot encoded data to a DataFrame
    final_df = pd.DataFrame(one_hot_encoded_data.toarray(), columns=column_names)
    
    # Concatenate the original dataframe with the one-hot encoded data
    final_df = pd.concat([input_data, final_df], axis=1)

    # Drop the original categorical columns
    final_df.drop(cat_cols, axis=1, inplace=True)

    # Print the tabulated dataframe
    print(tabulate(final_df, headers=final_df.columns, tablefmt='grid'))

    # Add a new column called predicted_sales
    final_df['predicted_sales'] = model.predict(final_df)[0]

    # Display the prediction
    st.write(f"The predicted sales are: {final_df['predicted_sales']}")
    st.table(final_df)

