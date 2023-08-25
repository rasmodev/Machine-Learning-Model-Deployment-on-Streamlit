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

st.image("https://pbs.twimg.com/media/DywhyJiXgAIUZej?format=jpg&name=medium")
st.title("Sales Prediction App")

st.caption("This app predicts sales patterns of Corporation Favorita over time in different stores in Ecuador based on the inputs.")

# Design the sidebar
st.sidebar.header("Description of The Required Input Fields")

# Create a sidebar with input field descriptions
st.sidebar.markdown("## Input Field Descriptions")
st.sidebar.markdown("**Store Number**: The number of the store.")
st.sidebar.markdown("**Product Family**: Product Family such as 'AUTOMOTIVE', 'BEAUTY', etc.")
st.sidebar.markdown("**Number of Items on Promotion**: Number of items on promotion within a particular shop.")
st.sidebar.markdown("**city**: City where the store is located.")
st.sidebar.markdown("**cluster**: Cluster number which is a grouping of similar stores.")
st.sidebar.markdown("**transactions**: Number of transactions.")
st.sidebar.markdown("**Crude Oil Price**: Daily Crude Oil Price.")

# Create the input fields
input_df = {}
col1, col2 = st.columns(2)
with col1:
    input_df['store_nbr'] = st.slider("Store Number", 0, 54)
    input_df['family'] = st.selectbox("Product Family", ['AUTOMOTIVE', 'BEAUTY', 'CELEBRATION', 'CLEANING', 'CLOTHING', 'FOODS', 
                                            'GROCERY', 'HARDWARE', 'HOME', 'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER', 
                                            'PET SUPPLIES', 'STATIONERY'])
    input_df['onpromotion'] = st.number_input("Number of Items on Promotion", step=1)
    input_df['city'] = st.selectbox("City",  ['Ambato', 'Babahoyo', 'Cayambe', 'Cuenca', 'Daule', 'El Carmen', 'Esmeraldas',
                                         'Guaranda', 'Guayaquil', 'Ibarra', 'Latacunga', 'Libertad', 'Loja', 'Machala', 'Manta',
                                         'Playas', 'Puyo', 'Quevedo', 'Quito', 'Riobamba', 'Salinas', 'Santo Domingo'])
    input_df['cluster'] = st.selectbox("Cluster", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    input_df['transactions'] = st.number_input("Number of Transactions")

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

    # Reorder the columns to match the training column order
    input_data = input_data[['store_nbr', 'family', 'onpromotion', 'city', 'cluster', 'transactions', 'dcoilwtico', 'year', 'month', 'day', 'day_of_week']]

    # Scale the Numerical Columns (Min-Max Scaling)
    # create an instance of StandardScaler
    scaler = StandardScaler()

    # select the numerical columns
    num_cols = ['transactions', 'dcoilwtico']

    # Scale the numerical columns
    input_data[num_cols] = scaler.fit_transform(input_data[num_cols])

    # Encode the categorical columns using the trained encoder
    cat_cols = ['family', 'city']
    encoded_data = encoder.transform(input_data[cat_cols])

    # Convert the encoded data to a DataFrame
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['family', 'city']))

    # Concatenate the original dataframe with the encoded data
    final_df = pd.concat([input_data, encoded_df], axis=1)

    # Drop the original categorical columns
    final_df.drop(cat_cols, axis=1, inplace=True)

    # Add a new column called predicted_sales
    final_df['predicted_sales'] = model.predict(final_df)

    # Print the tabulated dataframe
    print(tabulate(final_df, headers=final_df.columns, tablefmt='grid'))

    # Display the prediction
    st.write("The predicted sales are:")
    st.table(final_df['predicted_sales'])
    st.table(final_df)