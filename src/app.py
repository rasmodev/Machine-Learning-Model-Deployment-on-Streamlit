import pandas as pd
import streamlit as st
import numpy as np
import pickle

# Load the saved components:
with open("rf_model.pkl", "rb") as f:
    components = pickle.load(f)

# Extract the individual components
num_imputer = components["num_imputer"]
cat_imputer = components["cat_imputer"]
encoder = components["encoder"]
scaler = components["scaler"]
dt_model = components["models"]

st.image("https://pbs.twimg.com/media/DywhyJiXgAIUZej?format=jpg&name=medium")
st.title("Sales Prediction App")

st.caption("This app predicts sales patterns of Corporation Favorita over time in different stores in Ecuador based on the inputs.")

# Sidebar with input field descriptions
st.sidebar.header("Description of The Required Input Fields")
st.sidebar.markdown("**Store Number**: The number of the store.")
st.sidebar.markdown("**Product Family**: Product Family such as 'AUTOMOTIVE', 'BEAUTY', etc. "
                    "Details:\n"
                    " - **AUTOMOTIVE**: Products related to the automotive industry.\n"
                    " - **BEAUTY**: Beauty and personal care products.\n"
                    " - **CELEBRATION**: Products for celebrations and special occasions.\n"
                    " - **CLEANING**: Cleaning and household maintenance products.\n"
                    " - **CLOTHING**: Clothing and apparel items.\n"
                    " - **FOODS**: Food items and groceries.\n"
                    " - **GROCERY**: Grocery products.\n"
                    " - **HARDWARE**: Hardware and tools.\n"
                    " - **HOME**: Home improvement and decor products.\n"
                    " - **LADIESWEAR**: Women's clothing.\n"
                    " - **LAWN AND GARDEN**: Lawn and garden products.\n"
                    " - **LIQUOR,WINE,BEER**: Alcoholic beverages.\n"
                    " - **PET SUPPLIES**: Products for pets and animals.\n"
                    " - **STATIONERY**: Stationery and office supplies.")
st.sidebar.markdown("**Number of Items on Promotion**: Number of items on promotion within a particular shop.")
st.sidebar.markdown("**City**: City where the store is located.")
st.sidebar.markdown("**Cluster**: Cluster number which is a grouping of similar stores.")
st.sidebar.markdown("**Transactions**: Number of transactions.")
st.sidebar.markdown("**Crude Oil Price**: Daily Crude Oil Price.")


 # Create the input fields
input_data = {}
col1,col2,col3 = st.columns(3)
with col1:
    input_data['store_nbr'] = st.slider("Store Number",0,54)
    input_data['family'] = st.selectbox("Product Family", ['AUTOMOTIVE', 'BEAUTY', 'CELEBRATION', 'CLEANING', 'CLOTHING', 'FOODS', 
                                            'GROCERY', 'HARDWARE', 'HOME', 'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER', 
                                            'PET SUPPLIES', 'STATIONERY'])
    input_data['onpromotion'] =st.number_input("Number of Items on Promotion",step=1)
    input_data['state'] = st.selectbox("State Where The Store Is Located", ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
       'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza', 'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
       'El Oro', 'Esmeraldas', 'Manabi'])
    input_data['transactions'] = st.number_input("Number of Transactions", step=1)

with col2:    
    input_data['store_type'] = st.selectbox("Store Type",['A', 'B', 'C', 'D', 'E'])
    input_data['cluster'] = st.selectbox("Cluster", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    input_data['dcoilwtico'] = st.number_input("Crude Oil Price",step=1)
    input_data['year'] = st.number_input("Year",step=1)
with col3:    
    input_data['month'] = st.slider("Month",1,12)
    input_data['day'] = st.slider("Day",1,31)
    input_data['dayofweek'] = st.number_input("Day of Week (0=Sunday and 6=Satruday)",step=1)


  # Create a button to make a prediction
if st.button("Predict"):

    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Product Categorization Based on Families
    food_families = ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI', 'PRODUCE', 'DAIRY', 'POULTRY', 'EGGS', 'SEAFOOD']
    home_families = ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES']
    clothing_families = ['LINGERIE', 'LADYSWARE']
    grocery_families = ['GROCERY I', 'GROCERY II']
    stationery_families = ['BOOKS', 'MAGAZINES', 'SCHOOL AND OFFICE SUPPLIES']
    cleaning_families = ['HOME CARE', 'BABY CARE', 'PERSONAL CARE']
    hardware_families = ['PLAYERS AND ELECTRONICS', 'HARDWARE']

    # Apply the same preprocessing steps as done during training
    input_df['family'] = np.where(input_df['family'].isin(food_families), 'FOODS', input_df['family'])
    input_df['family'] = np.where(input_df['family'].isin(home_families), 'HOME', input_df['family'])
    input_df['family'] = np.where(input_df['family'].isin(clothing_families), 'CLOTHING', input_df['family'])
    input_df['family'] = np.where(input_df['family'].isin(grocery_families), 'GROCERY', input_df['family'])
    input_df['family'] = np.where(input_df['family'].isin(stationery_families), 'STATIONERY', input_df['family'])
    input_df['family'] = np.where(input_df['family'].isin(cleaning_families), 'CLEANING', input_df['family'])
    input_df['family'] = np.where(input_df['family'].isin(hardware_families), 'HARDWARE', input_df['family'])

    categorical_columns = ['family', 'store_type', 'state']
    numerical_columns = ['transactions', 'dcoilwtico']

    # Impute missing values
    input_df_cat = input_df[categorical_columns].copy()
    input_df_num = input_df[numerical_columns].copy()
    input_df_cat_imputed = cat_imputer.transform(input_df_cat)
    input_df_num_imputed = num_imputer.transform(input_df_num)

    # Encode categorical features
    input_df_cat_encoded = pd.DataFrame(encoder.transform(input_df_cat_imputed).toarray(),
                                        columns=encoder.get_feature_names_out(categorical_columns))

    # Scale numerical features
    input_df_num_scaled = scaler.transform(input_df_num_imputed)
    input_df_num_sc = pd.DataFrame(input_df_num_scaled, columns=numerical_columns)

    # Combine encoded categorical features and scaled numerical features
    input_df_processed = pd.concat([input_df_num_sc, input_df_cat_encoded], axis=1)

    # Make predictions using the trained model
    predictions = dt_model.predict(input_df_processed)

    # Display the predicted sales value to the user:
    st.write("The predicted sales are:", predictions[0])