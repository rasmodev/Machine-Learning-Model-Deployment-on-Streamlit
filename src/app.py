import streamlit as st

# Create a web app using Streamlit
st.title('Store Sales Forecasting Web App')

# Add input widgets for user input
store_type = st.selectbox('Select Store Type', ['Type A', 'Type B', 'Type C', 'Type D', 'Type E'])
location = st.selectbox('Select Location', ['Quito', 'Guayaquil', 'Santo Domingo', 'Cuenca', 'Manta'])
date = st.date_input('Select Date')

# Add a button to trigger the predictions
if st.button('Predict Sales'):
    # For now, let's just display the selected inputs
    st.write('Selected Store Type:', store_type)
    st.write('Selected Location:', location)
    st.write('Selected Date:', date)

