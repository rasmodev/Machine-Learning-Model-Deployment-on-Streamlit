# Sales Prediction App

This repository contains a Streamlit web application that predicts sales patterns of Corporation Favorita over time in different stores in Ecuador based on user inputs. The predictions are made using a trained machine learning model.

## Getting Started

To use the Sales Prediction App, follow these steps:

1. Clone this repository to your local machine.

2. Install the required Python packages using the following command:
   **pip install -r requirements.txt**

3. Run the Streamlit app using the following command:
   **streamlit run app.py**
   

5. The app will open in your web browser. You can enter the relevant data in the input fields and click the "Predict" button to get sales predictions.

## App Overview

The Sales Prediction App allows you to input various parameters related to the store, products, promotions, and more, in order to predict the sales pattern. Here's a brief overview of the input fields:

- **Store Number**: The number of the store.
- **Product Family**: Product Family such as 'AUTOMOTIVE', 'BEAUTY', etc. (Choose from the available options.)
- **Number of Items on Promotion**: Number of items on promotion within a particular shop.
- **State Where The Store Is Located**: The state where the store is located. (Choose from the available options.)
- **Transactions**: Number of transactions.
- **Store Type**: The type of the store. (Choose from the available options.)
- **Cluster**: Cluster number which is a grouping of similar stores. (Choose from the available options.)
- **Crude Oil Price**: Daily Crude Oil Price.
- **Year**: Year for prediction.
- **Month**: Month for prediction.
- **Day**: Day for prediction.
- **Day of Week**: Day of the week for prediction (0=Sunday and 6=Saturday).

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Matplotlib
- Plotly

## Model and Components

The sales prediction model was trained using various machine learning techniques. The trained model is loaded and used for making predictions in the app. The following components are used in the app:

- Data imputers (numerical and categorical)
- One-hot encoder for categorical features
- Standard scaler for numerical features
- Machine learning model (Random Forest Regressor)

## App Screenshot Before Prediction

![App Screenshot Before Prediction](https://github.com/rasmodev/Machine-Learning-Model-Deployment-on-Streamlit/blob/main/screenshots/App_interface_before_pred.png)

## App Screenshot After Prediction
![App Screenshot After Prediction](https://github.com/rasmodev/Machine-Learning-Model-Deployment-on-Streamlit/blob/main/screenshots/App_Interface_After_Pred.png)

## License

This project is licensed under the [MIT License](LICENSE).








