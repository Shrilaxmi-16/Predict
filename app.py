import streamlit as st
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
st.info('This App is for machine learning analysis')
with st.expander('Data'):
  st.write('## Dataset')
  data= pd.read_csv('https://raw.githubusercontent.com/sumukhahe/ML_Project/main/data/dataset.csv')
  data

# Preprocessing data for Prophet
def preprocess_data_for_prophet(data):
    # We need two columns: 'ds' (date) and 'y' (value we want to predict, i.e., Employment Demand)
    data = data[['year', 'Employment_demanded']].dropna()  # Drop rows with missing Employment_demanded
    data['year'] = pd.to_datetime(data['year'], format='%Y')
    data = data.rename(columns={'year': 'ds', 'Employment_demanded': 'y'})  # Prophet requires columns ds and y
    return data

# Main function to render the Streamlit app
def main():
    st.title('MGNREGA Employment Demand Prediction for Future Years')

    # Load the dataset
    data = load_data()

    # Preprocess the data for Prophet
    prophet_data = preprocess_data_for_prophet(data)

    # Display data
    st.subheader("Historical Data")
    st.dataframe(prophet_data)

    # Create and fit the Prophet model
    model = Prophet()
    model.fit(prophet_data)

    # Make future dataframe for 2024 and 2025 predictions
    future = model.make_future_dataframe(periods=2, freq='Y')
    forecast = model.predict(future)

    # Display the forecast data
    st.subheader("Forecasted Employment Demand for 2024 and 2025")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))  # Show only the future predictions

    # Plot the forecast
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Plot components (trend, yearly seasonality)
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
