import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st


model = joblib.load('model.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')

def collect_user_input():
    bedroom = st.number_input('No of Bedrooms')
    fullbath = st.number_input('No of Bathrooms')
    lotarea= st.number_input('Area(sqft)')
    age = st.number_input('How old is the house(years)?')
    location = st.selectbox('Location', options=['Urban', 'SubUrban', 'Rural'])

    input_data = pd.DataFrame(data= [[age, bedroom, fullbath, lotarea, location]],
        columns=['HouseAge', 'Bedroom', 'FullBath', 'LotArea', 'Location'])

    return input_data

st.title('House Price Prediction App')
st.write('Predict the price of an apartment')
input_data = collect_user_input()
numerical_column = input_data.select_dtypes('number').columns
input_data['Location'] = encoder.transform(input_data['Location'])
input_data[numerical_column]= scaler.transform(input_data[numerical_column])
    
if st.button('predict'):
    prediction = model.predict(input_data)
    st.success(f'The apartment would cost around USD {int(prediction[0])}.')