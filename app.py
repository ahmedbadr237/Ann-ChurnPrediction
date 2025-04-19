import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
robust_scaler = pickle.load(open('Models\\robus_scaler.pkl', 'rb'))
gender = pickle.load(open('Models\\gender.pkl', 'rb'))
model = load_model('Models\\no_ol_model.h5')
geo = pickle.load(open('Models\\geo.pkl', 'rb'))

### Title of the app
st.title("Customer Churn Prediction App")
geogrpahy = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender_data = st.selectbox('Gender',['Male','Female'])
age = st.number_input('Age', 18, 100, 30)
tenure = st.number_input('Tenure', 0, 10, 5)
balance = st.number_input('Balance')
num_of_products = st.number_input('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', [1, 0])
is_active_member = st.selectbox('Is Active Member', [1, 0])
estimated_salary = st.number_input('Estimated Salary')
credit_score = st.number_input('Credit Score', 0, 850, 700)

input_data = {'CreditScore':credit_score,
             'Age':age,
             'Tenure': tenure,
             'Balance': balance,
             'NumOfProducts': num_of_products,
             'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
             'EstimatedSalary': estimated_salary,
             'Gender': gender_data,
             'Geography': geogrpahy}

input_data = pd.DataFrame([input_data])
input_data = pd.concat([input_data,
                        pd.DataFrame(columns=gender.get_feature_names_out(),
                                     data=gender.transform(input_data[['Gender']]).toarray())],
                       axis=1).drop(columns=['Gender'])
input_data = pd.concat([input_data,
               pd.DataFrame(columns=geo.get_feature_names_out(),
                        data=geo.transform(input_data[['Geography']]).toarray())],axis=1)
input_data.drop(columns=['Geography'],inplace=True)
#predicting the result
if st.button('Predict'):
    prediction = model.predict(input_data)
    result = prediction[0][0]
    if result < 0.5:
        st.success('The customer will not leave the bank')
    else:
        st.error('The customer will leave the bank')

