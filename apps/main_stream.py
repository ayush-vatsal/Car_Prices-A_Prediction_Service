import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import requests

import hopsworks
import joblib

st.set_page_config(
    page_title='Car Prices Predictive Analysis',
    page_icon='',
    layout='wide'
)

project = hopsworks.login()
fs = project.get_feature_store()

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def get_model():
    mr = project.get_model_registry()
    model = mr.get_model("car_prices", version=1)
    model_dir = model.download()
    return joblib.load(model_dir + "/car_prices_model.pkl")

header = st.container()
model_train = st.container()
with header:
    st.title("Car Prices Predictive analysis")
    col_a, col_b = st.columns(2)
    km = col_a.number_input("Kilometers Driven", 1000, 1000000, 10000, 1000)
    engine = col_b.number_input("Engine size (in CC)", 600, 6000, 1200, 100)
    power = col_a.number_input("Maximum Power in BHP", 10.0, 1000.0, 80.0, 2.0)
    seats = col_b.slider("Number of Seats", 2, 10, 5, 1)
    age = col_a.slider("Age of the car in years", 1, 10, 2)
    seller = col_b.selectbox(
        "Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    fuel = col_a.selectbox(
        "Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = col_b.selectbox(
        "Transmission Type", ["Manual", "Automatic"])

input_list = [km, 12, engine, power, seats, age, seller, fuel, transmission]

if (input_list[6] == "Dealer"):
    input_list.pop(6)
    input_list.insert(6, 1)
    input_list.insert(7, 0)
    input_list.insert(8, 0)
if (input_list[6] == "Individual"):
    input_list.pop(6)
    input_list.insert(6, 0)
    input_list.insert(7, 1)
    input_list.insert(8, 0)
if (input_list[6] == "Trustmark Dealer"):
    input_list.pop(6)
    input_list.insert(6, 0)
    input_list.insert(7, 0)
    input_list.insert(8, 1)

if (input_list[9] == "CNG"):
    input_list.pop(9)
    input_list.insert(9, 1)
    input_list.insert(10, 0)
    input_list.insert(11, 0)
    input_list.insert(12, 0)
if (input_list[9] == "Diesel"):
    input_list.pop(9)
    input_list.insert(9, 0)
    input_list.insert(10, 1)
    input_list.insert(11, 0)
    input_list.insert(12, 0)
if (input_list[9] == "Electric"):
    input_list.pop(9)
    input_list.insert(9, 0)
    input_list.insert(10, 0)
    input_list.insert(11, 1)
    input_list.insert(12, 0)
if (input_list[9] == "Petrol"):
    input_list.pop(9)
    input_list.insert(9, 0)
    input_list.insert(10, 0)
    input_list.insert(11, 0)
    input_list.insert(12, 1)

if (input_list[13] == "Automatic"):
    input_list.pop(13)
    input_list.insert(13, 1)
    input_list.insert(14, 0)
if (input_list[13] == "Manual"):
    input_list.pop(13)
    input_list.insert(13, 0)
    input_list.insert(14, 1)
# mileage, engine, max_power, seats, age, seller_type, fuel_type, transmission_type
df = pd.DataFrame(input_list)
model = get_model()


with model_train:
    disp = st.columns(5)
    pred_button = disp[2].button('Evaluate price')
    if pred_button:
        res = model.predict(df.T)[0].round(4)
        with st.spinner():
            st.write(f'#### Evaluated price of the car(in lakhs): ₹ {res:,.4f}') 
