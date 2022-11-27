import gradio as gr
import numpy as np
from PIL import Image
import pandas as pd
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("car_prices", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/car_prices_model.pkl")



def car(km_driven, mileage, engine, max_power, seats, age, seller_type, fuel_type, transmission_type):
    input_list = []
    input_list.append(km_driven)
    input_list.append(mileage)
    input_list.append(engine)
    input_list.append(max_power)
    input_list.append(seats)
    input_list.append(age)
    input_list.append(seller_type)
    input_list.append(fuel_type)
    input_list.append(transmission_type)
    # 'res' is a list of predictions returned as the label.
    df = pd.dataframe(input_list)
    df = pd.get_dummies(data=df, columns=['seller_type', 'fuel_type', 'transmission_type'])
    df = df.rename(columns = {'seller_type_Trustmark Dealer': 'seller_type_Trustmark_Dealer'})    
    res = model.predict(df)           
    return res
        
demo = gr.Interface(
    fn=car,
    title="Car Price Predictive Analytics",
    description="Experiment with car details to predict the price of your car.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=10000.0, label="Kilometers driven"),
        gr.inputs.Number(default=12.0, label="Mileage (in KM/L)"),
        gr.inputs.Number(default=1199.0, label="Engine Size (in cc)"),
        gr.inputs.Number(default=70.0, label="Max Power (in BHP)"),
        gr.inputs.Number(default=5, label="Number of Seats"),
        gr.inputs.Number(default=4.0, label="Age of the car"),
        gr.inputs.Dropdown(choices=["Dealer", "Individual", "Trademark Dealer"]),
        gr.inputs.Dropdown(choices=["Petrol", "Diesel", "Other"]),
        gr.inputs.Dropdown(choices=["Automatic", "Manual"]),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()
