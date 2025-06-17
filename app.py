
import streamlit as st
import numpy as np
import pickle

# Load your trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("Laptop Price Predictor")

company = st.selectbox("Brand", ['HP','Dell','Apple','Acer','Lenovo'])
type = st.selectbox("Type", ['Ultrabook','Notebook','Gaming','Workstation'])
ram = st.selectbox("RAM (GB)", [4,8,16,32])
weight = st.number_input("Weight of the Laptop")
touchscreen = st.selectbox("Touchscreen", ['No','Yes'])
ips = st.selectbox("IPS Display", ['No','Yes'])
screen_size = st.number_input("Screen Size (inches)")
resolution = st.selectbox("Screen Resolution", ['1920x1080','1366x768','1600x900'])
cpu = st.selectbox("CPU Brand", ['Intel Core i5','Intel Core i7','AMD Ryzen','Intel Core i3'])
hdd = st.selectbox("HDD (GB)", [0,128,256,512,1024])
ssd = st.selectbox("SSD (GB)", [0,128,256,512,1024])
gpu = st.selectbox("GPU Brand", ['Nvidia','AMD','Intel'])
os = st.selectbox("Operating System", ['Windows','Mac','Linux','Others'])

# Calculate PPI
X_res = int(resolution.split('x')[0])
Y_res = int(resolution.split('x')[1])
ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

# Binary encoding
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os], dtype=object)
query = query.reshape(1, 12)

if st.button("Predict Price"):
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.success(f"The predicted price of this configuration is ₹{predicted_price}")
