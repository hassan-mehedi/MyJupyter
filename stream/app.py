import numpy as np
import pandas as pd
import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv("crop_prediction_model_one.csv")

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    input = np.array([[n, p, k, temperature, humidity, ph, rainfall]]).astype(np.float64)
    prediction = model.predict(input)
    return prediction[0]

def plotDrawer(x,y):
    fig = plt.figure(figsize=(15,15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x=x, y=y, hue="label", size="label", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 14})
    plt.yticks(fontsize=16)
    st.write(fig)

def main():
    html_temp = """
    <div style="background-color:#025246 ;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Which Crop To Cultivate? </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.title("Choose the values")

    n = st.sidebar.slider('Nitrogen', 0, 140)
    p = st.sidebar.slider('Phosphorus', 5, 145)
    k = st.sidebar.slider('Potassium', 5, 205)
    temperature = st.sidebar.slider('Temperature', 8.83, 43.68)
    humidity = st.sidebar.slider('Humidity', 14.26, 99.98)
    ph = st.sidebar.slider('pH', 3.50, 9.94)
    rainfall = st.sidebar.slider('Rainfall', 20.21, 298.56)

    st.subheader("Relation between features")
    fig = plt.figure(figsize=(15, 10))
    x = st.selectbox("Select a property", ('N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall'))
    y = st.selectbox("Select a property", ('P', 'N', 'ph', 'K', 'temperature', 'humidity', 'rainfall'))
    # Plot!
    if st.button("Visulaize"):
        plotDrawer(x, y)

        
    if st.button("Predict your crop"):
        output=predict_crop(n, p, k, temperature, humidity, ph, rainfall)
        st.success('The most suitable crop for your field is {}'.format(output.upper()))

if __name__=='__main__':
    main()
    