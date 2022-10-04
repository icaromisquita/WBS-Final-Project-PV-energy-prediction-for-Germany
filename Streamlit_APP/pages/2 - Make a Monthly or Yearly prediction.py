from requests import options
import streamlit as st

import pvlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

st.title("Pv energy Prediction")
 
st.write("""
### Project description
A collection of Machine Learning models to predict energy generation by a PV solar system in Germany.""")

st.write("""
#### All models were created assuming the following parameters: 
        🔻
        PV tilt angle = 35° \n 
        Pv system without solar tracker\n
        PV cells technology is Crystal Silicon\n
        Inverter = MICRO-0.25-I-OUTD-US-208/208V\n
        Module = Canadian Solar CS5P_220M 2009\n
        Mounting type = Free, it means the PV system is not used as a replacement for a building material. Ex. A building facade,and\n
        Losses were = 15%.\n
    🔺
""")


st.write("""##### It is provided a collection of pre-trained models, please feel free to choose the one that better fits your needs""")
import pickle
import pandas as pd
model1 = pickle.load(open('./data/model1_berlin10peak.sav', 'rb'))
model2 = pickle.load(open('./data/model2_berlin20peak.sav', 'rb'))
model3 = pickle.load(open('./data/model3_hamburg10peak.sav', 'rb'))
model4 = pickle.load(open('./data/model4_hamburg20peak.sav', 'rb'))
model5 = pickle.load(open('./data/model5_munich10peak.sav', 'rb'))
model6 = pickle.load(open('./data/model6_munich20peak.sav', 'rb'))
model7 = pickle.load(open('./data/model7_cologne10peak.sav', 'rb'))
model8 = pickle.load(open('./data/model8_cologne20peak.sav', 'rb'))

### Model selection

mapping = {1: "Model for Berlin with 10 Kwp", 
           2: "Model for Berlin with 20 Kwp",
           3: "Model for Hamburg with 10 KWp",
           4: "Model for Hamburg with 20 KWp",
           5: "Model for Munich with 10 KWp",
           6: "Model for Munich with 20 KWp",
           7: "Model for Cologne with 10 KWp",
           8: "Model for Cologne with 20 KWp"}
models = [model1, model2, model3, model4, model5, model6, model7, model8]


models_choice = st.radio("Model Selector", (1, 2,3,4,5,6,7,8), format_func=lambda x: mapping[x])
###
pv_dfbase = pd.read_csv("./data/pv_final_cleaned.csv") 
pv_dfbase.rename(columns={"IR(h)":"IR_h"}, inplace=True)
base_df = pd.read_csv("./data/pv_final_cleaned.csv") 

st.write("""For a prediction for more days, please upload a ".csv" file bellow, following the "pvbase.csv" provided on the bottom of the page""")

# st.write("""For Yearly prediction please upload a .csv file, using the base csv provided below""")

## Upload
uploaded_file = st.file_uploader("Upload a .csv file to make a prediction for a bigger timeframe", type=None, 
    key="uploaded_file", help="Please follow the base csv", )
if uploaded_file is not None:
    upload_df = pd.read_csv(uploaded_file)
    st.write(upload_df)
else:
    print("Please provide a valid csv file")
#####

st.write("## The model to follow for a valid input.")
st.write("\n")
st.write("\n")
####Download Button
st.write("### 📥The recommended file to create a valid input.📥")
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
csv = convert_df(base_df)

st.write("\n")
st.write("\n")

st.download_button(
    label="📂Base File Download📂",
    data=csv,
    file_name='base_df.csv',
    mime='text/csv',
)
##########

st.write("## Let's make a prediction")

st.write("\n")

if st.button("⚡Make a prediction⚡", key=None, help="On click makes a prediction with the inputed parameters"):
    prediction = models[models_choice].predict(upload_df)/1000
    #st.write('The Predict energy is {0:.2f} Wh'.format(prediction))
    #st.write('The Predict energy is {0:.2f} KWh'.format((prediction)/1000))
    y_predict = pd.DataFrame(data=prediction.round(2))
    y_predict.rename(columns={0:"Predicted Energy (KWh)"}, inplace=True)
    st.bar_chart(data=y_predict)
    

from PIL import Image
image = Image.open('./img/logo-removebg-preview.png')

st.image(image, caption='PV module representation')




