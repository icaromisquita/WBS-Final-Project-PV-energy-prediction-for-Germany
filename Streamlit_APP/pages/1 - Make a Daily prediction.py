from requests import options
import streamlit as st

st.title("Daily Pv energy Prediction")
 
st.write("""
## Project description
#### 🔆A Machine Learning model to predict the energy generate by a PV solar system with 10 KWp in Berlin on a daily basis🔆""")

st.write("\n")
st.write("\n")

import pickle
import pandas as pd
model1 = pickle.load(open('Streamlit_APP/data/model1_berlin10peak.sav', 'rb'))
pv_dfbase = pd.read_csv("Streamlit_APP/data/pv_final_cleaned.csv") 
pv_dfbase.rename(columns={"IR(h)":"IR_h"}, inplace=True)

# Initiating the input variables

input_df = pd.read_csv("Streamlit_APP/data/pv_base.csv")

st.write("""#### Please write all inputs as a decimal number.""")

input_df["temp_air"][0] = st.number_input("Air temperature", value=input_df["temp_air"][0],step=0.01, key="temp_air", help="The air temperature in C°")
input_df["relative_humidity"][0] = st.number_input("Relative humidity", value=input_df["relative_humidity"][0], min_value=0.0, max_value=100.0,step=0.01, 
        key="rel_humi", help="The Relative Humidity in %")
input_df["ghi"][0] = st.number_input("Ghi", value=input_df["ghi"][0],step=0.01, key="ghi", help="Global horizontal irradiance in W/m²")
input_df["dni"][0] = st.number_input("Dni", value=input_df["dni"][0],step=0.01, key="dni", help="Direct normal irradiance in W/m²")
input_df["dhi"][0] = st.number_input("Dhi", value=input_df["dhi"][0],step=0.01, key="dhi", help="Diffuse horizontal irradiance in W/m²")
input_df["IR(h)"][0] = st.number_input("IR(h)", value=input_df["IR(h)"][0], step=0.01, key="IR(h)", help="Infrared radiation downwards in W/m²")
input_df["wind_speed"][0] = st.number_input("Wind speed", value=input_df["wind_speed"][0], step=0.01, key="wind_speed", help="Wind speed at a height of 10 meters in m/s")
input_df["wind_direction"][0] = st.number_input("Wind direction", value=input_df["wind_direction"][0], min_value=0.0, max_value=360.0, step=0.01, 
        key="wind_direction", help="Wind direction at a height of 10 meters in degrees from east of north (°). Ex. 360 = 0 = north, 90 = East,  0 = undefined,calm)")
input_df["pressure"][0] = st.number_input("Pressure", value=input_df["pressure"][0], step=0.01, key="pressure", help="The site pressure in Pascal (Pa).")
input_df["apparent_zenith"][0] = st.number_input("Apparent zenith", value=input_df["apparent_zenith"][0],min_value=0.0, max_value=90.0, step=0.01, key="apparent_zenith", help="Refraction-corrected solar zenith angle in degrees(°).")
input_df["zenith"][0] = st.number_input("Zenith", value=input_df["zenith"][0], min_value=0.0, max_value=90.0, step=0.01, key="zenith", help="Zenith angle of the sun in degrees (°).")
input_df["apparent_elevation"][0] = st.number_input("Apparent elevation", value=input_df["apparent_elevation"][0], min_value=0.0, max_value=90.0, step=0.01, 
        key="apparent_elevation", help="Apparent sun elevation accounting for atmospheric refraction. in degrees (°).")
input_df["elevation"][0] = st.number_input("Elevation", value=input_df["elevation"][0], step=0.01, min_value=0.0, max_value=90.0,
        key="elevation", help="Actual elevation (not accounting for refraction) of the sun in decimal degrees (°), 0 = on horizon. The complement of the zenith angle(°).")
input_df["aoi"][0] = st.number_input("Aoi", value=input_df["aoi"][0], step=0.01, min_value=0.0, max_value=360.0,
        key="aoi", help="Angle of incidence of solar rays with respect to the module surface (°).") # Observe it
input_df["poa_global"][0] = st.number_input("Poa global", value=input_df["poa_global"][0], step=0.01,
        key="poa_global", help="Global irradiation in plane. Sum of diffuse and beam projection in W/m².")
input_df["poa_direct"][0] = st.number_input("Poa direct", value=input_df["poa_direct"][0], step=0.01, key="poa_direct", help="Direct/beam irradiation in plane in W/m².")
input_df["poa_diffuse"][0] = st.number_input("Poa diffuse", value=input_df["poa_diffuse"][0], step=0.01, key="poa_diffuse", help="Total diffuse irradiation in plane. sum of ground and sky diffuse in W/m².")
input_df["poa_sky_diffuse"][0] = st.number_input("Poa sky diffuse", value=input_df["poa_sky_diffuse"][0], step=0.01, key="poa_sky_diffuse", help="Diffuse irradiation in plane from scattered light in the atmosphere (without ground reflected irradiation) in W/m².")
input_df["poa_ground_diffuse"][0] = st.number_input("Poa ground diffuse", value=input_df["poa_ground_diffuse"][0], step=0.01, key="poa_ground_diffuse", help="In plane ground reflected irradiation in W/m².")
input_df["effective_irradiance"][0] = st.number_input("Effective irradiance", value=input_df["effective_irradiance"][0], step=0.01, key="effective_irradiance", help="Effective irradiance is total plane of array (POA) irradiance adjusted for angle of incidence losses, soiling, and spectral mismatch. In a general sense it can be thought of as the irradiance that is “available” to the PV array for power conversion. W/m².")
input_df["cell_temperature"][0] = st.number_input("PV cell temperature", value=input_df["cell_temperature"][0], step=0.01, min_value=0.0, max_value=50.0, key="cell_temperature", help="Nominal operating cell temperature in C°")
input_df["ac_current"][0] = st.selectbox('Please select the AC current of the inverter', (110.0, 220.0, 360.0, 440.0))  #For some reason don't change anything, maybe the influence is very low


input_df["season"][0] = st.number_input("Season", value=input_df["season"][0], step=1, min_value=0, max_value=3, key="season", help="Meteorological Season (0 = Winter, 1 = Spring, 2 = Summer and 3 = October)")
input_df["day"][0] = st.number_input("Day", value=input_df["day"][0], step=1, min_value=1, max_value=31, key="day", help="Day of the month")
input_df["month"][0] = st.number_input("Month", value=input_df["month"][0], step=1, min_value=1, max_value=12, key="month", help="Month in numerical values")
input_df["year"][0] = st.number_input("Year", value=input_df["year"][0], step=1, min_value=2000, max_value=2500, key="year", help="Year in numerical values")


input_df["weekday_name"][0] = st.select_slider('Select a weekday',value=input_df["weekday_name"][0],  
        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday","Saturday","Sunday"])
input_df["month_name"][0] = st.select_slider('Select a month of the year',value=input_df["month_name"][0], 
        options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

st.write("""#### Make a prediction""")

if st.button("⚡Click to predict⚡", key=None, help="On click makes a prediction with the inputed parameters"):
    prediction = model1.predict(input_df)[0]
    #st.write('The Predict energy is {0:.2f} Wh'.format(prediction))
    st.write('The Predict energy is {0:.2f} KWh'.format((prediction)/1000))
    st.write("""As a comparison the electricity consumption of private households by household size for 2018 on a daily basis:\n
    - One-person household - 1,943 KWh \n
    - Two-person household - 3,221 KWh \n
    - Three-person household - 4,978 KWh \n
    🔹 This includes electricity for heating, hot water (for hygiene tasks), lighting and electrical appliances.""")

    st.write("Source: [Destatis](https://www.destatis.de/EN/Themes/Society-Environment/Environment/Material-Energy-Flows/Tables/electricity-consumption-households.html)")
    
from PIL import Image
image = Image.open('Streamlit_APP/img/PvParameter-removebg-preview.png')

st.image(image, caption='Some solar parameters')
