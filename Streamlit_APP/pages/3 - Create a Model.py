from PIL import Image
from requests import options
import streamlit as st

st.title("Pv energy Prediction")
 
st.write("""
### Project description
Please select the variables to create a machine learning for a particular site""")


import pvlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import pandas as pd

pv_dfbase = pd.read_csv("Streamlit_APP/data/pv_final_cleaned.csv") 
pv_dfbase.rename(columns={"IR(h)":"IR_h"}, inplace=True)
base_df = pd.read_csv("Streamlit_APP/data/pv_final_cleaned.csv") 

######## END OF PRE TRAINED MODELS

# Initiating the input variables
URL = 'https://re.jrc.ec.europa.eu/api/v5_2/'

st.write(""" #### Please write all inputs as a decimal number. """)

lat = st.number_input("Site latitude", value=52.5 ,step=0.01, key="lat", help="Site latitude in degrees (°)")
long = st.number_input("Site longitude", value=13.4 ,step=0.01, key="long", help="Site longitude in degrees (°)")
location_name = st.text_input('Location name', value="Berlin", key="location_name", help="The location name. For example a city or a country")
altitude = st.number_input("Site altitude", value=34.0 ,step=0.01, key="alt", help="Site altitude in meters (m)")
timezone = st.text_input('Location timezone in GMT +/- hrs format', value="Etc/GMT-1", help="To find your timezone: https://www.ibm.com/docs/en/cloudpakw3700/2.3.0.0?topic=SS6PD2_2.3.0/doc/psapsys_restapi/time_zone_list.html ")
pv_tilt = st.number_input("PV modules tilt angle", value=35.0 ,step=0.01, key="pv_tilt", help="Site longitude in degrees (°)")
loss = st.number_input("Sum of system losses, in percent", value=15.0 ,step=0.1, min_value=0.0, max_value=270.0, key="loss", help="Sum of system losses, in percent (%)")


surface_azimuth_angle = st.number_input("Azimuth angle of the module surface", value=0.0 ,step=0.1, min_value=0.0, max_value=270.0, key="surface_azimuth_angle", help=" Azimuth angle North=0, East=90, South=180, West=270")
peakpowervar = st.number_input("Nominal power of the PV system in KWp", value=20.0 ,step=1.0, key="peakpowervar", help="The nominal power is the nameplate capacity of the PV system")

temp_model_par_name= st.selectbox('Select a temperature model parameter', index= 0, 
        options=["open_rack_glass_glass", "close_mount_glass_glass","open_rack_glass_polymer","insulated_back_glass_polymer"],key="temp_model_par_name", help="For a reference of the models: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.sapm_cell.html")
pvtechmaterial= st.selectbox('Select the PV module technology',index= 0, 
        options=["crystSi", "CIS", "CdTe", "Unknown"], key="pvtechmaterial", help="The material in which the solar cells were built")
mountingtype = st.selectbox('Type of mounting of the PV modules',index= 0, 
        options=["free", "building"], key="mountingtype", help="Type of mounting of the PV modules. 'free' for free-standing and 'building' for building-integrated.")

inverter_name = st.selectbox('Select your inverter',index= 0, 
        options=["ABB__MICRO_0_25_I_OUTD_US_208__208V_",
        "AU_Optronics__PM060MA1_240__208V_", "Advanced_Energy_Industries__AE_50TX_208__208V_"
        "Yes!_Solar__ES5400__208V_"], key="inverter_name", help="Type of mounting of the PV modules. 'free' for free-standing and 'building' for building-integrated.")
 
module_name = st.selectbox('Select your PV module',index= 0, 
        options=['Advent_Solar_AS160___2006_', 'Advent_Solar_Ventura_210___2008_',
       'Advent_Solar_Ventura_215___2009_', 'Aleo_S03_160__2007__E__',
       'Aleo_S03_165__2007__E__', 'Aleo_S16_165__2007__E__',
       'Aleo_S16_170__2007__E__', 'Aleo_S16_175__2007__E__',
       'Aleo_S16_180__2007__E__', 'Aleo_S16_185__2007__E__',
       'AstroPower_AP_100___2001_', 'AstroPower_AP_100__2000__E__',
       'AstroPower_AP_110___2001_', 'AstroPower_AP_110__1999__E__',
       'AstroPower_AP_120___2001_', 'AstroPower_AP_120__1999__E__',
       'AstroPower_AP_1206___1998_', 'AstroPower_AP_130___2001_',
       'AstroPower_AP_130__2002__E__', 'AstroPower_AP_50___2001_',
       'AstroPower_AP_50__2000__E__', 'AstroPower_AP_65__1999__E__',
       'AstroPower_AP_75___2001_', 'AstroPower_AP_75___2003_',
       'AstroPower_AP_75__2003__E__'], 
       key="module_name", help="Type of mounting of the PV modules. 'free' for free-standing and 'building' for building-integrated.")

##Retrieving the data
coordinates = [
    (lat, long, location_name, altitude, timezone) 
]

weather = pvlib.iotools.get_pvgis_tmy(lat, long, map_variables=True)
pvlib_weather = weather[0]

#retriving data for Solar Position
tmys = []
solposdf = pd.DataFrame()

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')


module = sandia_modules[module_name]  
inverter = sapm_inverters[inverter_name]
temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][temp_model_par_name]

#creating a dict with the input parameters
system = {'module': module, 
          'inverter': inverter,
          'surface_azimuth': surface_azimuth_angle}

# Making the call to retrive data

for location in coordinates:
    latitude, longitude, name, altitude, timezone = location
    weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)[0]
    weather.index.name = "utc_time"
    tmys.append(weather)

for location, weather in zip(coordinates, tmys):
    latitude, longitude, name, altitude, timezone = location
    system['surface_tilt'] = pv_tilt # Tilt angle of the module surface. Up=0, horizon=90.
    solpos = pvlib.solarposition.get_solarposition(
        time=weather.index,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        temperature=weather["temp_air"],
        pressure=pvlib.atmosphere.alt2pres(altitude)
    )
    solposdf = solpos

#Obtaning data for irradiance parameters, inverter and modules parameters

for location, weather in zip(coordinates, tmys):
    latitude, longitude, name, altitude, timezone = location
    system['surface_tilt'] = pv_tilt,   # Tilt angle of the module surface. Up=0, horizon=90.
    map_variables=True,
    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    aoi = pvlib.irradiance.aoi(
        system['surface_tilt'],
        system['surface_azimuth'],
        solpos["apparent_zenith"],
        solpos["azimuth"],
    )
       
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        system['surface_tilt'],
        system['surface_azimuth'],
        solpos['apparent_zenith'],
        solpos['azimuth'],
        weather['dni'],
        weather['ghi'],
        weather['dhi'],
        dni_extra=dni_extra,
        model='haydavies',
    )
    cell_temperature = pvlib.temperature.sapm_cell(
        total_irradiance['poa_global'],
        weather["temp_air"],
        weather["wind_speed"],
        **temperature_model_parameters,
    )
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        total_irradiance['poa_direct'],
        total_irradiance['poa_diffuse'],
        am_abs,
        aoi,
        module,
    )
    dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
    ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)

# Converting all variables to DataFrames
dni_extra = dni_extra.to_frame(name="dni_PVlib")
airmass = airmass.to_frame(name="airmass")
am_abs = am_abs.to_frame(name="am_abs")
aoi = aoi.to_frame(name="aoi")
cell_temperature = cell_temperature.to_frame(name="cell_temperature")
effective_irradiance = effective_irradiance.to_frame(name="effective_irradiance")
ac = ac.to_frame(name="ac_current")

# Joining all in one big dataframe
pv_berlin = pvlib_weather.join([solposdf,aoi, 
                                total_irradiance, cell_temperature, 
                                effective_irradiance, ac, dc], how="left", sort =True)
  
# Obtaining the target value (power values) from PVGIS will be used. 
import io
import json
from pathlib import Path
import requests
from pvlib.iotools import read_epw, parse_epw
import warnings
from pvlib._deprecation import pvlibDeprecationWarning
import datetime

# Dictionary mapping PVGIS names to pvlib names
VARIABLE_MAP = {
    'G(h)': 'ghi',
    'Gb(n)': 'dni',
    'Gd(h)': 'dhi',
    'G(i)': 'poa_global',
    'Gb(i)': 'poa_direct',
    'Gd(i)': 'poa_sky_diffuse',
    'Gr(i)': 'poa_ground_diffuse',
    'H_sun': 'solar_elevation',
    'T2m': 'temp_air',
    'RH': 'relative_humidity',
    'SP': 'pressure',
    'WS10m': 'wind_speed',
    'WD10m': 'wind_direction',
}


def get_pvgis_hourly(latitude, longitude, start=None, end=None,
                     raddatabase=None, components=True,
                     surface_tilt=0, surface_azimuth=0,
                     outputformat='json',
                     usehorizon=True, userhorizon=None,
                     pvcalculation=False,
                     peakpower= peakpowervar, 
                     pvtechchoice= pvtechmaterial,
                     mountingplace= mountingtype, 
                     loss=0, 
                     trackingtype=0,
                     optimal_surface_tilt=False, optimalangles=False,
                     url=URL, map_variables=True, timeout=60):
        
    # noqa: E501
    # use requests to format the query string by passing params dictionary
    params = {'lat': latitude, 'lon': longitude, 'outputformat': outputformat,
              'angle': surface_tilt, 'aspect': surface_azimuth,
              'pvcalculation': int(pvcalculation),
              'pvtechchoice': pvtechchoice, 'mountingplace': mountingplace,
              'trackingtype': trackingtype, 'components': int(components),
              'usehorizon': int(usehorizon),
              'optimalangles': int(optimalangles),
              'optimalinclination': int(optimal_surface_tilt), 'loss': loss}
    # pvgis only takes 0 for False, and 1 for True, not strings
    if userhorizon is not None:
        params['userhorizon'] = ','.join(str(x) for x in userhorizon)
    if raddatabase is not None:
        params['raddatabase'] = raddatabase
    if start is not None:
        params['startyear'] = start if isinstance(start, int) else start.year
    if end is not None:
        params['endyear'] = end if isinstance(end, int) else end.year
    if peakpower is not None:
        params['peakpower'] = peakpower
    
    res = requests.get(url + 'seriescalc', params=params, timeout=timeout)   
    if not res.ok:
        try:
            err_msg = res.json()
        except Exception:
            res.raise_for_status()
        else:
            raise requests.HTTPError(err_msg['message'])

    return read_pvgis_hourly(io.StringIO(res.text), pvgis_format=outputformat,
                             map_variables=map_variables)

def _parse_pvgis_hourly_json(src, map_variables):
    inputs = src['inputs']
    metadata = src['meta']
    data = pd.DataFrame(src['outputs']['hourly'])
    data.index = pd.to_datetime(data['time'], format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time', axis=1)
    data = data.astype(dtype={'Int': 'int'})  # The 'Int' column to be integer
    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)
    return data, inputs, metadata


def _parse_pvgis_hourly_csv(src, map_variables):
    # The first 4 rows are latitude, longitude, elevation, radiation database
    inputs = {}
    # 'Latitude (decimal degrees): 45.000\r\n'
    inputs['latitude'] = float(src.readline().split(':')[1])
    # 'Longitude (decimal degrees): 8.000\r\n'
    inputs['longitude'] = float(src.readline().split(':')[1])
    # Elevation (m): 1389.0\r\n
    inputs['elevation'] = float(src.readline().split(':')[1])
    # 'Radiation database: \tPVGIS-SARAH\r\n'
    inputs['radiation_database'] = src.readline().split(':')[1].strip()
    # Parse through the remaining metadata section (the number of lines for
    # this section depends on the requested parameters)
    while True:
        line = src.readline()
        if line.startswith('time,'):              
            names = line.strip().split(',')
            break       
        elif line.strip() != '':
            inputs[line.split(':')[0]] = line.split(':')[1].strip()
        elif line == '':  # If end of file is reached
            raise ValueError('No data section was detected. File has probably '
                             'been modified since being downloaded from PVGIS')    
    data_lines = []
    while True:
        line = src.readline()
        if line.strip() == '':
            break
        else:
            data_lines.append(line.strip().split(','))
    data = pd.DataFrame(data_lines, columns=names)
    data.index = pd.to_datetime(data['time'], format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time', axis=1)
    if map_variables:
        data = data.rename(columns=VARIABLE_MAP) 
    data = data.astype(float).astype(dtype={'Int': 'int'})
    metadata = {}
    for line in src.readlines():
        if ':' in line:
            metadata[line.split(':')[0]] = line.split(':')[1].strip()
    return data, inputs, metadata

def read_pvgis_hourly(filename, pvgis_format=None, map_variables=True):
   
    # get the PVGIS outputformat
    if pvgis_format is None:
        outputformat = Path(filename).suffix[1:].lower()
    else:
        outputformat = pvgis_format

    if outputformat == 'json':
        try:
            src = json.load(filename)
        except AttributeError:  # str/path has no .read() attribute
            with open(str(filename), 'r') as fbuf:
                src = json.load(fbuf)
        return _parse_pvgis_hourly_json(src, map_variables=map_variables)

    if outputformat == 'csv':
        try:
            pvgis_data = _parse_pvgis_hourly_csv(
                filename, map_variables=map_variables)
        except AttributeError:  # str/path has no .read() attribute
            with open(str(filename), 'r') as fbuf:
                pvgis_data = _parse_pvgis_hourly_csv(
                    fbuf, map_variables=map_variables)
        return pvgis_data

    # raise exception if pvgis format isn't in ['csv', 'json']
    err_msg = (
        "pvgis format '{:s}' was unknown, must be either 'json' or 'csv'")\
        .format(outputformat)
    raise ValueError(err_msg)

#Inputs for PVGIS data retrieval
start=pd.Timestamp('2006-12-01')
end=pd.Timestamp('2016-01-31')
raddatabase='PVGIS-SARAH2'

data = get_pvgis_hourly(latitude = lat, 
                    longitude = long, 
                    start=start, 
                    end=end,
                    raddatabase=raddatabase,
                    components=True, #Maybe need to change for 1
                    surface_tilt = pv_tilt, #this var comes from getting solar position part
                    surface_azimuth=0, # 0=south For Noth Hemisphere
                    outputformat='csv',
                    usehorizon=True, #Include effects of horizon
                    userhorizon=None, #Optional user specified elevation of horizon in degrees - Will not use it 
                    pvcalculation=True,
                    peakpower= peakpowervar, # Source: https://www.berlin.de/umweltatlas/en/energy/solar-systems/continually-updated/map-description/ 
                    pvtechchoice='crystSi',
                    mountingplace='free',  #Type of mounting for PV system. Options of 'free' for free-standing 
                                           #and 'building' for building-integrated.
                    loss= loss, #considering a global loss of 30% section 11.2.2 Source: https://www.ise.fraunhofer.de/content/dam/ise/en/documents/publications/studies/recent-facts-about-photovoltaics-in-germany.pdf 
                    trackingtype=0, #Type of mounting 0=fixed
                    optimal_surface_tilt=True, #Calculate the optimum tilt angle.
                    optimalangles=False, #Calculate the optimum tilt and azimuth angles.                      
                    url=URL, 
                    map_variables=True, # When true, renames columns of the Dataframe to pvlib variable names
                    timeout=30) # Time in seconds to wait for server response before timeout
PVgis_data = data[0]

# Creating a variable to be able to join the data from PVGIS with the one from PVLib
pvgis_temp = (
        PVgis_data
        .reset_index()
        .assign(
            date = lambda x: x["time"].dt.date,
            m = lambda x: x["time"].dt.hour,
            ymdh = lambda x: x["date"].astype(str) + " " + x["m"].astype(str)
        )
    )
#Joining both dataframes
pv_berlintemp = (
pv_berlin
    .reset_index()
    .assign(
        date = lambda x: x["time(UTC)"].dt.date,
        m = lambda x: x["time(UTC)"].dt.hour,
        ymdh = lambda x: x["date"].astype(str) + " " + x["m"].astype(str)
            )

)
pv_final = pd.merge(pv_berlintemp,pvgis_temp[["ymdh","P"]], on="ymdh", how="left")

#creating a  meteorological season variable
pv_final.reset_index()
pv_final['season'] = pv_final["time(UTC)"].dt.month%12 // 3 + 1

#Renaming the target column
pv_final.rename(columns={"P":"Power(Wh)"}, inplace=True)

#Creating the ML model
# creating the day variable
pv_final['day'] = pv_final["time(UTC)"].dt.day
pv_final['weekday_name'] = pv_final["time(UTC)"].dt.day_name()
pv_final['month_name'] =pv_final["time(UTC)"].dt.month_name()
pv_final['year'] =pv_final["time(UTC)"].dt.year
# Now dropping the time(UTC) column
pv_final.drop(columns=["time(UTC)"], inplace=True)
# Removing the values = 0
pv_final_cleaned = pv_final.loc[~(pv_final["Power(Wh)"]==0)]
(pv_final_cleaned == 0).all()


#Creating a function to plot the Power values distribution
def Pred_graph(Y_search):
    plt.figure(figsize = (16,8))
    counts, bins = np.histogram(Y_search)
    ax = plt.hist(bins[:-1], bins, weights=counts)
    plt.ylabel('Number of Observations')
    plt.xlabel( "Power (Wh)")
    ax = ax
    plt.show
    return st.pyplot(fig=ax)

############## THE DOWNLOAD PART AND THE BUTTON TO RUN THE MODEL 

# The external inputs

st.write("\n")
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
    file_name='pv_base.csv',
    mime='text/csv'
)
##########

#####
## Upload
st.write("\n")
uploaded_file = st.file_uploader("📤Upload a '.csv' file to make a prediction📤", type=None, 
    key="uploaded_file", help="Please follow the base csv")
if uploaded_file is not None:
    upload_df = pd.read_csv(uploaded_file)
    st.write(upload_df)
else:
    print("Please provide a valid csv file")
#####

st.write("\n")
st.write("### 🔸Let's create a model and make a prediction🔸")

if st.button("⚡Create my Model⚡", key=None, help="On click makes a prediction with the inputed parameters"):
    if uploaded_file is not None:
        
        # Training the model
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split 

        y = upload_df["Power(Wh)"]
        X = upload_df
        X = upload_df.drop(columns="Power(Wh)")

        # data splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1990)

        # Tuning the RidgeCV model

        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.impute import KNNImputer
        from sklearn.pipeline import make_pipeline

        from sklearn.preprocessing import Normalizer
        from sklearn.preprocessing import MinMaxScaler

        X_cat_columns = X_train.select_dtypes(include="object").columns
        X_num_columns = X_train.select_dtypes(exclude="object").columns

        #Setting the imputers, Scaler 
        imputer = KNNImputer()
        scaler = MinMaxScaler(feature_range=(0, 10))

        # create numerical pipeline
        numeric_pipe = make_pipeline(imputer,
                        scaler)
                        
        # create categorical pipeline, with the SimpleImputer(fill_value="N_A") and the OneHotEncoder
        categoric_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="N_A"),
            OneHotEncoder()
            )   

        from sklearn.compose import ColumnTransformer

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipe", numeric_pipe, X_num_columns),
                ("cat_pipe", categoric_pipe, X_cat_columns)  
                ]
            )

        from sklearn.model_selection import RandomizedSearchCV
        #from sklearn.model_selection import GridSearchCV

        full_pipeline = make_pipeline(preprocessor, 
                                    RidgeCV())

        param_grid = {    
            "columntransformer__cat_pipe__onehotencoder__handle_unknown" : ["ignore"],
            "ridgecv__alphas":(0.1, 2.0, 20.0),    
            "ridgecv__gcv_mode":[None, "auto", "svd", "eigen"],
            "ridgecv__fit_intercept":[True,False],
            }
    
        lm_search = RandomizedSearchCV(full_pipeline,
                        param_grid,
                        cv=10,
                        n_jobs=-1,
                        verbose=1)
        lm_search.fit(X_train, y_train)

        y_predict = lm_search.predict(X_test)
        
        prediction = lm_search.predict(upload_df)/1000
        y_predict = pd.DataFrame(data=prediction.round(2))
        y_predict.rename(columns={0:"Predicted Energy (KWh)"}, inplace=True)
        st.write("Predicted energy(KWh) X Days")
        st.bar_chart(data=y_predict)
    else:
        print("Please provide a valid csv file")  
    
    


   
